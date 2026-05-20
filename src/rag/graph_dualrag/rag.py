import json
import logging
import asyncio
from omegaconf import DictConfig
from tqdm.asyncio import tqdm
import numpy as np
import torch

from cr_utils import Logger
from src.dataset import Item
from src.rag.base import QA
from src.startup import RAGBuilder
from src.tools.agent.llm import LLMAgent
from src.tools.retriever.fast import aretrieve
from src.tools.ner.fast import aner
from src.tools.embedder.fast import aencode
# 复用原仓库中优秀的多智能体迭代引擎
from src.rag.duralrag.rag import Infer, KManager, Knowledge

# 导入本文核心图谱算法模块
from .heg import HeterogeneousEvidenceGraph
from .graphsage import RWRWalker, TopologyFeatureAggregator

log = logging.getLogger(__name__)

prompt_system1 = """请基于以下背景信息，简明直接地回答问题。

背景信息:
{docs}

问题: {question}
答案:"""

prompt_system2 = """你是一个具备高级逻辑推演能力的学术问答助手。
请基于以下经过异构证据图（HEG）拓扑寻路以及 GraphSAGE 重校准后的背景信息，进行长程跨文档多跳逻辑推理，给出最终的严密答案。
提示：背景语料已根据其拓扑增益得分进行了 U 型重排，首尾段落包含最重要的关联线索。请仔细分析各节点间的实体路径。

大模型先验推理链路 (Thought):
{thought}

重校准后的背景信息库:
{docs}

用户多跳提问: {question}
请输出详细的推理链并给出最终答案:"""


@RAGBuilder.register_module("graph_dualrag")
class GraphDualRAG(QA):
    def __init__(self, cfg: DictConfig, logger: Logger):
        super().__init__(cfg, logger)
        self.cfg = cfg
        method_cfg = self.cfg.task.method
        
        # 加载论文 4.2 节中的启发式路由参数
        self.tau_route = method_cfg.get('route_threshold', 0.6)
        self.alpha = method_cfg.get('alpha', 0.5)
        self.beta = method_cfg.get('beta', 0.5)
        self.gamma = method_cfg.get('gamma', 0.6)
        self.topk = method_cfg.get('retrieve_topk', 5)
        
        # 初始化双过程生成 Agent
        self.agent_sys1 = LLMAgent("system1", prompt_system1, cfg.task.base_llm)
        self.agent_sys2 = LLMAgent("system2", prompt_system2, cfg.task.base_llm)
        
        # 实例化多智能体大脑 (复用原本的 Infer 引擎)
        self.infer = Infer(cfg)
        
        # 初始化图计算核心组件
        self.rwr_walker = RWRWalker(
            restart_prob=method_cfg.get('rwr_restart_prob', 0.3),
            max_iters=method_cfg.get('rwr_max_iters', 100)
        )
        self.topology_aggregator = TopologyFeatureAggregator(
            hidden_dim=method_cfg.get('embedding_dim', 384), 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    async def aquery(self, data: Item):
        log_dir = f"log/{data.id}/llm"
        self.logger.mkdir(log_dir)
        trace_log = {}

        # 阶段 1: 异步启发式路由计算 Φ(Q)
        score_phi, init_docs, init_scores, init_idxs, query_entities = await self._aheuristic_routing(data.question)
        trace_log["router"] = {"phi_score": score_phi, "query_entities": query_entities}

        # 阶段 2: 路由分流
        if score_phi < self.tau_route:
            log.info(f"[{data.id}] 触发系统 1 (快思考), Φ(Q)={score_phi:.2f}")
            trace_log["decision"] = "System 1"
            rsp = await self._asystem1_fast_response(data.id, data.question, init_docs)
            return rsp, trace_log
        else:
            log.info(f"[{data.id}] 触发系统 2 (慢思考), 启动图谱拓扑迭代, Φ(Q)={score_phi:.2f}")
            trace_log["decision"] = "System 2"
            
            # 将初始召回数据转为 Chunk 格式
            initial_chunks = [{"chunk_id": idx, "text": doc} for doc, idx in zip(init_docs, init_idxs)]
            
            rsp, sys2_trace = await self._asystem2_slow_reasoning(data, initial_chunks, query_entities)
            trace_log.update(sys2_trace)
            return rsp, trace_log

    async def _aheuristic_routing(self, query: str):
        """实现 4.2 节：混合复杂度打分函数 Φ(Q)"""
        ner_task = aner(query)
        retrieve_task = aretrieve(self.cfg.corpus, query, self.topk)
        extracted_entities, retrieve_res = await asyncio.gather(ner_task, retrieve_task)
        
        q_len = max(len(query), 1)
        d_ent = min(1.0, len(extracted_entities) / (q_len * 0.1))
        
        docs, scores, idxs = retrieve_res
        c_ret = float(scores[0]) if scores else 0.0
            
        phi = self.alpha * d_ent + self.beta * (1.0 - c_ret)
        return phi, docs, scores, idxs, extracted_entities

    async def _asystem1_fast_response(self, task_id: str, query: str, docs: list[str]) -> str:
        context_str = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(docs)])
        return await self.agent_sys1.arun(task_id, placeholder={"docs": context_str, "question": query})

    async def _asystem2_slow_reasoning(self, data: Item, initial_chunks: list[dict], query_entities: list[str]):
        """
        核心系统 2：【迭代式实体探寻】 + 【非破坏性图谱特征重排】
        """
        sys2_trace = {}
        kmanager = KManager(self.cfg)
        global_chunks_dict = {c['chunk_id']: c for c in initial_chunks} # 用于存储探索到的所有去重文档
        
        # =========================================================
        # 步骤 A: 启发式多跳实体游走与文档收集 (摒弃旧版的 KS 摘要破坏)
        # =========================================================
        with tqdm(range(self.cfg.task.method.max_iter), leave=False, desc=f"图谱探索 {data.id}") as tbar:
            for step in tbar:
                # 1. 模拟大脑思考: 还需要继续多跳吗？
                thought_new, need_retrieve, _ = await self.infer.ainfer(data.id, placeholder={
                    "knowledge": Knowledge.dict2str(kmanager.kb), # 旧版遗留接口兼容
                    "question": data.question,
                    "thought": "\n".join(kmanager.thought),
                })
                kmanager.thought.append(thought_new)
                
                if not need_retrieve:
                    break # 知识已经足够，停止探索
                
                # 2. 意图提取: 下一跳搜什么？
                tbar.set_postfix(status="EI 实体提取")
                entity2key = await kmanager.aei(data)
                
                # 3. 执行检索并将原文档收集入图谱缓冲池，而不进行摘要破坏！
                tbar.set_postfix(status="Retrieve 扩充图谱")
                for entity, keywords in entity2key.items():
                    for keyword in keywords:
                        docs, scores, idxs = await aretrieve(self.cfg.corpus, keyword, self.topk)
                        for idx, doc in zip(idxs, docs):
                            if idx not in global_chunks_dict:
                                global_chunks_dict[idx] = {"chunk_id": idx, "text": doc}

        # 获取探索到的全量非结构化文档池
        all_chunks = list(global_chunks_dict.values())
        sys2_trace["total_explored_chunks"] = len(all_chunks)

        # =========================================================
        # 步骤 B: 内存动态图谱实例化 (第三章 HEG 物理构建)
        # =========================================================
        chunk_texts = [c['text'] for c in all_chunks]
        
        # 并发执行 NER 探针
        ner_tasks = [aner(text) for text in chunk_texts]
        chunk_entities_batch = await asyncio.gather(*ner_tasks)
        for i, chunk in enumerate(all_chunks):
            chunk['entities'] = [{'entity': e} for e in chunk_entities_batch[i]]
            
        # 获取 Embedding 
        chunk_embeddings = await aencode(chunk_texts)

        heg_builder = HeterogeneousEvidenceGraph()
        graph = heg_builder.build_graph(all_chunks, chunk_embeddings, sim_threshold=self.cfg.task.method.get('sim_threshold', 0.75))

        # =========================================================
        # 步骤 C: RWR 游走与 GraphSAGE 特征重校准 (第四章核心)
        # =========================================================
        # 提取 LLM 思考过程中的所有意图实体作为种子节点
        expanded_query_entities = query_entities + [e for k in kmanager.kb.keys() for e in kmanager.kb[k].contents]
        seed_nodes = heg_builder.get_seed_nodes_from_query(expanded_query_entities)
        
        rwr_scores = self.rwr_walker.compute_rwr(graph, seed_nodes) if seed_nodes else {}
        topology_features = self.topology_aggregator.aggregate_features(graph, chunk_embeddings)
        query_emb = (await aencode([data.question]))[0]

        # =========================================================
        # 步骤 D: U 型注意力重排与生成 (解决 Lost in the Middle)
        # =========================================================
        composite_scores = []
        for chunk in all_chunks:
            c_id = chunk['chunk_id']
            p_star = rwr_scores.get(c_id, 0.0)
            h_v = topology_features.get(c_id, None)
            
            sem_score = 0.0
            if h_v is not None and np.linalg.norm(h_v) > 0:
                sem_score = np.dot(query_emb, h_v) / (np.linalg.norm(query_emb) * np.linalg.norm(h_v))
                
            score_ci = self.gamma * p_star + (1.0 - self.gamma) * sem_score
            composite_scores.append((score_ci, chunk))
            
        # 根据拓扑得分降序，并执行 U 型非破坏性重排
        composite_scores.sort(key=lambda x: x[0], reverse=True)
        ranked_chunks = [item[1] for item in composite_scores]

        #保存未经 U 型重排的严格降序列表，专供评测器算 IR 指标！
        sys2_trace["ranked_docs"] = [c['text'] for c in ranked_chunks]

        reordered_chunks = self._u_shape_reorder(ranked_chunks)

        sys2_trace["reordered_docs"] = [c['text'] for c in reordered_chunks]

        # 组装最终提示词
        context_blocks = []
        # 为了防止 Token 超载，仅取重排后的前 N 个核心文档
        for c in reordered_chunks[:self.cfg.task.method.get('final_topk', 15)]:
            ent_str = ", ".join([e['entity'] for e in c.get('entities', [])])
            context_blocks.append(f"<path> ID: {c['chunk_id']} | 拓扑实体: [{ent_str}] </path>\n{c['text']}")
            
        context_str = "\n\n=============\n\n".join(context_blocks)
        
        # 传入全量探索出的先验思考 (Thought) 和经过图谱打分的真实原文档进行闭环生成
        rsp = await self.agent_sys2.arun(data.id, placeholder={
            "thought": "\n".join(kmanager.thought),
            "docs": context_str,
            "question": data.question
        })
        
        sys2_trace["final_thought"] = kmanager.thought
        return rsp, sys2_trace

    def _u_shape_reorder(self, ranked_items: list) -> list:
        if not ranked_items: return []
        reordered = []
        for i in range(0, len(ranked_items), 2):
            reordered.append(ranked_items[i])
        start_idx = len(ranked_items) - 1 - (len(ranked_items) % 2)
        for i in range(start_idx, 0, -2):
            reordered.append(ranked_items[i])
        return reordered