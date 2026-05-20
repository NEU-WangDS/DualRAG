import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class HeterogeneousEvidenceGraph:
    """
    异构证据图 (HEG) 动态构建模块
    """
    def __init__(self):
        # 采用无向图，保证后续 RWR 游走时的信号双向可达性
        self.graph = nx.Graph()
        
    def build_graph(self, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, sim_threshold: float = 0.75):
        """
        端到端的一键建图流水线
        :param chunks: 包含 chunk_id, text, entities 的字典列表
        :param chunk_embeddings: 由 Embedder 微服务返回的稠密向量矩阵
        :param sim_threshold: 语义相似度建图阈值 \theta_{sim}
        """
        self._add_chunk_nodes(chunks)
        self._add_entity_nodes_and_containment(chunks)
        self._build_co_occurrence_edges(chunks)
        self._build_semantic_edges(chunks, chunk_embeddings, sim_threshold)
        
        print(f"[HEG Builder] 图谱实例化完成. 节点数: {self.graph.number_of_nodes()}, "
              f"边数: {self.graph.number_of_edges()}")
        return self.graph

    def _add_chunk_nodes(self, chunks: List[Dict[str, Any]]):
        """1. 添加文本块节点"""
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            # 添加节点时注入元数据，区分 node_type
            self.graph.add_node(
                chunk_id, 
                node_type='chunk', 
                text=chunk.get('text', '')
            )

    def _add_entity_nodes_and_containment(self, chunks: List[Dict[str, Any]]):
        """2. 添加实体节点，并构建包含边 r_contain"""
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            entities = chunk.get('entities', [])
            
            for ent in entities:
                ent_name = ent['entity']  # 实体表面词
                ent_id = f"ENT_{ent_name}" # 加上前缀防止与 chunk_id 冲突
                
                # 如果实体不存在，则新增实体节点
                if not self.graph.has_node(ent_id):
                    self.graph.add_node(
                        ent_id, 
                        node_type='entity', 
                        name=ent_name
                    )
                
                # 构建 r_contain 边 (Chunk <-> Entity)
                self.graph.add_edge(
                    chunk_id, 
                    ent_id, 
                    relation='r_contain', 
                    weight=1.0  # 精确包含关系，权重设为 1.0
                )

    def _build_co_occurrence_edges(self, chunks: List[Dict[str, Any]]):
        """3. 构建实体共现边 r_co-occur"""
        for chunk in chunks:
            entities = chunk.get('entities', [])
            # 提取同一 chunk 内的所有唯一实体 ID
            ent_ids = list(set([f"ENT_{ent['entity']}" for ent in entities]))
            
            # 对同一 chunk 内的实体进行两两全连接
            for i in range(len(ent_ids)):
                for j in range(i + 1, len(ent_ids)):
                    u, v = ent_ids[i], ent_ids[j]
                    
                    if self.graph.has_edge(u, v):
                        # 如果已存在共现边，则增加共现频次权重
                        self.graph[u][v]['weight'] += 0.5 
                    else:
                        # 新建共现边
                        self.graph.add_edge(
                            u, 
                            v, 
                            relation='r_co-occur', 
                            weight=1.0
                        )

    def _build_semantic_edges(self, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, threshold: float):
        """4. 构建粗粒度语义关联边 r_sim (GPU 矩阵乘法模拟)"""
        num_chunks = len(chunks)
        if num_chunks < 2 or chunk_embeddings is None:
            return

        # 计算余弦相似度矩阵 (对应论文中 S = Norm(M_emb) x Norm(M_emb)^T)
        sim_matrix = cosine_similarity(chunk_embeddings)
        
        # 遍历上三角矩阵，提取大于阈值的边
        for i in range(num_chunks):
            for j in range(i + 1, num_chunks):
                sim_score = sim_matrix[i][j]
                if sim_score >= threshold:
                    u_id = chunks[i]['chunk_id']
                    v_id = chunks[j]['chunk_id']
                    
                    self.graph.add_edge(
                        u_id, 
                        v_id, 
                        relation='r_sim', 
                        weight=float(sim_score) # 权重设为实际的相似度得分
                    )

    def get_seed_nodes_from_query(self, query_entities: List[str]) -> List[str]:
        """工具方法：根据用户的 Query 实体，从图中找出对应的种子节点 ID"""
        seed_nodes = []
        for q_ent in query_entities:
            ent_id = f"ENT_{q_ent}"
            if self.graph.has_node(ent_id):
                seed_nodes.append(ent_id)
        return seed_nodes