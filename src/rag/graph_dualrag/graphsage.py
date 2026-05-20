import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

# =====================================================================
# 模块 1：基于马尔可夫迭代的带重启偏置随机游走 (RWR)
# 对应论文 4.3.1 节：p^(t+1) = (1 - c) W^T p^(t) + c p^(0)
# =====================================================================
class RWRWalker:
    def __init__(self, restart_prob: float = 0.3, max_iters: int = 100, tol: float = 1e-6):
        self.c = restart_prob      # 重启概率 (论文中的 c)
        self.max_iters = max_iters # 最大迭代次数
        self.tol = tol             # 收敛容忍度 (\epsilon)

    def compute_rwr(self, graph: nx.Graph, seed_nodes: List[str]) -> Dict[str, float]:
        """计算 RWR 稳态分布得分"""
        nodes = list(graph.nodes())
        node_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        if n == 0 or not seed_nodes:
            return {node: 0.0 for node in nodes}

        # 1. 构建邻接矩阵 A (提取边权重)
        A = nx.to_numpy_array(graph, nodelist=nodes, weight='weight')
        
        # 2. 构建归一化转移概率矩阵 W (按行归一化)
        row_sums = A.sum(axis=1)
        # 防止除以 0 的情况
        row_sums[row_sums == 0] = 1.0 
        W = A / row_sums[:, np.newaxis]
        
        # 3. 初始化独热种子向量 p^(0)
        p_0 = np.zeros(n)
        valid_seeds = [seed for seed in seed_nodes if seed in node_idx]
        if not valid_seeds:
            return {node: 1.0/n for node in nodes} # 退化为均匀分布
            
        for seed in valid_seeds:
            p_0[node_idx[seed]] = 1.0 / len(valid_seeds)
            
        # 4. 马尔可夫链幂法迭代 (Power Iteration)
        p_t = np.copy(p_0)
        W_T = W.T # 矩阵转置 W^T
        
        for i in range(self.max_iters):
            # 核心公式: p^(t+1) = (1 - c) W^T p^(t) + c p^(0)
            p_next = (1 - self.c) * np.dot(W_T, p_t) + self.c * p_0
            
            # 收敛检测 (L1 范数)
            if np.linalg.norm(p_next - p_t, ord=1) < self.tol:
                print(f"[RWR Walker] 拓扑游走在第 {i+1} 轮收敛.")
                p_t = p_next
                break
            p_t = p_next
            
        # 5. 映射回节点 ID
        return {nodes[i]: float(p_t[i]) for i in range(n)}


# =====================================================================
# 模块 2：轻量级归纳式图神经网络 (GraphSAGE)
# 对应论文 4.3.2 节：基于局部邻域的特征聚合与拼接
# =====================================================================
class SAGELayer(nn.Module):
    """单层 GraphSAGE 聚合算子"""
    def __init__(self, in_features: int, out_features: int):
        super(SAGELayer, self).__init__()
        # W 权重矩阵，输入维度是聚合特征(in) + 自身特征(in) = 2 * in
        self.W = nn.Linear(in_features * 2, out_features)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        :param h: 节点特征矩阵 [N, d]
        :param adj: 归一化后的邻接矩阵 [N, N]
        """
        # 1. 均值聚合器 (Mean Aggregator): h_N(v) = Mean({h_u})
        # 通过与归一化邻接矩阵相乘，瞬间实现全图节点的局部邻域均值池化
        h_neighbor = torch.matmul(adj, h)
        
        # 2. 特征拼接 (Concatenation): [h_v || h_N(v)]
        h_concat = torch.cat([h, h_neighbor], dim=-1)
        
        # 3. 非线性激活: \sigma(W * h_concat)
        h_new = F.relu(self.W(h_concat))
        
        # 4. L2 归一化 (防止多跳聚合后特征爆炸)
        return F.normalize(h_new, p=2, dim=-1)


class LightweightGraphSAGE(nn.Module):
    """两阶 GraphSAGE 网络模型"""
    def __init__(self, hidden_dim: int):
        super(LightweightGraphSAGE, self).__init__()
        # 两阶邻域采样，对应图 4-3 中的层 1 和层 2
        self.layer1 = SAGELayer(hidden_dim, hidden_dim)
        self.layer2 = SAGELayer(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(h, adj)
        h2 = self.layer2(h1, adj)
        return h2


# =====================================================================
# 模块 3：拓扑特征融合引擎 (桥接 HEG 与 PyTorch)
# =====================================================================
class TopologyFeatureAggregator:
    def __init__(self, hidden_dim: int = 768, device: str = 'cpu'):
        self.device = torch.device(device)
        # 初始化免训练/可微调的 GraphSAGE 模块
        self.sage_model = LightweightGraphSAGE(hidden_dim).to(self.device)
        self.sage_model.eval() # 默认开启评估模式，用于推理阶段

    @torch.no_grad()
    def aggregate_features(self, graph: nx.Graph, chunk_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将离散的图结构与特征张量结合，输出融合了多跳拓扑信息的新嵌入
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        dim = chunk_embeddings.shape[1]
        
        # 1. 初始化全图特征张量矩阵 h^(0)
        h_0 = np.zeros((n, dim))
        chunk_idx = 0
        
        # 针对异构节点的精妙初始化：
        # - Text Chunk 节点使用真实的大模型 Embedding
        # - Entity 节点初始化为 0，后续通过图卷积吸收 Chunk 的语义
        for i, node in enumerate(nodes):
            if graph.nodes[node].get('node_type') == 'chunk':
                if chunk_idx < len(chunk_embeddings):
                    h_0[i] = chunk_embeddings[chunk_idx]
                    chunk_idx += 1
                    
        h_tensor = torch.FloatTensor(h_0).to(self.device)

        # 2. 构建 PyTorch 友好的归一化拉普拉斯/邻接矩阵
        A = nx.to_numpy_array(graph, nodelist=nodes, weight='weight')
        row_sums = A.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        adj_normalized = A / row_sums[:, np.newaxis]
        adj_tensor = torch.FloatTensor(adj_normalized).to(self.device)

        # 3. GraphSAGE 前向传播计算
        h_final = self.sage_model(h_tensor, adj_tensor)
        
        # 4. 转回 NumPy 并建立映射
        h_final_np = h_final.cpu().numpy()
        
        # 仅返回 Chunk 节点的新特征供后续 RAG 重排使用
        final_features = {}
        for i, node in enumerate(nodes):
            if graph.nodes[node].get('node_type') == 'chunk':
                final_features[node] = h_final_np[i]
                
        return final_features