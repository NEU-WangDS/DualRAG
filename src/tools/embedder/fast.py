import os
import numpy as np
from openai.types import CreateEmbeddingResponse
import requests
import aiohttp
from tenacity import retry, stop_never, wait_random_exponential

import time
from cr_utils import CostManagers

class FastApiEmbedder:
    @staticmethod
    def create_embedding(corpus: list[str], batch_size=1) -> np.ndarray:
        @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
        def ask_embedding(corpus: list[str]):
            response = requests.post(f"{os.getenv('fastapi_embed')}/create_embedding/", json=corpus, proxies=None)
            response = CreateEmbeddingResponse(**response.json())
            return response

        response_embs = []
        for start_idx in range(0, len(corpus), batch_size):
            batch_data = corpus[start_idx:start_idx+batch_size]
            batch_embs: CreateEmbeddingResponse = ask_embedding(batch_data)
            response_embs += [emb.embedding for emb in batch_embs.data]
        response_embs = np.array(response_embs)
        return response_embs

    @staticmethod
    async def acreate_embedding(corpus: list[str], batch_size=1) -> np.ndarray:
        @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
        async def aask_embedding(corpus: list[str]):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{os.getenv('fastapi_embed')}/create_embedding/", json=corpus) as response:
                    data = await response.json()
                    response = CreateEmbeddingResponse(**data)
                    return response

        response_embs = []
        for start_idx in range(0, len(corpus), batch_size):
            batch_data = corpus[start_idx:start_idx+batch_size]
            batch_embs: CreateEmbeddingResponse = await aask_embedding(batch_data)
            response_embs += [emb.embedding for emb in batch_embs.data]
        response_embs = np.array(response_embs)
        return response_embs

    @staticmethod
    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def get_dim():
        response = requests.get(f"{os.getenv('fastapi_embed')}/get_dim/")
        return response.json()

    @staticmethod
    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def get_max_seq_length():
        response = requests.get(f"{os.getenv('fastapi_embed')}/get_max_seq_length/")
        return response.json()


@retry(stop=stop_never, wait=wait_random_exponential(multiplier=1, min=1, max=10))
async def aencode(texts: list[str]) -> np.ndarray:
    """
    异步获取文本的 Embedding 向量矩阵。
    封装了 FastApiEmbedder，并加入了 CostManagers 时间开销追踪，
    与 aner 和 aretrieve 保持生态一致。
    """
    start = time.time()
    # 默认使用较大的 batch_size 加速并发，可根据显存自行调整
    embs = await FastApiEmbedder.acreate_embedding(texts, batch_size=16) 
    rsp_time = time.time() - start
    # 记录开销日志
    CostManagers().update_cost(0, 0, 0, rsp_time, "tool_embedder")
    return embs