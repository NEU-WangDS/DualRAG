import sys
import os
from typing import Callable, Awaitable, TypeVar, ParamSpec
from functools import partial
import dotenv
import time
import pickle as pkl
import numpy as np
import faiss
from pydantic import BaseModel
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf

sys.path.append("../../")
dotenv.load_dotenv("../../.env")
from src.tools.embedder import FastApiEmbedder

P = ParamSpec('P')
T = TypeVar("T")

def make_async(func: Callable[P, T], executor: ThreadPoolExecutor | None = None) -> Callable[P, Awaitable[T]]:
    """使用线程池执行阻塞函数，防止阻塞 FastAPI 的主事件循环"""
    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)
    return _async_wrapper

# 动态加载配置文件，对齐 AutoDL 数据盘路径
cfg_path = "../../config/main.yaml"
cfg = OmegaConf.load(cfg_path)
workspace = cfg.workspace

app = FastAPI()
embedder = FastApiEmbedder()

# 初始化专门用于 Faiss CPU 查询的线程池
faiss_executor = ThreadPoolExecutor(max_workers=4) 

class DenseRetriever:
    def __init__(self, embedder: FastApiEmbedder, index_path: str, corpus_path: str, topk=10):
        self.embedder = embedder
        
        print(f"Read index from: {index_path}")
        start = time.time()
        
        # 直接使用 CPU 读取索引，不再执行任何 faiss.index_cpu_to_gpu 相关的操作
        self.index = faiss.read_index(index_path)
        print(f"Read index time: {time.time() - start:.2f}s (CPU Mode)")
            
        print(f"Read corpus from: {corpus_path}")
        start = time.time()
        with open(corpus_path, "rb") as f:
            self.corpus: list[str] = pkl.load(f)
        print(f"Read corpus time: {time.time() - start:.2f}s")
        self.topk = topk

    # 将阻塞的 CPU search 操作剥离出来，以便被线程池调用
    def _sync_search(self, query_emb: np.ndarray, topk: int):
        return self.index.search(query_emb, topk)

    async def retrieve(self, query: str, topk: int = None):
        topk = self.topk if topk is None else topk
        
        # 1. 异步获取 Embedding
        query_emb: np.ndarray = await self.embedder.acreate_embedding([query])
        query_emb = query_emb.astype(np.float32)
        
        # 2. 将同步的 CPU 检索操作丢入线程池，释放主线程处理其他并发请求
        async_search = make_async(self._sync_search, faiss_executor)
        scores, idxs = await async_search(query_emb, topk)
        
        scores, idxs = scores[0], idxs[0]
        docs = [self.corpus[idx] for idx in idxs]
        return docs, scores.tolist(), idxs.tolist()


# 定义需要加载的语料库列表
corpus_li = ["hotpotqa", "2wikimultihopqa", "musique"] 
retrievers = {}

print("Initializing Retrievers...")
for c in corpus_li:
    c_pkl = os.path.join(workspace, f"{c}.pkl")
    c_idx = os.path.join(workspace, f"{c}.index")
    
    # 安全加载机制：只加载数据盘中实际存在的文件，避免 FileNotFoundError
    if os.path.exists(c_idx) and os.path.exists(c_pkl):
        retrievers[c] = DenseRetriever(embedder, c_idx, c_pkl, topk=10)
        print(f"[Success] Loaded {c} onto CPU.")
    else:
        print(f"[Warning] Skip {c}: Missing index or pkl at {workspace}")


class RetrieverRequest(BaseModel):
    source: str
    query: str
    topk: int = 10

@app.post("/retrieve/")
async def retrieve(request: Request):
    body = await request.json()
    params = RetrieverRequest(**body)
    
    source = "hotpotqa" if params.source == "wiki" else params.source
    
    # 增加前置校验，避免 KeyError
    if source not in retrievers:
        raise HTTPException(status_code=400, detail=f"Corpus '{source}' not loaded or missing.")
        
    docs, scores, idxs = await retrievers[source].retrieve(params.query, topk=params.topk)
    ret = {
        "idxs": idxs,
        "docs": docs,
        "scores": scores,
    }
    return StreamingResponse(
        BytesIO(json.dumps(ret).encode('utf-8')),
        media_type="application/json",
    )

@app.get("/corpus_len/")
async def corpus_len():
    return {k: len(v.corpus) for k, v in retrievers.items()}