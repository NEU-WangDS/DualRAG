import sys
sys.path.append(".")
import os
import json
import pandas as pd

from scripts.logs import logs
from src.corpus import  hash_object
from src.evaluator import  retrieval_metrics

dataset = "hotpotqa" # "nq", "eli5", "asqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle", "strategyqa"
print(f"dataset: {dataset}")

logs = {
    "hotpotqa": {
        # "native": ["0521_165818-native"],
        #"graph_dualrag": ["0521_184631-graph_dualrag"]
        #"graph_dualrag": ["0521_184738-graph_dualrag"]
        #"graph_dualrag": ["0521_184824-graph_dualrag"]
        #"graph_dualrag": ["0521_184920-graph_dualrag"]
        #"graph_dualrag": ["0521_185008-graph_dualrag"]
        #"graph_dualrag": ["0521_185059-graph_dualrag"]
        #"graph_dualrag": ["0521_185149-graph_dualrag"]
        "graph_dualrag": ["0521_185241-graph_dualrag"]  
    }
}

#k_values = [3, 10, 50, 200]
k_values = [5, 10, 20]
# methods = [
#     "native",
#     "ircot",
#     "hrag",
#     "hrag-mem",
#     "lazykrag",
# ]
methods = [
    # "native",
    "graph_dualrag"
]


def get_docs_retrieved(trace: dict, method: str) -> list[str]:
    docs = set()
    if "native" in method:
        # 🚨 修改：直接读取纯 Faiss 的 retrieve 结果，不读 rerank！
        retrieve = trace["retrieve"]
        # 根据 Faiss 分数进行严格降序排序，保证公平
        sorted_docs = sorted(
            retrieve["docs"].items(), 
            key=lambda x: retrieve["scores"][x[0]], 
            reverse=True
        )
        return [doc for doc_id, doc in sorted_docs]
    # if "native" in method:
    #     rerank = trace["rerank"]
    #     docs.update(set([doc for doc_id, doc in rerank["docs"].items() if rerank["scores"][doc_id] > 0]))
    elif "ircot" in method:
        for i in trace:
            rerank = trace[i]["rerank"] if "rerank" in trace[i] else trace[i]["retrieve"]["rerank"]
            docs.update(set([doc for doc_id, doc in rerank["docs"].items() if rerank["scores"][doc_id] > 0]))
    elif "hrag" in method:
        for i in trace["trace"]:
            if "learn" in trace["trace"][i]:
                docs.update(set([doc for v in trace["trace"][i]["learn"].values() for doc in v["new_docs"].values()]))
    elif "lazykrag" in method:
        docs.update(set([doc for doc in trace["docs"].values()]))
    elif "graph_dualrag" in method:
        # 优先读取严格降序的 ranked_docs
        doc_list = trace.get("ranked_docs", trace.get("reordered_docs", trace.get("system1_docs", [])))
        
        ordered_docs = []
        seen = set()
        for d in doc_list:
            if d not in seen:
                seen.add(d)
                ordered_docs.append(d)
        # 直接返回 ordered_docs，千万不能过 list(set())！
        return ordered_docs
    else:
        raise NotImplementedError
    return list(docs)


def eval_one(log_filepath: str, method: str):
    with open(log_filepath) as f:
        log_content = json.load(f)
        content_key = "sentences" if dataset == "hotpotqa" else "content"

        titles: list[str] = log_content["metadata"]["context"]["title"]
        contents: list[list[str]] = log_content["metadata"]["context"][content_key]

        # doc_id = [titles.index(title) for title in log_content["metadata"]["supporting_facts"]["title"]]  # 有效 doc id
        # sent_id = log_content["metadata"]["supporting_facts"]["sent_id"]                                  # 有效 sentence id

        doc_id = [titles.index(title) for title in log_content["metadata"]["supporting_facts"]["title"] if title in titles]
        sent_id = log_content["metadata"]["supporting_facts"]["sent_id"]

        support_titles = [titles[i] for i in doc_id]
        join_token = "" if dataset == "hotpotqa" else " "
        support_contents: list[str] = [join_token.join(contents[di]) for di in doc_id]
        docs = [f"#### {title}\n\n{content}" for title, content in zip(support_titles, support_contents)]
        docs_id = [hash_object(doc) for doc in docs]

        docs_retrieved = get_docs_retrieved(log_content["trace"], method)
        docs_retrieved_id = [hash_object(doc) for doc in docs_retrieved]

        score = {doc_id: 1 for doc_id in docs_retrieved_id}
        relevance = {doc_id: 1 for doc_id in docs_id}

        # print(json.dumps(docs, indent=4))
        # print(json.dumps(docs_retrieved, indent=4))
        # print(json.dumps(score, indent=4))
        # print(json.dumps(relevance, indent=4))
        return score, relevance


def eval_method_one(log_base_path: str, method: str, cnt: int):
    ids = [f.replace(".json", "") for f in os.listdir(log_base_path) if f.endswith(".json")]
    # ret = {f"{cnt}_{id}": eval_one(f"{log_base_path}/{id}.json", method) for id in ids}
    # scores = {id: v[0] for id, v in ret.items()}
    # relevance = {id: v[1] for id, v in ret.items()}
    scores = {}
    relevance = {}
    for id in ids:
        s, r = eval_one(f"{log_base_path}/{id}.json", method)
        
        # 🚨 终极防线：如果这道题的标准答案为空，直接跳过它！
        # 绝不把空字典交给底层的 pytrec_eval，彻底杜绝 C++ 段错误 (Segfault)
        if len(r) > 0:
            scores[f"{cnt}_{id}"] = s
            relevance[f"{cnt}_{id}"] = r

    return scores, relevance


def eval_method(method: str):
    scores_all = {}
    relevance_all = {}
    for cnt, log_dir in enumerate(logs[dataset][method]):
        #log_base_path = f"log/rag/{dataset}/{log_dir}/output"
        log_base_path = f"outputs/rag/{dataset}/{log_dir}/output"
        scores, relevance = eval_method_one(log_base_path, method, cnt)
        scores_all.update(scores)
        relevance_all.update(relevance)
    metrics = retrieval_metrics(scores_all, relevance_all, k_values)
    return metrics


def eval_dataset():
    # 构建表头
    header_top = ["recall"] * len(k_values) + ["ndcg"] * len(k_values) + ["map"] * len(k_values)
    header_bottom = k_values * 3
    headers = pd.MultiIndex.from_arrays([header_top, header_bottom])

    table_data = []
    for method in methods:
        metrics = eval_method(method)
        row = []
        for metric in ["recall", "ndcg", "map"]:
            for k in k_values:
                key = f"{metric}@{k}"
                row.append(metrics[metric][key])
        table_data.append(row)

    df = pd.DataFrame(table_data, columns=headers, index=methods)
    print(df)


eval_dataset()
