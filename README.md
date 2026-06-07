
## <a name="quick_start"></a>:flight_departure:Quick Start

1. 环境配置

  ```bash
  uv sync --extra llm --extra retrieve
  # 同时写有python-version与pyproject.toml
  # 环境激活 source activate dualrag_env
  ```

2. Retriever
 
  下载数据集，构造检索库。

  ```bash
  # 1. 在 `config/main.yaml` 中把 `task` 修改成 "corpus"
  # 2. 对于不同的数据集修改一下 `corpus` 参数 (或者 `dataset` 参数)
  python main.py

  # 运行main之后进入流程scr/stratup
  #会启动scripts的download，之后进入src/dataset把不同数据集统一成 Item/Dataset，并把语料整理成可检索文档
  ```
3. 微服务
 #封装成独立的接口，通过FASTAPI串联
  #向量
  ```bash
  cd DualRAG/server/embedder
  ./run.sh

  # 检索
  cd DualRAG/server/retriever
  ./run.sh
  ```
  #重排

  cd DualRAG/server/rerank
  ./run.sh

  #NER 实体识别
  cd DualRAG/server/ner
  ./run.sh

# LLM Servers 大模型

  ```bash
  cd DualRAG/server/vllm
  # 缺权限时 chmod +x run_qwen.sh
  ./run_qwen.sh
  ```

4. 跑RAG

# 1. 在 `config/main.yaml` 中把 `task` 修改成 "rag"
# 2. main.yaml中datasets可选择不同数据集 
# 3. rag.yaml 可调rag方法与训练集大小等 method中有每个rag方法的具体参数配置
# 4. rag方法的具体实现放在src/rag下
# 5.流程约为：main， Hydra 读取 config 配置，Runbuilder看任务为rag，进入 RagRunner，加载数据集，实例化配置指定的 RAG 方法，并批量处理问题，最终每道题都会保存回答、标准答案、评测指标和完整 trace，总体结果写入 evaluate.csv。
  ```bash
  python main.py
  ```

  - The experiment configurations are in `config/`. You can find the configs for different systems, models, and datasets there.

  - The output will be saved in `output/`. You can find the logs in the corresponding subdirectory.




