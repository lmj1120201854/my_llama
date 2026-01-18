数据集：
预训练数据集：
ddzhu123/seq-monkey(https://www.modelscope.cn/datasets/ddzhu123/seq-monkey/files)
开源mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2数据集
处理过程：压缩包->(解压缩)->jsonl->(预处理)->seq_monkey_datawhale.jsonl
预处理后数据格式: {"text": value} 其中value长度为512

监督微调数据集：
swift/train_3.5M_CN(https://www.modelscope.cn/datasets/swift/train_3.5M_CN/files)
开源train_3.5M_CN.json数据集
处理过程：json->(预处理)->BelleGroup_sft.jsonl
预处理前数据格式：
[{"from": "human", "value": value}, {"from": "assistant", "value": value}, ...] 多轮对话
预处理后数据格式：
[{"role": "system", "content": "你是一个AI助手"}, {"role": "user", "content": value}, {"role": "assistant", "content": value}, ...] 有系统提示词的多轮对话标准模版


模型架构
遵从Llama2的模型架构

