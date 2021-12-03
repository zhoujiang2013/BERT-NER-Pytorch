## Chinese NER using Bert

BERT for Chinese NER. 

### dataset list

1. cner: datasets/cner
2. CLUENER: https://github.com/CLUEbenchmark/CLUENER

### model list

1. BERT+Softmax
2. BERT+CRF
3. BERT+Span

### requirement

1. 1.1.0 =< PyTorch < 1.5.0
2. cuda=9.0
3. python3.6+

### input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```

### run the code

1. Modify the configuration information in `run_ner_xxx.py` or `run_ner_xxx.sh` .
2. `sh scripts/run_ner_xxx.sh`

**note**: file structure of the model

```text
├── prev_trained_model
|  └── bert_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
|  |  └── ......
```

### CLUENER result

The overall performance of BERT on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.7897     | 0.8031     | 0.7963    |
| BERT+CRF     | 0.7977 | 0.8177 | 0.8076 |
| BERT+Span    | 0.8132 | 0.8092 | 0.8112 |
| BERT+Span+adv    | 0.8267 | 0.8073 | **0.8169** |
| BERT-small(6 layers)+Span+kd    | 0.8241 | 0.7839 | 0.8051 |
| BERT+Span+focal_loss    | 0.8121 | 0.8008 | 0.8064 |
| BERT+Span+label_smoothing   | 0.8235 | 0.7946 | 0.8088 |

### ALBERT for CLUENER

The overall performance of ALBERT on **dev**:

| model  | version       | Accuracy(entity) | Recall(entity) | F1(entity) | Train time/epoch |
| ------ | ------------- | ---------------- | -------------- | ---------- | ---------------- |
| albert | base_google   | 0.8014           | 0.6908         | 0.7420     | 0.75x            |
| albert | large_google  | 0.8024           | 0.7520         | 0.7763     | 2.1x             |
| albert | xlarge_google | 0.8286           | 0.7773         | 0.8021     | 6.7x             |
| bert   | google        | 0.8118           | 0.8031         | **0.8074**     | -----            |
| albert | base_bright   | 0.8068           | 0.7529         | 0.7789     | 0.75x            |
| albert | large_bright  | 0.8152           | 0.7480         | 0.7802     | 2.2x             |
| albert | xlarge_bright | 0.8222           | 0.7692         | 0.7948     | 7.3x             |

### Cner result

The overall performance of BERT on **dev(test)**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9586(0.9566)     | 0.9644(0.9613)     | 0.9615(0.9590)     |
| BERT+CRF     | 0.9562(0.9539)     | 0.9671(**0.9644**) | 0.9616(0.9591)     |
| BERT+Span    | 0.9604(**0.9620**) | 0.9617(0.9632)     | 0.9611(**0.9626**) |
| BERT+Span+focal_loss    | 0.9516(0.9569) | 0.9644(0.9681)     | 0.9580(0.9625) |
| BERT+Span+label_smoothing   | 0.9566(0.9568) | 0.9624(0.9656)     | 0.9595(0.9612) |


## tips
### 自动下载预训练模型
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

### 手动下载并加载

tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                            do_lower_case=args.do_lower_case,)
model = model_class.from_pretrained(args.model_name_or_path, config=config)

### 镜像
docker run -it -d -p 8888:8888 --ipc=host --name pytorch -v /home/root/docker_dir/:/docker_dir/ pytorch/pytorch:1.3-cuda10.1-cudnn7-devel /bin/bash

### 运行
export PYTHONIOENCODING=utf-8 
python run_ner_crf.py --model_type=bert --model_name_or_path=/workspace/BERT-NER-Pytorch/prev_trained_model/bert-base-chinese --task_name=cner --do_train --do_eval --do_lower_case --data_dir=/workspace/BERT-NER-Pytorch/datasets/cner/ --train_max_seq_length=128 --eval_max_seq_length=512 --per_gpu_train_batch_size=24 --per_gpu_eval_batch_size=24 --learning_rate=3e-5 --crf_learning_rate=1e-3 --num_train_epochs=4.0 --logging_steps=-1 --save_steps=-1 --output_dir=/workspace/BERT-NER-Pytorch/outputs/cner_output/ --overwrite_output_dir --seed=42
