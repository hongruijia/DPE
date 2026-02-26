#!/usr/bin/env bash

MODEL_PATH="${MODEL_PATH:-path_to_Qwen2.5-VL-7B-Instruct_solver_v3/global_step_10/actor/huggingface}"
DATASET_PATH="${DATASET_PATH:-path_to_Vision-SR1-47K}"
OUTPUT_DIR="${OUTPUT_DIR:-path_to_weakness_analysis}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
RANDOM_SEED="${RANDOM_SEED:-42}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_WORKERS="${MAX_WORKERS:-3}"       
MIN_INTERVAL="${MIN_INTERVAL:-2.0}" 



mkdir -p "$OUTPUT_DIR"

SAMPLED_DATA="$OUTPUT_DIR/sampled_data.json"

python3 -c "
import pandas as pd
import json
import random
import os
import glob

dataset_path = '$DATASET_PATH'
sample_size = $SAMPLE_SIZE
random_seed = $RANDOM_SEED
output_file = '$SAMPLED_DATA'

print(f'   数据集路径: {dataset_path}')


if os.path.isdir(dataset_path):

    parquet_files = sorted(glob.glob(os.path.join(dataset_path, '*.parquet')))
    print(f'   找到 {len(parquet_files)} 个parquet文件')
    
    if not parquet_files:
        print('   目录中没有找到parquet文件')
        exit(1)
    

    random.seed(random_seed)
    selected_file = random.choice(parquet_files)
    print(f'   随机选择: {os.path.basename(selected_file)}')
    
    df = pd.read_parquet(selected_file)
else:

    print(f'   加载单个文件')
    df = pd.read_parquet(dataset_path)

print(f'   数据量: {len(df)}')


random.seed(random_seed)
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=random_seed)
    print(f'   采样数量: {len(df)}')
else:
    print(f'   数据不足，使用全部 {len(df)} 条')


import base64
import numpy as np

def convert_to_serializable(obj):
    \"\"\"将对象转换为可JSON序列化的格式\"\"\"
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, dict):

        if 'bytes' in obj and isinstance(obj['bytes'], bytes):
            return base64.b64encode(obj['bytes']).decode('utf-8')
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    else:
        return obj

samples = []
for idx, row in df.iterrows():

    sample = {
        'id': str(idx),
        'question': str(row.get('problem', row.get('question', ''))),
        'answer': str(row.get('answer', '')),
        'image': convert_to_serializable(row.get('images', row.get('image', ''))),
        'problem_type': str(row.get('problem_type', 'unknown'))
    }
    samples.append(sample)

with open(output_file, 'w') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f'   ✓ 已保存到: {output_file}')
if samples:
    print(f'   样本示例: 问题={samples[0][\"question\"][:50]}...')
"


STAGE1_OUTPUT="$OUTPUT_DIR/stage1_generation"
mkdir -p "$STAGE1_OUTPUT"

bash run_parallel_generation.sh \
    "$MODEL_PATH" \
    "$SAMPLED_DATA" \
    "$STAGE1_OUTPUT" \
    $NUM_GPUS

ANSWERS_FILE="$STAGE1_OUTPUT/all_answers.json"

if [ ! -f "$ANSWERS_FILE" ]; then
    echo ""
    echo "❌ 阶段1失败：未找到答案文件"
    exit 1
fi


STAGE2_OUTPUT="$OUTPUT_DIR/stage2_analysis"
mkdir -p "$STAGE2_OUTPUT"

python3 analyze_weaknesses.py \
    --answers "$ANSWERS_FILE" \
    --output "$STAGE2_OUTPUT" \
    --max-workers $MAX_WORKERS \
    --min-interval $MIN_INTERVAL

if [ $? -eq 0 ]; then
    echo ""
    echo "阶段2完成！"
    

    echo ""
    echo "整理最终结果..."
    
    cp "$STAGE2_OUTPUT/weakness_analysis.json" "$OUTPUT_DIR/"
    cp "$STAGE2_OUTPUT/generation_weights.json" "$OUTPUT_DIR/"
    cp "$STAGE2_OUTPUT/weakness_prompt.txt" "$OUTPUT_DIR/"
    cp "$STAGE2_OUTPUT/summary.json" "$OUTPUT_DIR/"
    cp "$STAGE2_OUTPUT/detailed_results.jsonl" "$OUTPUT_DIR/"
    
    echo "   ✓ 结果已复制到: $OUTPUT_DIR"
else
    echo ""
    echo "阶段2失败"
    exit 1
fi


