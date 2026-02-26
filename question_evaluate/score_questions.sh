#!/usr/bin/env bash

MODEL_PATH="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
INPUT_JSON="${2}"
OUTPUT_JSON="${3}"
NUM_GPUS="${4:-8}"

if [ -z "$INPUT_JSON" ]; then
    echo "错误: 必须提供输入JSON文件"
fi


if [ -z "$OUTPUT_JSON" ]; then

    OUTPUT_JSON="${INPUT_JSON%.json}_scored.json"
fi



if [ ! -f "$INPUT_JSON" ]; then
    echo "输入文件不存在: $INPUT_JSON"
    exit 1
fi


TEMP_DIR=$(mktemp -d)
echo "   临时目录: $TEMP_DIR"

python3 -c "
import json
import os
import sys

input_file = '$INPUT_JSON'
temp_dir = '$TEMP_DIR'
num_gpus = $NUM_GPUS


with open(input_file, 'r') as f:
    data = json.load(f)

print(f'   总数据量: {len(data)}')


chunk_size = (len(data) + num_gpus - 1) // num_gpus
print(f'   每份大小: ~{chunk_size}')

for i in range(num_gpus):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(data))
    chunk = data[start_idx:end_idx]
    
    if not chunk:
        continue
    
    chunk_file = os.path.join(temp_dir, f'chunk_{i}.json')
    with open(chunk_file, 'w') as f:
        json.dump(chunk, f)
    
    print(f'   分片 {i}: {len(chunk)} 条 -> chunk_{i}.json')

print('   ✓ 分割完成')
"

pids=()
output_files=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    CHUNK_FILE="$TEMP_DIR/chunk_${i}.json"
    

    if [ ! -f "$CHUNK_FILE" ]; then
        continue
    fi
    
    OUTPUT_CHUNK="$TEMP_DIR/scored_${i}.json"
    output_files+=("$OUTPUT_CHUNK")
    
    echo "   启动 GPU $i: chunk_${i}.json -> scored_${i}.json"
    
    CUDA_VISIBLE_DEVICES=$i python3 question_evaluate/score_questions.py \
        --model "$MODEL_PATH" \
        --input_file "$CHUNK_FILE" \
        --output_file "$OUTPUT_CHUNK" \
        --suffix "$i" \
        --num_samples 9 \
        --skip_existing &
    
    pids+=($!)
done


if [ ${#pids[@]} -gt 0 ]; then
    wait ${pids[0]}
    echo "   ✓ GPU 0 完成"
fi


timeout_duration=3600

(
    sleep $timeout_duration
    echo "超时！强制终止剩余任务..."
    for i in $(seq 1 $((${#pids[@]} - 1))); do
        if kill -0 ${pids[$i]} 2>/dev/null; then
            kill -9 ${pids[$i]} 2>/dev/null
            echo "   ✗ 强制终止 GPU $i"
        fi
    done
) &

timeout_pid=$!


for i in $(seq 1 $((${#pids[@]} - 1))); do
    if wait ${pids[$i]} 2>/dev/null; then
        echo "   ✓ GPU $i 完成"
    else
        echo "   ✗ GPU $i 失败或被终止"
    fi
done


kill $timeout_pid 2>/dev/null || true


python3 -c "
import json
import os
import glob

temp_dir = '$TEMP_DIR'
output_json = '$OUTPUT_JSON'


scored_files = sorted(glob.glob(os.path.join(temp_dir, 'scored_*.json')))
print(f'   找到 {len(scored_files)} 个结果文件')

if not scored_files:
    print('没有找到任何结果文件')
    exit(1)


all_data = []
for f in scored_files:
    try:
        with open(f, 'r') as fp:
            chunk_data = json.load(fp)
            all_data.extend(chunk_data)
            print(f'   ✓ {os.path.basename(f)}: {len(chunk_data)} 条')
    except Exception as e:
        print(f'   ✗ {os.path.basename(f)}: 读取失败 - {e}')

print(f'   总计: {len(all_data)} 条')


os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
with open(output_json, 'w') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f'   ✓ 已保存: {output_json}')


scores = [item.get('score', 0.0) for item in all_data if 'score' in item]
if scores:
    print(f'')
    print(f'评分统计:')
    print(f'   - 已评分: {len(scores)} 条')
    print(f'   - 平均分: {sum(scores)/len(scores):.3f}')
    print(f'   - 最高分: {max(scores):.3f}')
    print(f'   - 最低分: {min(scores):.3f}')
    print(f'   - 高质量(≥0.8): {sum(1 for s in scores if s >= 0.8)} 条 ({sum(1 for s in scores if s >= 0.8)/len(scores)*100:.1f}%)')
    print(f'   - 中质量(0.5-0.8): {sum(1 for s in scores if 0.5 <= s < 0.8)} 条')
    print(f'   - 低质量(<0.5): {sum(1 for s in scores if s < 0.5)} 条')
"


rm -rf "$TEMP_DIR"


