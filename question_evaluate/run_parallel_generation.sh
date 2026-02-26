#!/usr/bin/env bash


set -e

MODEL_PATH="${1}"
INPUT_JSON="${2}"
OUTPUT_DIR="${3}"
NUM_GPUS="${4:-8}"

if [ -z "$MODEL_PATH" ] || [ -z "$INPUT_JSON" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "用法: bash run_parallel_generation.sh <model_path> <input_json> <output_dir> [num_gpus]"
    exit 1
fi


mkdir -p "$OUTPUT_DIR"


TEMP_DIR=$(mktemp -d)
echo "   临时目录: $TEMP_DIR"

python3 -c "
import json
import os

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


echo ""
echo "步骤2: 并行生成答案（$NUM_GPUS 个GPU）..."

pids=()
output_files=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    CHUNK_FILE="$TEMP_DIR/chunk_${i}.json"
    

    if [ ! -f "$CHUNK_FILE" ]; then
        continue
    fi
    
    OUTPUT_CHUNK="$OUTPUT_DIR/answers_gpu${i}.json"
    output_files+=("$OUTPUT_CHUNK")
    
    echo "   启动 GPU $i: chunk_${i}.json -> answers_gpu${i}.json"
    
    CUDA_VISIBLE_DEVICES=$i python3 generate_answers_parallel.py \
        --model "$MODEL_PATH" \
        --input "$CHUNK_FILE" \
        --output "$OUTPUT_CHUNK" \
        --gpu $i &
    
    pids+=($!)
done

echo ""
echo "等待所有GPU完成..."


if [ ${#pids[@]} -gt 0 ]; then
    wait ${pids[0]}
    echo "   ✓ GPU 0 完成"
fi


timeout_duration=7200

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

echo ""
echo "所有GPU任务完成"


echo ""
echo "步骤3: 合并结果..."

OUTPUT_JSON="$OUTPUT_DIR/all_answers.json"

python3 -c "
import json
import os
import glob

output_dir = '$OUTPUT_DIR'
output_json = '$OUTPUT_JSON'


answer_files = sorted(glob.glob(os.path.join(output_dir, 'answers_gpu*.json')))
print(f'   找到 {len(answer_files)} 个结果文件')

if not answer_files:
    print(没有找到任何结果文件')
    exit(1)


all_data = []
for f in answer_files:
    try:
        with open(f, 'r') as fp:
            chunk_data = json.load(fp)
            all_data.extend(chunk_data)
            print(f'   ✓ {os.path.basename(f)}: {len(chunk_data)} 条')
    except Exception as e:
        print(f'   ✗ {os.path.basename(f)}: 读取失败 - {e}')

print(f'   总计: {len(all_data)} 条')


with open(output_json, 'w') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f'   ✓ 已保存: {output_json}')


success_count = sum(1 for item in all_data if not item.get('generation_error', False))
error_count = len(all_data) - success_count

print(f'')
print(f'生成统计:')
print(f'   - 成功: {success_count} 条 ({success_count/len(all_data)*100:.1f}%)')
print(f'   - 失败: {error_count} 条 ({error_count/len(all_data)*100:.1f}%)')
"


echo ""
echo "步骤4: 清理临时文件..."
rm -rf "$TEMP_DIR"
echo "   ✓ 已清理: $TEMP_DIR"


