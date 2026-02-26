#!/usr/bin/env bash

mkdir -p path_to_local_parquet


MODEL_PATH="${MODEL_PATH:-path_to_model_path}"
OUTPUT_DIR="${OUTPUT_DIR:-path_to_local_parquet}"
SAVE_NAME="${SAVE_NAME:-save_name}"
NUM_GPUS="${NUM_GPUS:-8}"
FILTER_BY_SCORE="${FILTER_BY_SCORE:-true}"
MIN_SCORE="${MIN_SCORE:-0.3}"
MAX_SCORE="${MAX_SCORE:-0.8}"
TARGET_SAMPLES="${TARGET_SAMPLES:-1024}"
ENABLE_AUTO_FILL="${ENABLE_AUTO_FILL:-true}"


INPUT_FILES=("$@")

echo "模型: $MODEL_PATH"
echo "输入文件数: ${#INPUT_FILES[@]}"
for f in "${INPUT_FILES[@]}"; do
    echo "  - $f"
done
echo "输出目录: $OUTPUT_DIR"
echo "保存名称: $SAVE_NAME"
echo "GPU数量: $NUM_GPUS"
if [ "$FILTER_BY_SCORE" = "true" ]; then
    echo "Score筛选: 启用 (范围: [$MIN_SCORE, $MAX_SCORE])"
else
    echo "Score筛选: 禁用（保留所有数据）"
fi
echo "目标样本数: $TARGET_SAMPLES"
if [ "$ENABLE_AUTO_FILL" = "true" ]; then
    echo "自动补充: 启用（不足时从其余数据补充到${TARGET_SAMPLES}条）"
else
    echo "自动补充: 禁用"
fi


SCORED_FILES=()

for input_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "文件不存在，跳过: $input_file"
        continue
    fi
    

    scored_file="${input_file%.json}_scored.json"
    

    if [ -f "$scored_file" ]; then
        echo ""
        echo "检测到已评分文件: $scored_file"
        read -p "是否重新评分？[y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   跳过评分，使用现有文件"
            SCORED_FILES+=("$scored_file")
            continue
        fi
    fi
    
    bash question_evaluate/score_questions.sh \
        "$MODEL_PATH" \
        "$input_file" \
        "$scored_file" \
        "$NUM_GPUS"
    
    if [ $? -eq 0 ]; then
        echo "  评分完成: $scored_file"
        SCORED_FILES+=("$scored_file")
    else
        echo "  评分失败: $input_file"
    fi
    
    echo "--------------------------------------------------------------------------------"
done

for f in "${SCORED_FILES[@]}"; do
    echo "  - $f"
done
echo "================================================================================"

if [ ${#SCORED_FILES[@]} -eq 0 ]; then
    echo ""
    echo "没有可用的评分文件，退出"
    exit 1
fi

MERGE_CMD="python3 question_evaluate/merge_and_convert.py \
    --input_files ${SCORED_FILES[@]} \
    --output_dir $OUTPUT_DIR \
    --save_name $SAVE_NAME \
    --target_samples $TARGET_SAMPLES"

if [ "$FILTER_BY_SCORE" = "true" ]; then
    MERGE_CMD="$MERGE_CMD --filter_by_score --min_score $MIN_SCORE --max_score $MAX_SCORE"
fi

if [ "$ENABLE_AUTO_FILL" = "true" ]; then
    MERGE_CMD="$MERGE_CMD --enable_补充"
else
    MERGE_CMD="$MERGE_CMD --disable_补充"
fi

eval $MERGE_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "工作流完成！"
    echo "================================================================================"
    echo "输出文件:"
    echo "  - Parquet: $OUTPUT_DIR/${SAVE_NAME}_train.parquet"
    echo "  - JSON: $OUTPUT_DIR/${SAVE_NAME}_merged.json"
    echo "  - 摘要: $OUTPUT_DIR/${SAVE_NAME}_train_summary.json"
    echo "================================================================================"
    

    echo ""
    echo "最终数据统计:"
    python3 -c "
import pandas as pd
df = pd.read_parquet('$OUTPUT_DIR/${SAVE_NAME}_train.parquet')
print(f'总样本数: {len(df)}')
print(f'')
print(f'Score统计:')
print(f'  - 平均: {df[\"score\"].mean():.3f}')
print(f'  - 中位数: {df[\"score\"].median():.3f}')
print(f'  - 最大: {df[\"score\"].max():.3f}')
print(f'  - 最小: {df[\"score\"].min():.3f}')
print(f'')
print(f'问题类型分布:')
for ptype, count in df['problem_type'].value_counts().items():
    print(f'  - {ptype}: {count} ({count/len(df)*100:.1f}%)')
"
    
else
    echo ""
    echo "================================================================================"
    echo "转换失败"
    echo "================================================================================"
    exit 1
fi

