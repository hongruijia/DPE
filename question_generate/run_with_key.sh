#!/usr/bin/env bash

DATA_PATH="path_to_Vision-SR1-47K"
OUTPUT_DIR="path_to_outputs"
NUM_SAMPLES=10
SAVE_NAME="optimized_generated"
SUFFIX="v1"
MAX_WORKERS=5
DASHSCOPE_KEY=""
OPENAI_KEY=""
OPENAI_URL=""
WEAKNESS_CONTEXT=""
CATEGORY_QUOTAS=""
TARGET_MODE="input"  

while [ $# -gt 0 ]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="--parallel"
            shift
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --no-resume)
            NO_RESUME="--no-resume"
            shift
            ;;
        --no-clean-tmp)
            NO_CLEAN_TMP="--no-clean-tmp"
            shift
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --key)
            DASHSCOPE_KEY="$2"
            shift 2
            ;;
        --openai-key)
            OPENAI_KEY="$2"
            shift 2
            ;;
        --openai-url)
            OPENAI_URL="$2"
            shift 2
            ;;
        --input)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save-name)
            SAVE_NAME="$2"
            shift 2
            ;;
        --sample-size)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --weakness-context)
            WEAKNESS_CONTEXT="$2"
            shift 2
            ;;
        --category-quotas|--image-type-quotas)
            CATEGORY_QUOTAS="$2"
            shift 2
            ;;
        --weakness-prompt)
            WEAKNESS_CONTEXT="$2"
            shift 2
            ;;
        --target-mode)
            TARGET_MODE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DASHSCOPE_KEY" ] && [ -z "$OPENAI_KEY" ]; then
    echo "错误: 必须至少指定一个 API Key"
    echo ""
    echo "用法:"
    echo "  bash run_with_key.sh --key 'dashscope_key' --num_samples 100"
    echo "  bash run_with_key.sh --openai-key 'openai_key' --num_samples 100"
    echo "  bash run_with_key.sh --key 'dashscope_key' --openai-key 'openai_key' --num_samples 100"
    echo ""
    echo "使用 --help 查看详细帮助"
    exit 1
fi

if [ -n "$DASHSCOPE_KEY" ]; then
    export DASHSCOPE_API_KEY="$DASHSCOPE_KEY"
    echo "✓ Dashscope API Key 已设置 (用于 Qwen 模型)"
fi

if [ -n "$OPENAI_KEY" ]; then
    export OPENAI_API_KEY="$OPENAI_KEY"
    echo "✓ OpenAI API Key 已设置 (用于 Gemini/Claude/O3)"
fi

if [ -n "$OPENAI_URL" ]; then
    export OPENAI_BASE_URL="$OPENAI_URL"
    echo "✓ OpenAI Base URL 已设置: $OPENAI_URL"
fi

export TMP_IMAGE_SUFFIX="$SUFFIX"

echo "======================================"
echo "问题生成系统"
echo "======================================"
echo "运行参数:"
echo "  - 样本数: $NUM_SAMPLES"
echo "  - 目标模式: $TARGET_MODE"
if [ "$TARGET_MODE" = "input" ]; then
    echo "    (处理前 $NUM_SAMPLES 个输入样本)"
else
    echo "    (生成 $NUM_SAMPLES 个成功结果)"
fi
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 文件后缀: $SUFFIX"
echo "  - 文件名: ${SAVE_NAME}_${SUFFIX}"
echo "  - 临时图片目录: tmp-image/$SUFFIX (独立)"
echo "  - API Key: ${DASHSCOPE_KEY:0:10}...${DASHSCOPE_KEY: -5} (已隐藏)"
if [ -n "$PARALLEL" ]; then
    echo "  - 执行模式: 并行 (线程数: $MAX_WORKERS)"
else
    echo "  - 执行模式: 串行"
fi
if [ -z "$NO_RESUME" ]; then
    echo "  - 断点续传: 启用"
else
    echo "  - 断点续传: 禁用"
fi
if [ -z "$NO_CLEAN_TMP" ]; then
    echo "  - 自动清理临时图片: 启用"
else
    echo "  - 自动清理临时图片: 禁用"
fi
echo "======================================"

# 切换到 question_generate 目录
cd "$(dirname "$0")"

# 运行优化的生成程序
CMD="python3 question_generate_optimized.py \
    --data_path \"$DATA_PATH\" \
    --num_samples $NUM_SAMPLES \
    --output_dir \"$OUTPUT_DIR\" \
    --save_name \"$SAVE_NAME\" \
    --suffix \"$SUFFIX\" \
    --max_workers $MAX_WORKERS \
    --target-mode \"$TARGET_MODE\" \
    $PARALLEL \
    $NO_RESUME \
    $NO_CLEAN_TMP \
    --verbose"

# 添加弱点上下文（如果提供）
if [ -n "$WEAKNESS_CONTEXT" ]; then
    CMD="$CMD --weakness_context \"$WEAKNESS_CONTEXT\""
    echo "  - 弱点上下文: $WEAKNESS_CONTEXT"
fi

# 添加类别配额（如果提供）
if [ -n "$CATEGORY_QUOTAS" ]; then
    CMD="$CMD --category_quotas \"$CATEGORY_QUOTAS\""
    echo "  - 类别配额: $CATEGORY_QUOTAS"
fi

# 执行命令
eval $CMD

echo ""
echo "======================================"
echo "脚本执行完成"
echo "======================================"
echo "生成文件位置:"
echo "  - 增量文件: ${OUTPUT_DIR}/${SAVE_NAME}_${SUFFIX}_incremental_*.jsonl"
echo "  - 最终文件: ${OUTPUT_DIR}/${SAVE_NAME}_${SUFFIX}.json"
echo "  - 检查点: ${OUTPUT_DIR}/${SAVE_NAME}_${SUFFIX}_checkpoint.json"
echo "======================================"

