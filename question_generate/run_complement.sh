#!/usr/bin/env bash

CHECKPOINT_FILE=""
FINAL_FILE=""
QUOTA_FILE=""
OUTPUT_DIR=""
SAVE_NAME="optimized_generated"
MAX_WORKERS=5
DASHSCOPE_KEY=""
OPENAI_KEY=""
OPENAI_URL=""
WEAKNESS_CONTEXT=""
MERGE="--merge"


while [ $# -gt 0 ]; do
    case $1 in
        --checkpoint-file)
            CHECKPOINT_FILE="$2"
            shift 2
            ;;
        --final-file)
            FINAL_FILE="$2"
            shift 2
            ;;
        --quota-file)
            QUOTA_FILE="$2"
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
        --parallel)
            PARALLEL="--parallel"
            shift
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --no-resume)
            NO_RESUME="--no-resume"
            shift
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
        --weakness-context)
            WEAKNESS_CONTEXT="$2"
            shift 2
            ;;
        --no-merge)
            MERGE="--no-merge"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT_FILE" ]; then
    echo "❌ 错误: 必须指定 --checkpoint-file"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -z "$FINAL_FILE" ]; then
    echo "❌ 错误: 必须指定 --final-file"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -z "$QUOTA_FILE" ]; then
    echo "❌ 错误: 必须指定 --quota-file"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "❌ 错误: 必须指定 --output-dir"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -z "$DASHSCOPE_KEY" ] && [ -z "$OPENAI_KEY" ]; then
    echo "❌ 错误: 必须至少指定一个 API Key"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -n "$DASHSCOPE_KEY" ]; then
    export DASHSCOPE_API_KEY="$DASHSCOPE_KEY"
    echo "✓ Dashscope API Key 已设置"
fi

if [ -n "$OPENAI_KEY" ]; then
    export OPENAI_API_KEY="$OPENAI_KEY"
    echo "✓ OpenAI API Key 已设置"
fi

if [ -n "$OPENAI_URL" ]; then
    export OPENAI_BASE_URL="$OPENAI_URL"
    echo "✓ OpenAI Base URL 已设置: $OPENAI_URL"
fi


if [ -n "$PARALLEL" ]; then
    echo "⚡ 并行模式: 启用 (线程数: $MAX_WORKERS)"
else
    echo "⚡ 并行模式: 禁用"
fi
if [ "$MERGE" = "--merge" ]; then
    echo "🔗 自动合并: 启用"
else
    echo "🔗 自动合并: 禁用"
fi
echo "======================================"


cd "$(dirname "$0")"


CMD="python3 question_generate_complement.py \
    --checkpoint-file \"$CHECKPOINT_FILE\" \
    --final-file \"$FINAL_FILE\" \
    --quota-file \"$QUOTA_FILE\" \
    --output-dir \"$OUTPUT_DIR\" \
    --save-name \"$SAVE_NAME\" \
    --max-workers $MAX_WORKERS \
    $PARALLEL \
    $NO_RESUME \
    $MERGE \
    --verbose"

if [ -n "$WEAKNESS_CONTEXT" ]; then
    CMD="$CMD --weakness-context \"$WEAKNESS_CONTEXT\""
    echo "  - 弱点上下文: $WEAKNESS_CONTEXT"
fi



eval $CMD


