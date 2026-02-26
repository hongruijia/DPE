import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") 

SERPER_API_KEY = your_serper_api_key  

DASHSCOPE_BASE_URL = "http://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_IMAGE_EDIT_URL = "https://dashscope.aliyuncs.com/api/v1"

QWEN_VL_MODEL = "qwen-vl-max-latest"
QWEN3_VL_MODEL = "qwen3-vl-plus"  
QWEN_IMAGE_EDIT_MODEL = "qwen-image-edit-plus"

O3_MODEL = "o3-2025-04-16"
GEMINI_MODEL = "gemini-2.5-pro"
CLAUDE_MODEL = "claude-sonnet-4-20250514-thinking"

STORAGE_PATH = os.getenv("STORAGE_PATH", "")

MAX_RETRIES = 5
BASE_DELAY = 2
MAX_DELAY = 30
REQUEST_TIMEOUT = 60  
MAX_SEARCH_IMAGES = 3  
MAX_GENERATED_IMAGES = 3  


