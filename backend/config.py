import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
BASE_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "base_workflow.json")
BASE_WORKFLOW_NOLORA_PATH = os.path.join(os.path.dirname(__file__), "base_workflow_nolora.json")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
COMFYUI_TIMEOUT = int(os.getenv("COMFYUI_TIMEOUT", "600"))
