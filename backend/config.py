import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
BASE_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "base_workflow.json")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
COMFYUI_TIMEOUT = int(os.getenv("COMFYUI_TIMEOUT", "120"))
