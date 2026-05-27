import uuid
import time
import asyncio
import base64
import random
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import BASE_WORKFLOW_PATH, BASE_WORKFLOW_NOLORA_PATH, COMFYUI_URL, COMFYUI_TIMEOUT
from backend.pipeline.intent import rewrite_prompt
from backend.pipeline.comfyui_client import ComfyUIClient
from pathlib import Path

app = FastAPI(title="Game Art Agent Demo")

HISTORY_FILE = Path(__file__).parent / "generation_history.json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    prompt: str
    use_lora: bool = True
    free_prompt: bool = False
    smart_rewrite: bool = False


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "processing",
        "steps": [],
        "image": None,
        "info": None,
        "error": None,
    }
    asyncio.create_task(run_pipeline(task_id, req.prompt, req.use_lora, req.free_prompt, req.smart_rewrite))
    return {"task_id": task_id}


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


async def run_pipeline(task_id: str, user_input: str, use_lora: bool = True, free_prompt: bool = False, smart_rewrite: bool = False):
    start_time = time.monotonic()

    def add_step(name: str, status: str):
        tasks[task_id]["steps"].append({"name": name, "status": status})

    try:
        workflow_path = BASE_WORKFLOW_PATH if use_lora else BASE_WORKFLOW_NOLORA_PATH
        with open(workflow_path, "r", encoding="utf-8") as f:
            base_workflow = json.load(f)

        if smart_rewrite:
            add_step("rewrite", "in_progress")
            prompts = rewrite_prompt(user_input)
            add_step("rewrite", "done")

            display_preview = _strip_triggers(prompts["positive"])
            add_step("free_prompt", "in_progress")
            workflow_str = json.dumps(base_workflow, ensure_ascii=False)
            workflow_str = workflow_str.replace("{PROMPT}", prompts["positive"])
            workflow_str = workflow_str.replace("{NEGATIVE}", prompts["negative"])
            workflow = json.loads(workflow_str)
            for node in workflow.values():
                if node.get("class_type") == "KSampler":
                    node["inputs"]["seed"] = random.randint(0, 2**31 - 1)
            character_name = "smart"
            template_id = "smart_rewrite"
            template_desc = f"智能改写: {display_preview[:40]}..."
            score = 1.0
            add_step("free_prompt", "done")
        else:
            add_step("free_prompt", "in_progress")
            workflow_str = json.dumps(base_workflow, ensure_ascii=False)
            workflow_str = workflow_str.replace("{PROMPT}", user_input)
            workflow_str = workflow_str.replace("{NEGATIVE}",
                "blurry, low quality, worst quality, jpeg artifacts, watermark, text, signature, nsfw, lowres, bad anatomy, extra limbs, deformed")
            workflow = json.loads(workflow_str)
            for node in workflow.values():
                if node.get("class_type") == "KSampler":
                    node["inputs"]["seed"] = random.randint(0, 2**31 - 1)
            character_name = "free"
            template_id = "free_prompt"
            template_desc = "自由输入模式"
            score = 1.0
            add_step("free_prompt", "done")

        add_step("generate", "in_progress")
        client = ComfyUIClient(base_url=COMFYUI_URL, timeout=COMFYUI_TIMEOUT)
        prompt_id = await client.submit_workflow(workflow)
        image_bytes = await client.wait_for_result(prompt_id)
        add_step("generate", "done")

        elapsed = round(time.monotonic() - start_time, 1)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Extract effective prompts from the workflow
        effective_prompt = ""
        effective_negative = ""
        for node in workflow.values():
            meta = node.get("_meta", {})
            if node.get("class_type") == "CLIPTextEncode":
                if meta.get("title") == "Positive Prompt":
                    effective_prompt = node["inputs"]["text"]
                elif meta.get("title") == "Negative Prompt":
                    effective_negative = node["inputs"]["text"]

        # Strip trigger words from display text
        display_prompt = _strip_triggers(effective_prompt)

        tasks[task_id].update({
            "status": "done",
            "image": f"data:image/png;base64,{image_b64}",
            "info": {
                "character": character_name,
                "template": template_id,
                "template_desc": template_desc,
                "score": round(score, 2),
                "elapsed": elapsed,
                "use_lora": use_lora,
                "prompt": display_prompt,
                "negative_prompt": effective_negative,
            },
        })

        # Append to history
        _save_history(task_id, display_prompt, template_id, elapsed, use_lora)

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["steps"].append(
            {"name": "error", "status": "failed", "detail": traceback.format_exc()}
        )


def _strip_triggers(text: str) -> str:
    """Remove LoRA trigger words from display text."""
    prefixes = ("guofeng, chinese style,", "guofeng, chinese style", "guofeng,", "guofeng")
    lower = text.lower()
    for p in prefixes:
        if lower.startswith(p):
            return text[len(p):].strip(", ")
    return text


def _save_history(task_id: str, prompt: str, template: str, elapsed: float, use_lora: bool):
    record = {
        "task_id": task_id,
        "prompt": prompt,
        "template": template,
        "elapsed": elapsed,
        "use_lora": use_lora,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text("utf-8"))
        except Exception:
            history = []
    history.insert(0, record)
    # Keep last 50 records
    history = history[:50]
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


@app.get("/api/history")
async def get_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text("utf-8"))
    return []


import os as _os
_frontend_dir = _os.path.join(_os.path.dirname(__file__), "..", "frontend")
if _os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
