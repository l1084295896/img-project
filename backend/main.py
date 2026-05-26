import uuid
import time
import asyncio
import base64
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import TEMPLATES_DIR, BASE_WORKFLOW_PATH, COMFYUI_URL, COMFYUI_TIMEOUT
from backend.pipeline.intent import recognize_intent
from backend.pipeline.template_matcher import match_template
from backend.pipeline.workflow_builder import build_workflow
from backend.pipeline.comfyui_client import ComfyUIClient

app = FastAPI(title="Game Art Agent Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    prompt: str


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
    asyncio.create_task(run_pipeline(task_id, req.prompt))
    return {"task_id": task_id}


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


async def run_pipeline(task_id: str, user_input: str):
    """Execute the full generation pipeline and update task state."""
    start_time = time.monotonic()

    def add_step(name: str, status: str):
        tasks[task_id]["steps"].append({"name": name, "status": status})

    try:
        # Step 1: Intent recognition
        add_step("intent", "in_progress")
        intent = recognize_intent(user_input)
        add_step("intent", "done")

        # Step 2: Template matching
        add_step("template", "in_progress")
        template, score = match_template(intent, TEMPLATES_DIR)
        if template is None or score < 0.1:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = "未找到匹配的模板，请补充更多描述（姿势、视角、情绪等）"
            add_step("template", "failed")
            return
        add_step("template", "done")

        # Step 3: Build workflow
        add_step("workflow", "in_progress")
        import json as _json
        with open(BASE_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            base_workflow = _json.load(f)
        character_name = intent.get("character") or "character"
        workflow = build_workflow(template, base_workflow, character_name)
        add_step("workflow", "done")

        # Step 4: Generate image via ComfyUI
        add_step("generate", "in_progress")
        client = ComfyUIClient(base_url=COMFYUI_URL, timeout=COMFYUI_TIMEOUT)
        prompt_id = await client.submit_workflow(workflow)
        image_bytes = await client.wait_for_result(prompt_id)
        add_step("generate", "done")

        # Done
        elapsed = round(time.monotonic() - start_time, 1)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        tasks[task_id].update({
            "status": "done",
            "image": f"data:image/png;base64,{image_b64}",
            "info": {
                "character": intent.get("character"),
                "template": template["id"],
                "template_desc": template["description"],
                "score": round(score, 2),
                "elapsed": elapsed,
            },
        })

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["steps"].append(
            {"name": "error", "status": "failed", "detail": traceback.format_exc()}
        )


# Mount frontend after API routes
import os as _os
_frontend_dir = _os.path.join(_os.path.dirname(__file__), "..", "frontend")
if _os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
