# Game Art Agent Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangChain-driven pipeline that takes natural language prompts from game artists, recognizes intent, matches prompt templates, and generates character art via ComfyUI with a LoRA-tuned model.

**Architecture:** FastAPI backend serves a single-page HTML frontend. A linear LangChain pipeline (intent→match→build→generate) processes requests. Qwen3 (DashScope) handles NLP. ComfyUI runs locally with the user's `last.safetensors` LoRA. Templates stored as JSON files with keyword-based retrieval.

**Tech Stack:** Python 3.11+, FastAPI, LangChain, dashscope (Qwen3), httpx, pytest, vanilla HTML/JS

---

## File Map

| File | Purpose |
|------|---------|
| `backend/config.py` | Env vars: DASHSCOPE_API_KEY, COMFYUI_URL |
| `backend/pipeline/intent.py` | Calls Qwen3 → returns structured `{category, intent, character, pose, ...}` |
| `backend/pipeline/template_matcher.py` | Loads template JSON files, scores by tag overlap, returns best match |
| `backend/pipeline/workflow_builder.py` | Merges template params into `base_workflow.json` skeleton |
| `backend/pipeline/comfyui_client.py` | Submits workflow JSON to ComfyUI API, polls for result image |
| `backend/main.py` | FastAPI app: `POST /api/generate`, `GET /api/result/{task_id}`, static file mount |
| `backend/templates/*.json` | 3 sample prompt templates for demo |
| `backend/tests/*.py` | Unit tests for each pipeline module + integration test for API |
| `frontend/index.html` | Single-page UI: input, progress display, image preview, confirm/retry |
| `requirements.txt` | All Python dependencies |

---

### Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `backend/__init__.py`
- Create: `backend/config.py`
- Create: `backend/pipeline/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `.env.example`

- [ ] **Step 1: Write project config and requirements**

```bash
mkdir -p g:/img-project/backend/pipeline g:/img-project/backend/tests g:/img-project/backend/templates g:/img-project/frontend
```

`requirements.txt`:
```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
langchain>=0.3.0
langchain-community>=0.3.0
dashscope>=1.20.0
httpx>=0.27.0
python-dotenv>=1.0.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
pytest-mock>=3.14.0
```

`backend/__init__.py` (empty)

`backend/pipeline/__init__.py` (empty)

`backend/tests/__init__.py` (empty)

`backend/config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
BASE_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "base_workflow.json")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
COMFYUI_TIMEOUT = int(os.getenv("COMFYUI_TIMEOUT", "120"))
```

`.env.example`:
```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
COMFYUI_URL=http://localhost:8188
QWEN_MODEL=qwen-max
COMFYUI_TIMEOUT=120
```

- [ ] **Step 2: Install dependencies and verify**

```bash
cd g:/img-project && pip install -r requirements.txt
```

```bash
python -c "import fastapi; import langchain; import dashscope; import httpx; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd g:/img-project && git add -A && git commit -m "feat: scaffold project structure and dependencies"
```

---

### Task 2: Intent Recognizer (intent.py)

**Files:**
- Create: `backend/pipeline/intent.py`
- Create: `backend/tests/test_intent.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_intent.py`:
```python
import pytest
import json
from unittest.mock import patch, MagicMock
from backend.pipeline.intent import recognize_intent


def make_mock_response(content_dict):
    """Helper to create a mock DashScope response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.output = MagicMock()
    mock.output.choices = [
        MagicMock(message=MagicMock(content=json.dumps(content_dict)))
    ]
    return mock


@patch("backend.pipeline.intent.Generation")
def test_recognize_intent_returns_structured_data(mock_gen):
    mock_gen.return_value.call.return_value = make_mock_response({
        "category": "角色",
        "intent": "action_pose",
        "character": "艾琳",
        "pose": "战斗姿态",
        "mood": "张力",
        "angle": "正面",
        "extra_description": None
    })

    result = recognize_intent("生成一张艾琳的战斗姿态")

    assert result["category"] == "角色"
    assert result["intent"] == "action_pose"
    assert result["character"] == "艾琳"
    assert result["pose"] == "战斗姿态"


@patch("backend.pipeline.intent.Generation")
def test_recognize_intent_fills_missing_fields_with_none(mock_gen):
    mock_gen.return_value.call.return_value = make_mock_response({
        "category": "场景",
        "intent": "scene_generation",
        "character": None,
        "pose": None,
        "mood": None,
        "angle": "鸟瞰",
        "extra_description": "古城"
    })

    result = recognize_intent("生成古代城市鸟瞰图")

    assert result["character"] is None
    assert result["angle"] == "鸟瞰"


@patch("backend.pipeline.intent.Generation")
def test_recognize_intent_handles_api_error(mock_gen):
    mock_gen.return_value.call.side_effect = Exception("API error")

    with pytest.raises(Exception, match="Intent recognition failed"):
        recognize_intent("测试prompt")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd g:/img-project && python -m pytest backend/tests/test_intent.py -v
```

Expected: FAIL — module not found or function not defined

- [ ] **Step 3: Write minimal implementation**

`backend/pipeline/intent.py`:
```python
import json
import dashscope
from dashscope import Generation
from backend.config import DASHSCOPE_API_KEY, QWEN_MODEL

dashscope.api_key = DASHSCOPE_API_KEY

SYSTEM_PROMPT = """你是一个游戏美术需求分析助手。根据用户输入的自然语言描述，提取以下结构化信息，以JSON格式返回。

返回格式：
{
  "category": "角色|场景|道具|UI",
  "intent": "生成意图类型，如 stand_pose, action_pose, expression, three_view, illustration, scene_generation",
  "character": "角色名称，非角色类可为null",
  "pose": "姿势描述，不确定为null",
  "mood": "情绪/氛围，不确定为null",
  "angle": "视角，不确定为null",
  "extra_description": "额外描述，不确定为null"
}

规则：
- 只返回JSON，不要包含```json```标记或其他任何文字
- 如果字段无法从输入中确定，设为null
- category 必须填写"""


def recognize_intent(user_input: str) -> dict:
    """Extract structured intent and parameters from natural language input via Qwen3."""
    try:
        response = Generation.call(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            result_format="message"
        )

        if response.status_code != 200:
            raise Exception(f"Qwen3 API error: status={response.status_code}")

        content = response.output.choices[0].message.content
        result = json.loads(content)

        defaults = {"category": "", "intent": "", "character": None,
                    "pose": None, "mood": None, "angle": None,
                    "extra_description": None}
        for key, default in defaults.items():
            result.setdefault(key, default)

        return result

    except json.JSONDecodeError:
        raise Exception(f"Intent recognition failed: invalid JSON response: {content}")
    except Exception as e:
        raise Exception(f"Intent recognition failed: {str(e)}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd g:/img-project && python -m pytest backend/tests/test_intent.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd g:/img-project && git add backend/pipeline/__init__.py backend/pipeline/intent.py backend/tests/test_intent.py && git commit -m "feat: add intent recognition via Qwen3"
```

---

### Task 3: Template Matcher (template_matcher.py)

**Files:**
- Create: `backend/pipeline/template_matcher.py`
- Create: `backend/templates/char_stand_front.json`
- Create: `backend/templates/char_action_fight.json`
- Create: `backend/templates/char_expression.json`
- Create: `backend/tests/test_template_matcher.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Write sample template data**

`backend/templates/char_stand_front.json`:
```json
{
  "id": "char_stand_front",
  "description": "角色正面站立、中性表情",
  "tags": ["站立", "正面", "中性", "全身", "概念设计"],
  "positive_prompt": "(masterpiece:1.2), (best quality:1.0), {character_name}, standing, front view, neutral expression, full body, white background",
  "negative_prompt": "blurry, low quality, distorted face, extra limbs, watermark, nsfw",
  "workflow_params": {
    "steps": 25,
    "cfg_scale": 7.0,
    "width": 768,
    "height": 1024,
    "sampler": "euler_ancestral",
    "lora_strength": 0.8
  }
}
```

`backend/templates/char_action_fight.json`:
```json
{
  "id": "char_action_fight",
  "description": "角色战斗姿态",
  "tags": ["战斗", "动作", "动态", "武器", "张力", "攻击"],
  "positive_prompt": "(masterpiece:1.2), (best quality:1.0), {character_name}, battle stance, dynamic pose, action scene, dramatic lighting, weapon",
  "negative_prompt": "blurry, low quality, deformed body, extra fingers, watermark, nsfw",
  "workflow_params": {
    "steps": 30,
    "cfg_scale": 8.0,
    "width": 1024,
    "height": 768,
    "sampler": "dpmpp_2m_karras",
    "lora_strength": 0.75
  }
}
```

`backend/templates/char_expression.json`:
```json
{
  "id": "char_expression",
  "description": "角色表情特写",
  "tags": ["表情", "特写", "面部", "情绪", "头像", "肖像"],
  "positive_prompt": "(masterpiece:1.2), (best quality:1.0), {character_name}, portrait, detailed face, expressive, close-up, soft lighting",
  "negative_prompt": "blurry, low quality, distorted face, extra limbs, watermark, nsfw, full body",
  "workflow_params": {
    "steps": 25,
    "cfg_scale": 7.5,
    "width": 768,
    "height": 768,
    "sampler": "euler_ancestral",
    "lora_strength": 0.85
  }
}
```

- [ ] **Step 2: Write the failing test**

`backend/tests/conftest.py`:
```python
import os
import pytest

@pytest.fixture
def templates_dir():
    return os.path.join(os.path.dirname(__file__), "..", "templates")
```

`backend/tests/test_template_matcher.py`:
```python
import pytest
from backend.pipeline.template_matcher import load_templates, match_template, calculate_score


def test_load_templates_loads_all_json_files(templates_dir):
    templates = load_templates(templates_dir)
    assert len(templates) == 3
    ids = {t["id"] for t in templates}
    assert "char_stand_front" in ids
    assert "char_action_fight" in ids
    assert "char_expression" in ids


def test_calculate_score_exact_match():
    params = {"pose": "站立", "mood": "中性", "angle": "正面"}
    template = {"tags": ["站立", "正面", "中性", "全身"]}

    score = calculate_score(params, template)
    assert score == 3 / 4  # 3 tags matched out of 4


def test_calculate_score_no_match():
    params = {"pose": "战斗", "mood": "张力"}
    template = {"tags": ["站立", "正面", "中性", "全身"]}

    score = calculate_score(params, template)
    assert score == 0.0


def test_calculate_score_handles_empty_tags():
    params = {"pose": "战斗"}
    template = {"tags": []}

    score = calculate_score(params, template)
    assert score == 0.0


def test_match_template_returns_best_match(templates_dir):
    params = {"pose": "战斗", "mood": "张力", "angle": "正面", "category": "角色"}

    best, score = match_template(params, templates_dir)
    assert best["id"] == "char_action_fight"
    assert score > 0


def test_match_template_returns_low_score_when_no_good_match(templates_dir):
    params = {"pose": "飞行", "mood": "愉快", "angle": "仰视", "category": "角色"}

    best, score = match_template(params, templates_dir)
    assert score < 0.3
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd g:/img-project && python -m pytest backend/tests/test_template_matcher.py -v
```

Expected: FAIL — module not found

- [ ] **Step 4: Write minimal implementation**

`backend/pipeline/template_matcher.py`:
```python
import json
import os


def load_templates(templates_dir: str) -> list[dict]:
    """Load all JSON template files from the templates directory."""
    templates = []
    if not os.path.isdir(templates_dir):
        return templates

    for filename in sorted(os.listdir(templates_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(templates_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                template = json.load(f)
                templates.append(template)

    return templates


def calculate_score(params: dict, template: dict) -> float:
    """Score a template against extracted params by tag intersection.

    Score = matched_tags / total_template_tags.
    """
    param_keywords = set()
    for key in ("pose", "mood", "angle", "category"):
        value = params.get(key)
        if value:
            param_keywords.add(value)

    template_tags = set(template.get("tags", []))
    if not template_tags:
        return 0.0

    intersection = param_keywords & template_tags
    return len(intersection) / len(template_tags)


def match_template(params: dict, templates_dir: str) -> tuple[dict | None, float]:
    """Find the best matching template for the given params.

    Returns (best_template, score). If no templates loaded, returns (None, 0.0).
    Score ranges from 0.0 (no match) to 1.0 (perfect match).
    """
    templates = load_templates(templates_dir)
    if not templates:
        return None, 0.0

    best_template = None
    best_score = 0.0

    for template in templates:
        score = calculate_score(params, template)
        if score > best_score:
            best_score = score
            best_template = template

    return best_template, best_score
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd g:/img-project && python -m pytest backend/tests/test_template_matcher.py -v
```

Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
cd g:/img-project && git add backend/pipeline/template_matcher.py backend/templates/ backend/tests/test_template_matcher.py backend/tests/conftest.py && git commit -m "feat: add template matcher with keyword-based scoring"
```

---

### Task 4: Workflow Builder (workflow_builder.py)

**Files:**
- Create: `backend/pipeline/workflow_builder.py`
- Create: `backend/tests/fixtures/mock_base_workflow.json`
- Create: `backend/tests/test_workflow_builder.py`

- [ ] **Step 1: Write mock base_workflow fixture**

```bash
mkdir -p g:/img-project/backend/tests/fixtures
```

`backend/tests/fixtures/mock_base_workflow.json`:
```json
{
  "last_node_id": 7,
  "nodes": {
    "1": {
      "type": "CheckpointLoaderSimple",
      "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
    },
    "2": {
      "type": "LoraLoader",
      "inputs": {
        "lora_name": "last.safetensors",
        "strength_model": 0.8,
        "strength_clip": 0.8
      }
    },
    "3": {
      "type": "CLIPTextEncode",
      "inputs": {"text": "{PROMPT}"}
    },
    "4": {
      "type": "CLIPTextEncode",
      "inputs": {"text": "{NEGATIVE}"}
    },
    "5": {
      "type": "KSampler",
      "inputs": {
        "seed": 42,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "denoise": 1.0
      }
    },
    "6": {
      "type": "VAEDecode",
      "inputs": {}
    },
    "7": {
      "type": "SaveImage",
      "inputs": {"filename_prefix": "output"}
    }
  },
  "links": [[1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [6, 7]]
}
```

- [ ] **Step 2: Write the failing test**

`backend/tests/test_workflow_builder.py`:
```python
import json
import os
import pytest
from backend.pipeline.workflow_builder import build_workflow, fill_prompts


@pytest.fixture
def base_workflow():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "mock_base_workflow.json")
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture
def sample_template():
    return {
        "id": "char_action_fight",
        "positive_prompt": "(masterpiece:1.2), {character_name}, battle stance",
        "negative_prompt": "blurry, low quality",
        "workflow_params": {
            "steps": 30,
            "cfg_scale": 8.0,
            "width": 1024,
            "height": 768,
            "sampler": "dpmpp_2m_karras",
            "lora_strength": 0.75
        }
    }


def test_fill_prompts_replaces_placeholders(base_workflow, sample_template):
    character_name = "艾琳"
    filled = fill_prompts(base_workflow, sample_template, character_name)

    workflow_str = json.dumps(filled)
    assert "{PROMPT}" not in workflow_str
    assert "{NEGATIVE}" not in workflow_str
    assert "{character_name}" not in workflow_str
    assert "艾琳" in workflow_str
    assert "(masterpiece:1.2)" in workflow_str


def test_build_workflow_sets_ksampler_params(base_workflow, sample_template):
    result = build_workflow(sample_template, base_workflow, "艾琳")

    ksampler = None
    for node in result["nodes"].values():
        if node["type"] == "KSampler":
            ksampler = node
            break

    assert ksampler is not None
    assert ksampler["inputs"]["steps"] == 30
    assert ksampler["inputs"]["cfg"] == 8.0
    assert ksampler["inputs"]["sampler_name"] == "dpmpp_2m_karras"


def test_build_workflow_sets_lora_strength(base_workflow, sample_template):
    result = build_workflow(sample_template, base_workflow, "艾琳")

    lora = None
    for node in result["nodes"].values():
        if node["type"] == "LoraLoader":
            lora = node
            break

    assert lora is not None
    assert lora["inputs"]["strength_model"] == 0.75
    assert lora["inputs"]["strength_clip"] == 0.75


def test_build_workflow_does_not_mutate_input(base_workflow, sample_template):
    original = json.dumps(base_workflow)
    build_workflow(sample_template, base_workflow, "艾琳")
    assert json.dumps(base_workflow) == original
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd g:/img-project && python -m pytest backend/tests/test_workflow_builder.py -v
```

Expected: FAIL — module not found

- [ ] **Step 4: Write minimal implementation**

`backend/pipeline/workflow_builder.py`:
```python
import json
import copy


def fill_prompts(workflow: dict, template: dict, character_name: str) -> dict:
    """Replace prompt placeholders in workflow JSON with template values."""
    workflow_str = json.dumps(workflow)
    positive = template["positive_prompt"].replace("{character_name}", character_name)
    negative = template["negative_prompt"].replace("{character_name}", character_name)
    workflow_str = workflow_str.replace("{PROMPT}", positive)
    workflow_str = workflow_str.replace("{NEGATIVE}", negative)
    return json.loads(workflow_str)


def build_workflow(template: dict, base_workflow: dict, character_name: str) -> dict:
    """Build a complete ComfyUI workflow by filling a base skeleton with template params.

    Does not mutate the input base_workflow.
    """
    workflow = fill_prompts(copy.deepcopy(base_workflow), template, character_name)
    params = template.get("workflow_params", {})

    for node in workflow.get("nodes", {}).values():
        node_type = node.get("type", "")

        if node_type == "KSampler":
            if "steps" in params:
                node["inputs"]["steps"] = params["steps"]
            if "cfg_scale" in params:
                node["inputs"]["cfg"] = params["cfg_scale"]
            if "sampler" in params:
                node["inputs"]["sampler_name"] = params["sampler"]

        elif node_type == "LoraLoader":
            if "lora_strength" in params:
                node["inputs"]["strength_model"] = params["lora_strength"]
                node["inputs"]["strength_clip"] = params["lora_strength"]

        elif node_type == "EmptyLatentImage":
            if "width" in params:
                node["inputs"]["width"] = params["width"]
            if "height" in params:
                node["inputs"]["height"] = params["height"]

    return workflow
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd g:/img-project && python -m pytest backend/tests/test_workflow_builder.py -v
```

Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
cd g:/img-project && git add backend/pipeline/workflow_builder.py backend/tests/test_workflow_builder.py backend/tests/fixtures/ && git commit -m "feat: add workflow builder with placeholder substitution"
```

---

### Task 5: ComfyUI Client (comfyui_client.py)

**Files:**
- Create: `backend/pipeline/comfyui_client.py`
- Create: `backend/tests/test_comfyui_client.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_comfyui_client.py`:
```python
import pytest
import httpx
from unittest.mock import patch, AsyncMock
from backend.pipeline.comfyui_client import ComfyUIClient


@pytest.fixture
def client():
    return ComfyUIClient(base_url="http://localhost:8188", timeout=10)


@pytest.fixture
def sample_workflow():
    return {"nodes": {"1": {"type": "KSampler"}}, "links": []}


@pytest.mark.asyncio
async def test_submit_workflow_returns_prompt_id(client, sample_workflow):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prompt_id": "abc123"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        prompt_id = await client.submit_workflow(sample_workflow)

    assert prompt_id == "abc123"


@pytest.mark.asyncio
async def test_submit_workflow_raises_on_error(client, sample_workflow):
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            await client.submit_workflow(sample_workflow)


@pytest.mark.asyncio
async def test_wait_for_result_returns_image_bytes(client):
    mock_history = AsyncMock()
    mock_history.status_code = 200
    mock_history.raise_for_status = MagicMock()
    mock_history.json.return_value = {
        "abc123": {
            "outputs": {
                "7": {
                    "images": [
                        {"filename": "output_00001.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
    }

    mock_image = AsyncMock()
    mock_image.status_code = 200
    mock_image.raise_for_status = MagicMock()
    mock_image.content = b"fake_image_bytes"

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = [mock_history, mock_image]
        result = await client.wait_for_result("abc123")

    assert result == b"fake_image_bytes"


@pytest.mark.asyncio
async def test_wait_for_result_retries_until_ready(client):
    """First two polls return empty history, third returns the image."""
    mock_empty = AsyncMock()
    mock_empty.status_code = 200
    mock_empty.raise_for_status = MagicMock()
    mock_empty.json.return_value = {}

    mock_ready = AsyncMock()
    mock_ready.status_code = 200
    mock_ready.raise_for_status = MagicMock()
    mock_ready.json.return_value = {
        "abc123": {
            "outputs": {
                "7": {
                    "images": [
                        {"filename": "out.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
    }

    mock_image = AsyncMock()
    mock_image.status_code = 200
    mock_image.raise_for_status = MagicMock()
    mock_image.content = b"delayed_result"

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = [mock_empty, mock_empty, mock_ready, mock_image]
        result = await client.wait_for_result("abc123")

    assert result == b"delayed_result"


@pytest.mark.asyncio
async def test_wait_for_result_raises_timeout(client):
    mock_empty = AsyncMock()
    mock_empty.status_code = 200
    mock_empty.raise_for_status = MagicMock()
    mock_empty.json.return_value = {}

    with patch("httpx.AsyncClient.get", return_value=mock_empty):
        with pytest.raises(TimeoutError, match="timed out"):
            await client.wait_for_result("abc123")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd g:/img-project && python -m pytest backend/tests/test_comfyui_client.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

`backend/pipeline/comfyui_client.py`:
```python
import time
import asyncio
import httpx


class ComfyUIClient:
    """Async client for ComfyUI's REST API."""

    def __init__(self, base_url: str = "http://localhost:8188", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def submit_workflow(self, workflow: dict) -> str:
        """Submit a workflow JSON, return prompt_id."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/prompt",
                json={"prompt": workflow},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["prompt_id"]

    async def wait_for_result(self, prompt_id: str) -> bytes:
        """Poll ComfyUI history until the result image is ready.

        Returns raw image bytes. Raises TimeoutError if generation exceeds timeout.
        """
        start = time.monotonic()
        async with httpx.AsyncClient() as client:
            while time.monotonic() - start < self.timeout:
                resp = await client.get(
                    f"{self.base_url}/api/history/{prompt_id}",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    for node_output in outputs.values():
                        images = node_output.get("images", [])
                        if images:
                            img_info = images[0]
                            img_resp = await client.get(
                                f"{self.base_url}/api/view",
                                params={
                                    "filename": img_info["filename"],
                                    "subfolder": img_info.get("subfolder", ""),
                                    "type": img_info.get("type", "output"),
                                },
                                timeout=30,
                            )
                            img_resp.raise_for_status()
                            return img_resp.content

                await asyncio.sleep(1)

        raise TimeoutError(f"ComfyUI generation timed out after {self.timeout}s")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd g:/img-project && python -m pytest backend/tests/test_comfyui_client.py -v
```

Expected: 5 PASS (may need `pytest-asyncio` configured — add `pytest.ini` if needed)

- [ ] **Step 5: Add pytest config if needed**

`pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
testpaths = backend/tests
```

- [ ] **Step 6: Commit**

```bash
cd g:/img-project && git add backend/pipeline/comfyui_client.py backend/tests/test_comfyui_client.py pytest.ini && git commit -m "feat: add ComfyUI async API client"
```

---

### Task 6: FastAPI Server + Pipeline Orchestration (main.py)

**Files:**
- Create: `backend/main.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Write the failing integration test**

`backend/tests/test_main.py`:
```python
import pytest
import base64
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from backend.main import app, tasks


@pytest.fixture
def client():
    tasks.clear()
    return TestClient(app)


def test_generate_returns_task_id(client):
    with patch("backend.main.asyncio.create_task"):
        resp = client.post("/api/generate", json={"prompt": "生成艾琳的战斗姿态"})

    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert len(data["task_id"]) == 8


def test_get_result_returns_404_for_unknown_task(client):
    resp = client.get("/api/result/nonexist")
    assert resp.status_code == 404


def test_get_result_returns_processing_status(client):
    tasks["test123"] = {
        "status": "processing",
        "steps": [
            {"name": "intent", "status": "done"},
            {"name": "template", "status": "in_progress"}
        ],
        "image": None,
        "info": None,
        "error": None,
    }

    resp = client.get("/api/result/test123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert len(data["steps"]) == 2


def test_get_result_returns_done_with_image(client):
    tasks["done1"] = {
        "status": "done",
        "steps": [
            {"name": "intent", "status": "done"},
            {"name": "template", "status": "done"},
            {"name": "workflow", "status": "done"},
            {"name": "generate", "status": "done"},
        ],
        "image": "iVBORw0KGgo=",
        "info": {"character": "艾琳", "template": "char_action_fight", "elapsed": 12.3},
        "error": None,
    }

    resp = client.get("/api/result/done1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["image"] == "iVBORw0KGgo="
    assert data["info"]["character"] == "艾琳"


def test_frontend_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd g:/img-project && python -m pytest backend/tests/test_main.py -v
```

Expected: FAIL — module not found or import error

- [ ] **Step 3: Write minimal implementation**

`backend/main.py`:
```python
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
```

- [ ] **Step 4: Prepare for test — create a minimal frontend placeholder**

```bash
echo "<html><body>Demo</body></html>" > g:/img-project/frontend/index.html
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd g:/img-project && python -m pytest backend/tests/test_main.py -v
```

Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
cd g:/img-project && git add backend/main.py backend/tests/test_main.py frontend/index.html && git commit -m "feat: add FastAPI server with pipeline orchestration"
```

---

### Task 7: Frontend (index.html)

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: Write the complete single-page UI**

Read current placeholder first, then replace it:

`frontend/index.html`:
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Game Art Agent Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f14;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .container { max-width: 900px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 24px; margin-bottom: 24px; color: #fff; }

  .input-section {
    background: #1a1a24; border-radius: 12px; padding: 20px; margin-bottom: 24px;
    border: 1px solid #2a2a3a;
  }
  .input-section textarea {
    width: 100%; height: 60px; background: #0f0f14; border: 1px solid #2a2a3a;
    border-radius: 8px; padding: 12px; color: #e0e0e0; font-size: 15px;
    resize: vertical; font-family: inherit;
  }
  .input-section textarea:focus { outline: none; border-color: #6c5ce7; }
  .input-section button {
    margin-top: 12px; padding: 10px 28px; background: #6c5ce7; color: #fff;
    border: none; border-radius: 8px; font-size: 15px; cursor: pointer;
    transition: background 0.2s;
  }
  .input-section button:hover { background: #7d6ff0; }
  .input-section button:disabled { background: #3a3a4a; cursor: not-allowed; }

  .result-section {
    background: #1a1a24; border-radius: 12px; padding: 20px;
    border: 1px solid #2a2a3a; min-height: 200px;
  }
  .result-section h2 { font-size: 18px; margin-bottom: 16px; }

  .steps { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .step {
    padding: 4px 12px; border-radius: 6px; font-size: 13px;
    background: #2a2a3a; color: #888;
  }
  .step.done { background: #1a3a2a; color: #4ade80; }
  .step.in_progress { background: #3a2a1a; color: #facc15; animation: pulse 1s infinite; }
  .step.failed { background: #3a1a1a; color: #f87171; }
  @keyframes pulse { 50% { opacity: 0.6; } }

  .image-area {
    text-align: center; margin-bottom: 16px;
  }
  .image-area img {
    max-width: 100%; max-height: 500px; border-radius: 8px;
    border: 1px solid #2a2a3a;
  }
  .image-area .placeholder {
    display: flex; align-items: center; justify-content: center;
    height: 200px; color: #555; font-size: 14px;
  }

  .info { font-size: 13px; color: #888; margin-bottom: 16px; }
  .info span { margin-right: 16px; }

  .actions { display: flex; gap: 12px; justify-content: center; }
  .actions button {
    padding: 8px 24px; border-radius: 8px; font-size: 14px; cursor: pointer;
    border: 1px solid #2a2a3a; background: transparent; color: #e0e0e0;
  }
  .actions button.primary { background: #6c5ce7; border-color: #6c5ce7; color: #fff; }
  .actions button:hover { opacity: 0.85; }
  .actions button:disabled { opacity: 0.4; cursor: not-allowed; }
  .actions button.danger { border-color: #f87171; color: #f87171; }

  .error { background: #3a1a1a; border: 1px solid #f87171; border-radius: 8px;
    padding: 12px; color: #f87171; font-size: 14px; margin-bottom: 16px; }
</style>
</head>
<body>
<div class="container">
  <h1>Game Art Agent Demo</h1>

  <div class="input-section">
    <textarea id="promptInput" placeholder="描述你想要的图片，例如：生成一张艾琳的战斗姿态"></textarea>
    <button id="generateBtn" onclick="generate()">生成图片</button>
  </div>

  <div class="result-section" id="resultSection">
    <h2>生成结果</h2>
    <div class="steps" id="steps"></div>
    <div class="error" id="errorMsg" style="display:none"></div>
    <div class="image-area" id="imageArea">
      <div class="placeholder">等待输入 prompt 开始生成</div>
    </div>
    <div class="info" id="infoArea"></div>
    <div class="actions" id="actions" style="display:none">
      <button class="primary" onclick="saveImage()">确认保存</button>
      <button class="danger" onclick="retry()">重新生成</button>
    </div>
  </div>
</div>

<script>
let currentTaskId = null;
let pollTimer = null;

const stepNames = {
  intent: "意图识别", template: "模板匹配",
  workflow: "构建工作流", generate: "图片生成", error: "错误"
};

async function generate() {
  const prompt = document.getElementById("promptInput").value.trim();
  if (!prompt) return;

  const btn = document.getElementById("generateBtn");
  btn.disabled = true;
  btn.textContent = "处理中...";

  resetUI();

  try {
    const resp = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt })
    });
    const data = await resp.json();
    currentTaskId = data.task_id;
    poll();
  } catch (e) {
    showError("请求失败: " + e.message);
    btn.disabled = false;
    btn.textContent = "生成图片";
  }
}

function resetUI() {
  document.getElementById("steps").innerHTML = "";
  document.getElementById("errorMsg").style.display = "none";
  document.getElementById("imageArea").innerHTML = '<div class="placeholder">处理中...</div>';
  document.getElementById("infoArea").innerHTML = "";
  document.getElementById("actions").style.display = "none";
}

function poll() {
  if (!currentTaskId) return;

  fetch("/api/result/" + currentTaskId)
    .then(r => r.json())
    .then(data => {
      renderSteps(data.steps);
      renderError(data.error);

      if (data.status === "done") {
        stopPolling();
        renderImage(data.image);
        renderInfo(data.info);
        showActions();
        document.getElementById("generateBtn").disabled = false;
        document.getElementById("generateBtn").textContent = "生成图片";
      } else if (data.status === "error") {
        stopPolling();
        document.getElementById("generateBtn").disabled = false;
        document.getElementById("generateBtn").textContent = "生成图片";
      }
    })
    .catch(e => {
      stopPolling();
      showError("轮询失败: " + e.message);
    });
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function renderSteps(steps) {
  if (!steps) return;
  const el = document.getElementById("steps");
  el.innerHTML = steps.map(s => {
    const name = stepNames[s.name] || s.name;
    return `<span class="step ${s.status}">${name} ${s.status === 'in_progress' ? '...' : '✓'}</span>`;
  }).join("");
}

function renderError(error) {
  const el = document.getElementById("errorMsg");
  if (error) {
    el.style.display = "block";
    el.textContent = error;
  } else {
    el.style.display = "none";
  }
}

function renderImage(imageB64) {
  const el = document.getElementById("imageArea");
  if (imageB64) {
    el.innerHTML = `<img src="${imageB64}" alt="生成结果" />`;
  }
}

function renderInfo(info) {
  if (!info) return;
  const el = document.getElementById("infoArea");
  const parts = [];
  if (info.character) parts.push(`角色: ${info.character}`);
  if (info.template_desc) parts.push(`模板: ${info.template_desc}`);
  if (info.elapsed) parts.push(`耗时: ${info.elapsed}s`);
  el.innerHTML = parts.map(p => `<span>${p}</span>`).join("");
}

function showActions() {
  document.getElementById("actions").style.display = "flex";
}

function showError(msg) {
  const el = document.getElementById("errorMsg");
  el.style.display = "block";
  el.textContent = msg;
}

function saveImage() {
  const img = document.querySelector("#imageArea img");
  if (!img) return;
  const a = document.createElement("a");
  a.href = img.src;
  a.download = "generated_" + currentTaskId + ".png";
  a.click();
}

function retry() {
  document.getElementById("generateBtn").click();
}

// Start polling when generate is called
const origGenerate = generate;
generate = function() {
  stopPolling();
  origGenerate().then(() => {
    pollTimer = setInterval(poll, 1000);
  });
};
</script>
</body>
</html>
```

- [ ] **Step 2: Verify frontend is served**

```bash
cd g:/img-project && python -m pytest backend/tests/test_main.py::test_frontend_served -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
cd g:/img-project && git add frontend/index.html && git commit -m "feat: add single-page frontend UI"
```

---

### Task 8: End-to-End Verification

**Files:**
- Create: `backend/base_workflow.json` (placeholder — hermes provides the real one)

- [ ] **Step 1: Create placeholder for base_workflow.json**

Copy the mock fixture as placeholder (will be replaced by hermes):

```bash
cp g:/img-project/backend/tests/fixtures/mock_base_workflow.json g:/img-project/backend/base_workflow.json
```

- [ ] **Step 2: Create .env file from example**

```bash
cp g:/img-project/.env.example g:/img-project/.env
```

Then instruct the user to edit `.env` with their actual DashScope API key.

- [ ] **Step 3: Run all tests**

```bash
cd g:/img-project && python -m pytest backend/tests/ -v
```

Expected: all tests pass

- [ ] **Step 4: Start the server**

```bash
cd g:/img-project && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] **Step 5: Manual smoke test**

1. Open `http://localhost:8000` in browser
2. Type a test prompt: "生成艾琳的战斗姿态"
3. Verify UI shows processing steps
4. Verify result image or clear error message is displayed

- [ ] **Step 6: Commit**

```bash
cd g:/img-project && git add backend/base_workflow.json .env.example && git commit -m "feat: add placeholder base_workflow and e2e setup"
```

---

## Prerequisites from hermes (blocking tasks)

Before Task 7 (ComfyUI integration) can work end-to-end:

1. ComfyUI Python server running on `localhost:8188`
2. `last.safetensors` placed in `ComfyUI/models/loras/`
3. SD 1.5 checkpoint in `ComfyUI/models/checkpoints/`
4. Workflow built and exported — replacing `backend/base_workflow.json`
5. API verified with curl (submit + get result)

The server will start and serve the frontend without ComfyUI — but generation will fail at the ComfyUI step until hermes delivers these items.
