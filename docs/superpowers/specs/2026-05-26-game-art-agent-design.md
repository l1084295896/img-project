# Game Art Agent Demo — Design Spec

**Date**: 2026-05-26
**Status**: Approved

---

## 1. 项目目标

为游戏美术部门构建一个 AI Agent Demo，美术人员输入自然语言描述，系统自动完成意图识别、RAG 模板匹配、ComfyUI 图像生成，输出角色美术图。

## 2. 技术选型

| 层面 | 选择 | 说明 |
|------|------|------|
| 前端 | HTML + JS 单页面 | 输入框 + 结果展示 + 人工确认 |
| 后端框架 | FastAPI | Python，轻量 |
| AI 编排 | LangChain | 线性 Pipeline |
| LLM | Qwen3 云端 API (DashScope) | 合并为一次调用：意图识别 + 参数提取 |
| RAG | 关键词匹配 | Prompt 模板库（JSON 文件，10-30 个），无向量数据库 |
| 图像生成 | ComfyUI API | 通过填充预置 Workflow JSON 骨架来调用 |
| LoRA 模型 | `last.safetensors` | kohya_ss 微调的 SD 1.5 角色模型，19 张图训练 |
| 质量检查 | 人工确认 | Demo 不做自动检查 |

## 3. 架构总览

```
用户输入 → ① 意图识别+参数提取(Qwen3) → ② 模板匹配(RAG) → ③ Workflow合成 → ④ ComfyUI生成 → ⑤ 前端展示 → 人工确认
```

### 各步骤职责

| 步骤 | 组件 | 输入 | 输出 |
|------|------|------|------|
| ① | IntentRecognizer + ParamExtractor (合并) | 用户自然语言 | `{category, intent, character, pose, mood, angle, ...}` |
| ② | TemplateMatcher | 结构化参数 | 最佳匹配的 Prompt 模板 JSON |
| ③ | WorkflowBuilder | 模板 + `base_workflow.json` | 完整的 ComfyUI Workflow JSON |
| ④ | ComfyUIClient | Workflow JSON | 生成图片 (base64) |
| ⑤ | 前端展示 | 图片 + 处理信息 | 用户确认/重试 |

### 外部依赖

- **Qwen3 API** (DashScope)：步骤 ①
- **ComfyUI Server** (本地 `localhost:8188`)：步骤 ④，由 hermes 负责搭建
- **Prompt 模板库**（本地 `templates/` 目录）：步骤 ②
- **LoRA 模型** (`models/loras/last.safetensors`)

## 4. Prompt 模板库设计

### 文件格式

每个模板一个 JSON 文件，存放在 `templates/` 目录。模板围绕"同一角色 × 不同需求场景"组织：

```json
{
  "id": "char_action_fight",
  "description": "角色战斗姿态",
  "tags": ["战斗", "动作", "动态", "武器", "张力"],
  "positive_prompt": "(masterpiece:1.2), {character_name}, battle stance, dynamic pose, action scene, dramatic lighting",
  "negative_prompt": "blurry, low quality, deformed body, extra fingers, watermark",
  "workflow_params": {
    "steps": 30,
    "cfg_scale": 8.0,
    "width": 1024,
    "height": 768,
    "sampler": "DPM++ 2M Karras",
    "lora_strength": 0.75
  }
}
```

- `{character_name}` 是占位符，由意图识别环节填入
- `tags` 用于关键词匹配
- `workflow_params` 是该模板下调试好的最佳 ComfyUI 参数

### 检索策略

1. 关键词匹配：计算提取参数与模板 tags 的交集得分，选最高分模板
2. 降级：匹配度低于阈值 (0.3) 时，使用 Qwen3 Embedding 做语义相似度匹配
3. 兜底：仍无匹配时提示用户补充描述

## 5. ComfyUI 集成

### 基础 Workflow 骨架

hermes 在 ComfyUI 界面中搭建最小工作流并导出为 `base_workflow.json`：

```
Load Checkpoint (SD 1.5) → Load LoRA (last.safetensors) → CLIP Encode (prompt) → KSampler → VAE Decode → Save Image
```

### 程序填充方式

程序运行时不是构建新 workflow，而是替换骨架中的占位符：

| 占位符 | 来源 |
|--------|------|
| `{PROMPT}` | 模板 `positive_prompt` + Qwen3 提取的具体描述 |
| `{NEGATIVE}` | 模板 `negative_prompt` |
| `lora_strength` | 模板 `workflow_params.lora_strength` |
| `steps`, `cfg`, `sampler`, `width`, `height` | 模板 `workflow_params` |

### API 调用

```
POST http://localhost:8188/api/prompt  (提交 workflow JSON)
GET  http://localhost:8188/api/history/{prompt_id}  (轮询获取结果)
```

## 6. 前端设计

单页面，三个区域：

1. **输入区**：文本输入框 + 生成按钮
2. **结果区**：生成的图片 + 处理信息（意图、匹配模板、耗时）
3. **操作区**：确认保存 / 重新生成

与后端通信：

- `POST /api/generate` — 提交生成请求，返回 `task_id`
- `GET /api/result/{task_id}` — 轮询获取结果（含图片 base64 和处理信息）

生成过程中前端展示 pipeline 各步骤状态。

## 7. 项目结构

```
g:/img-project/
├── backend/
│   ├── main.py                 # FastAPI 入口
│   ├── pipeline/
│   │   ├── intent.py           # Step ①: Qwen3 意图识别+参数提取
│   │   ├── template_matcher.py # Step ②: 模板匹配
│   │   ├── workflow_builder.py # Step ③: Workflow JSON 合成
│   │   └── comfyui_client.py   # Step ④: ComfyUI API 调用
│   ├── templates/              # Prompt 模板库
│   │   ├── char_stand_front.json
│   │   ├── char_action_fight.json
│   │   └── ...
│   └── base_workflow.json      # ComfyUI Workflow 骨架 (hermes 提供)
├── frontend/
│   └── index.html              # 单页面
└── requirements.txt
```

## 8. 错误处理

- Qwen3 API 失败：返回错误信息给前端，提示用户重试
- 模板匹配无结果：引导用户补充描述
- ComfyUI 超时/失败：返回错误信息，前端显示"生成失败，请重试"
- ComfyUI 离线：启动时健康检查，前端显示"服务不可用"

## 9. 项目依赖方

- **hermes**：ComfyUI Python 版搭建，导出 `base_workflow.json`，验证 API 调用
