# Game Art Agent — 智能国风图像生成系统

基于 **Qwen 大模型 + SDXL + LoRA** 的流水线式图像生成工具。用户输入中文描述，Qwen 自动改写为优化的英文提示词，由 ComfyUI 执行生图。

---

## 系统架构

```
用户中文输入 → Qwen 智能改写 → 英文 SDXL 提示词 → ComfyUI + 国风 LoRA → 出图
```

### 两种工作模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **智能改写**（默认） | Qwen 将中文描述改写为英文 SDXL 提示词，并自动生成场景匹配的负向提示词 | 日常使用，效果最好 |
| 自由输入 | 直接输入英文提示词，不经改写直接送入 SD | 手动调试 prompt |

---

## 模型微调细节

本系统使用的国风 LoRA 文件 `chinese-style-sdxl.safetensors` 的技术参数如下：

| 参数 | 值 |
|------|-----|
| 文件名 | `chinese-style-sdxl.safetensors` |
| 文件大小 | ~325 MB |
| 基座模型 | SDXL 1.0 |
| LoRA Rank | 64 |
| LoRA Alpha | 32 |
| 精度 | float16 |
| 训练目标 | UNet（未训练 text encoder） |
| 触发词 | ` Chinese style` |
| 推荐权重 | 0.65（model 和 clip 均使用此值） |

由于未训练 text encoder，提示词中的触发词是激活风格的关键——提示词**必须**以 ` Chinese style` 开头才能触发国风效果。Qwen 智能改写模式会自动添加触发词前缀，自由输入模式下用户需自行添加。

推荐采样参数：**30 步基础采样 + 1.5× 潜空间超分 + 15 步高分辨率修复**，CFG 6.5/6.0，采样器 `dpmpp_2m` + `karras`。

---

## 启动命令

### 前置条件

- NVIDIA 显卡（推荐 8GB+ 显存，6GB 需开启 lowvram）
- Python 3.12（推荐，3.14 暂不支持 PyTorch CUDA）
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（Python 版）
- Qwen API Key（[DashScope](https://dashscope.aliyun.com/) 申请）

### 1. 启动 ComfyUI

```powershell
cd D:\ComfyUI_python
.venv\Scripts\activate.bat
python main.py --lowvram
```

看到 `To see the GUI go to: http://127.0.0.1:8188` 表示启动成功。

> 如果显存 >= 8GB 可去掉 `--lowvram`，出图更快。

### 2. 配置环境变量

将项目根目录下的 `.env.example` 重命名为 `.env`，填入你的 DashScope API Key：

```
DASHSCOPE_API_KEY=你的API_KEY
COMFYUI_URL=http://localhost:8188
QWEN_MODEL=qwen-max
COMFYUI_TIMEOUT=600
```

### 3. 安装依赖

```powershell
cd G:\img-project
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 4. 启动后端

```powershell
python -m uvicorn backend.main:app --port 8001
```

### 5. 打开前端

浏览器访问 `http://localhost:8001`，即可输入描述开始生图。

---

## 文件结构

### 后端 (`backend/`)

| 文件 | 任务 |
|------|------|
| `main.py` | FastAPI 服务入口。定义 `/api/generate`（提交生图任务）、`/api/result/{task_id}`（轮询结果）、`/api/history`（历史记录）三个接口，以及核心流水线 `run_pipeline()`（处理两种工作模式的调度、工作流构造、ComfyUI 提交、结果提取和历史保存） |
| `config.py` | 从 `.env` 读取配置：DashScope API Key、ComfyUI 地址、Qwen 模型名、超时时间等，并导出两个工作流文件路径 |
| `base_workflow.json` | 带 LoRA 的 ComfyUI 工作流模板。包含 `{PROMPT}` 和 `{NEGATIVE}` 占位符，运行时由后端替换。流程：加载 Checkpoint → 加载 LoRA → 双 CLIP 编码 → 空潜空间 → 基础 KSampler → 1.5× 潜空间超分 → HiresFix KSampler → VAE 解码 → 保存 |
| `base_workflow_nolora.json` | 不含 LoRA 节点的 ComfyUI 工作流模板，其余结构与带 LoRA 版相同。用于自由输入模式下不加载风格 LoRA 时使用 |
| `generation_history.json` | 自动生成的历史记录文件，保存最近 50 条生图记录（task_id、提示词、模板类型、耗时、时间戳） |

### 流水线 (`backend/pipeline/`)

| 文件 | 任务 |
|------|------|
| `intent.py` | 封装 Qwen API 调用，提供 `rewrite_prompt()` 函数：将用户的中文描述改写为 SDXL 友好的英文提示词（包含触发词前缀），同时根据场景类型自动生成差异化的负向提示词 |
| `comfyui_client.py` | ComfyUI 异步 API 客户端。提交工作流 JSON 到 `/api/prompt`，轮询 `/api/history/{prompt_id}` 获取生成结果图片 |

### 测试 (`backend/tests/`)

| 文件 | 任务 |
|------|------|
| `test_main.py` | 测试三个 API 接口的正确响应（task_id 格式、404 处理、状态轮询），以及前端文件服务是否正常 |
| `test_intent.py` | 测试 `rewrite_prompt()` 的正向/负向提示词输出、空白字符裁剪、API 错误处理（非 200 状态码、非法 JSON、网络异常）、空输入校验 |
| `test_comfyui_client.py` | 测试 ComfyUI 客户端的工作流提交、结果轮询重试逻辑、超时异常处理 |

### 前端 (`frontend/`)

| 文件 | 任务 |
|------|------|
| `index.html` | 单页 Web 界面。包含模式切换（智能改写/自由输入）、LoRA 开关、提示词输入框、生成进度步骤展示、结果图片预览、正向/负向提示词展示、确认保存/重新生成按钮、历史记录面板（支持一键复用提示词） |

### 根目录

| 文件 | 任务 |
|------|------|
| `requirements.txt` | Python 依赖清单（FastAPI、DashScope、httpx、pytest 等） |
| `pytest.ini` | pytest 配置（asyncio 自动模式、测试路径） |
| `.env.example` | 环境变量模板，供用户复制为 `.env` 填入真实配置 |
| `README.md` | 项目说明文档 |

---

## 主要功能

1. **中文智能改写** — Qwen 将中文白描转化为 SDXL 友好的英文提示词，无需掌握 prompt engineering
2. **负向提示词自动生成** — 根据场景类型自动生成差异化负向词（人物→排除畸形手部；风景→排除现代建筑；国风→排除3D渲染）
3. **提示词记录与复用** — 每次生成保存正负向提示词，历史面板可一键复用
4. **种子随机化** — 每次生成自动随机种子，避免重复出图
5. **两种输入模式** — 智能改写 / 自由输入，灵活切换
6. **本地运行** — 完全本地化，无次数限制，数据不外传
7. **触发词隐藏** — 模型触发词仅在内部工作流中使用，前端不暴露

---

## 与现有 AI 生图服务的对比

| 特性 | 本项目 | 在线 AI 生图网站 |
|------|--------|-----------------|
| 输入语言 | **中文直接输入**，无需写英文 prompt | 必须写英文 prompt，需要技巧 |
| Prompt 优化 | Qwen **自动改写+扩展** | 用户自己写，无辅助 |
| 负向提示词 | **场景自适应**生成 | 需要用户手动填写 |
| 模型可控性 | 完全控制采样器/步数/CFG/**LoRA** | 黑盒模型，参数固定 |
| 风格一致性 | 可固定 LoRA，**垂直领域可控** | 通用模型，风格随机 |
| 成本 | 一次显卡投入，**无限使用** | 按次付费或订阅制 |
| 数据隐私 | **完全本地**，不传输到外部 | 图片上传到服务商服务器 |
| 历史记录 | 自动保存+**提示词复用** | 依赖浏览器历史 |
| 扩展性 | 可接入任何 SD 模型/LoRA | 限定平台提供的模型 |

---

## 未来扩展方向

- [ ] **迭代优化** — 对已生成图片提出修改意见（如"眼睛再大一点"），Qwen 定向修改 prompt 重新生成
  <details>
  <summary>实现思路</summary>

  当前流水线是单向的：输入描述 → 出图。迭代优化需要形成"反馈 → 修改 → 出图"的闭环。实现步骤：

  1. **新增 API 端点** `/api/refine`：接收 `task_id` 和用户的修改意见（中文文本）
  2. **构建 Qwen 改写 prompt**：将"原始正向提示词 + 用户修改意见"发给 Qwen，要求其仅修改对应部分，保留其余描述
  3. **复用工作流**：用修改后的提示词重新生成，保持 seed 不变（让未修改的部分视觉上保持一致），仅替换正负向 prompt
  4. **前端交互**：在生成结果的"重新生成"按钮旁增加"修改"按钮，弹出文本框让用户输入修改意见

  核心挑战在于 seed 复现——需在 `run_pipeline` 中保存每次生成的 seed，refine 时使用相同 seed。
  </details>

- [ ] **角色一致性** — 持久化角色外观特征，不同姿势/场景下保持同一角色形象
  <details>
  <summary>实现思路</summary>

  SDXL 难以仅通过 prompt 精确复现同一角色外观。实际可行的方案组合：

  1. **Prompt 固化**：在历史记录中为每个角色维护一份"外观描述模板"（发色、瞳色、服饰、体型），每次生成时自动拼入
  2. **Seed 锚定**：对同一角色使用固定 seed 范围（如取角色名 hash 作为 seed），结合少量随机偏移，SDXL 在相似 seed 下输出有相关性
  3. **长期方案 — IP-Adapter / FaceID**：在 ComfyUI 中安装 IP-Adapter 节点，为每个角色拍摄一张参考图，通过图像特征注入引导面部一致性。需要额外下载 IP-Adapter 模型和 CLIP Vision 编码器
  4. **长期方案 — 角色 LoRA**：收集同一角色的 10-20 张高质量图片，用 Kohya 脚本训练专属 LoRA

  短期内推荐方案 1+2，成本最低。方案 3 需要约 2GB 额外模型文件。方案 4 需要 GPU 训练时间约 30-60 分钟。
  </details>

- [ ] **参数智能推荐** — Qwen 根据场景类型自动推荐 CFG、步数、采样器组合
  <details>
  <summary>实现思路</summary>

  在 Qwen 改写 prompt 的同时，增加参数推理输出：

  1. **扩展 Qwen 输出格式**：在 `REWRITE_SYSTEM_PROMPT` 中增加参数推荐指令，将返回 JSON 从 `{"positive": ..., "negative": ...}` 扩展为 `{"positive": ..., "negative": ..., "params": {"steps": 30, "cfg": 6.5, "sampler": "dpmpp_2m", "scheduler": "karras"}}`
  2. **场景-参数映射知识**：在 system prompt 中嵌入规则（人物特写→低 CFG 避免过度锐化，风景→高 steps 增强细节，暗色调→euler ancestral 增加随机性）
  3. **后端应用**：从 Qwen 返回中解析 `params`，写入 KSampler 节点的对应字段
  4. **用户可覆盖**：前端保留手动参数调节选项，Qwen 推荐值作为默认值

  改动集中在 `intent.py`（修改 system prompt）和 `main.py`（应用推荐参数到 workflow）。
  </details>

- [ ] **批量生成套图** — 一个角色自动生成全身立绘、表情特写、战斗姿态全套
  <details>
  <summary>实现思路</summary>

  将单次生成扩展为并行/串行批量：

  1. **定义套图规格**：在前端或配置中定义套图模板列表，每项包含模板描述和覆盖参数（如头像特写用 768×1024 竖构图、全身立绘用 1024×1024）
  2. **Qwen 批量改写**：将角色名发送给 Qwen，要求其为"全身立绘、战斗姿态、表情特写、半身像"四个场景各生成一版提示词
  3. **并行提交 ComfyUI**：后端为每个场景构造独立 workflow，通过 `asyncio.gather` 并行提交到 ComfyUI（ComfyUI 支持队列，会依次执行）
  4. **前端展示**：结果以网格形式展示四张图，用户可逐张保存

  注意 ComfyUI 默认单 GPU 串行生图，并行提交仅减少 API 往返延迟。如需真正并行需多 GPU 或使用 ComfyUI 的 batch 功能。
  </details>

- [ ] **参考图反推** — 上传图片，Qwen 逆向生成 prompt，再在此基础上修改生成
  <details>
  <summary>实现思路</summary>

  需借助多模态视觉模型（Qwen-VL 或 CLIP Interrogator）：

  1. **方案 A — CLIP Interrogator**：在 ComfyUI 中安装 CLIP Interrogator 节点，本地运行，将上传图片反推为 prompt 文本，无需额外 API 费用
  2. **方案 B — Qwen-VL**：通过 DashScope 的 `qwen-vl-max` 模型，将图片以 base64 发送，要求其描述图片内容并转化为 SDXL prompt 格式
  3. **后端新增** `/api/interrogate` 端点：接收图片上传，调用反推服务，返回 prompt
  4. **前端交互**：增加图片上传区域，反推成功后自动填入 prompt 输入框，用户可在此基础上修改

  方案 A 完全本地且免费，但 prompt 质量不如大模型。方案 B 需要 DashScope 多模态 API 额度，但描述更准确。
  </details>

- [ ] **多 LoRA 混合** — 支持同时加载多个 LoRA，按权重混合风格

  在 ComfyUI 工作流中串联多个 LoraLoader 节点，每个节点加载不同的 LoRA 文件并设置独立权重。后端需扩展 `config.py`（支持 LoRA 列表配置）和 `main.py`（动态插入 LoraLoader 节点）。

---

## 效果展示
<img width="1085" height="946" alt="Snipaste_2026-05-27_04-37-46" src="https://github.com/user-attachments/assets/cac55f20-2d3a-40e2-991e-b8883bcf646f" />
<img width="1276" height="991" alt="Snipaste_2026-05-27_04-19-21" src="https://github.com/user-attachments/assets/ad083c26-2a5f-4748-8d6c-adcb3b1b1cca" />
<img width="958" height="586" alt="image" src="https://github.com/user-attachments/assets/2ce40b1c-3651-4914-b1bb-464de9f9d894" />
<img width="1536" height="1536" alt="output_00013_" src="https://github.com/user-attachments/assets/360787f4-bdd1-4073-9e98-a829c8c20608" />
<img width="1536" height="1536" alt="output_00002_" src="https://github.com/user-attachments/assets/2c8cbac9-9857-421c-9434-80627a2ad51b" />
<img width="1536" height="1536" alt="output_00003_" src="https://github.com/user-attachments/assets/b78c2046-94f2-48f5-bdea-ec45e62e805a" />
<img width="1536" height="1536" alt="output_00005_" src="https://github.com/user-attachments/assets/4a439b78-b3ca-4c98-8ed3-9852d0c08816" />
<img width="1536" height="1536" alt="output_00009_" src="https://github.com/user-attachments/assets/5e1bdd60-b6d3-4ef7-83bb-088861b63d5f" />
<img width="1536" height="1536" alt="output_00017_" src="https://github.com/user-attachments/assets/96b62d40-a034-40a5-9040-1372b1f99fe9" />
<img width="1536" height="1536" alt="output_00006_" src="https://github.com/user-attachments/assets/8146f59e-b4df-49e6-8aa4-9f1483469d16" />
<img width="1536" height="1536" alt="output_00004_" src="https://github.com/user-attachments/assets/4a0d821d-c5af-4e34-ad7f-f5c25b753e6e" />
<img width="1536" height="1536" alt="output_00010_" src="https://github.com/user-attachments/assets/3c7d3d3d-a932-4fe8-80fe-f463641cecb9" />


