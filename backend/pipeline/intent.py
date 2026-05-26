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
