import json
import dashscope
from dashscope import Generation
from backend.config import DASHSCOPE_API_KEY, QWEN_MODEL

dashscope.api_key = DASHSCOPE_API_KEY

REWRITE_SYSTEM_PROMPT = """You are an expert prompt engineer for SDXL image generation with a Chinese style LoRA. Given the user's Chinese description, output both a positive prompt and a negative prompt optimized for SDXL.

Positive prompt rules:
- Always start with "guofeng, Chinese style, masterpiece, best quality,"
- Use simple concrete English words
- Describe subject, action, setting, composition, and style in that order
- Keep under 80 words, no weight notation

Negative prompt rules:
- Identify potential quality issues specific to the scene (e.g., bad hands for characters, bad perspective for landscapes, style pollution for traditional art)
- Always include: "blurry, low quality, jpeg artifacts, watermark, text, signature, lowres"
- For character scenes add: "bad anatomy, extra limbs, mutated hands, extra fingers, deformed face, western armor, 3D render, photorealistic"
- For landscapes add: "people, cars, modern buildings, power lines, ugly architecture, photorealistic, 3D render"
- For traditional Chinese art style add: "oil painting, western style, cartoon, anime, 3D render, photorealistic"
- Keep under 50 words

Return ONLY valid JSON:
{"positive": "the positive prompt text", "negative": "the negative prompt text"}"""


def rewrite_prompt(user_input: str) -> dict:
    """Rewrite user's Chinese input into optimized SDXL prompts via Qwen. Returns {"positive": ..., "negative": ...}."""
    if not user_input or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")

    try:
        response = Generation.call(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"请为以下描述生成正负向提示词：{user_input}"}
            ],
            temperature=0.3,
            result_format="message"
        )

        if response.status_code != 200:
            raise Exception(f"Qwen API error: status={response.status_code}")

        content = response.output.choices[0].message.content
        result = json.loads(content)
        return {"positive": result["positive"].strip(), "negative": result["negative"].strip()}

    except json.JSONDecodeError:
        raise Exception(f"Prompt rewrite failed: invalid JSON response: {content}")
    except Exception as e:
        raise Exception(f"Prompt rewrite failed: {str(e)}")
