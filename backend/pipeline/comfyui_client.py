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
