import requests
from typing import Any, Optional, Dict, List

class OllamaService:
    def __init__(self, config, agent_type: str):
        self.model = config[agent_type]["API_MODEL"]
        self.api_base = config[agent_type]["API_BASE"]

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        n: int = 1,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        # Flatten message content
        prompt_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                prompt_parts.extend(str(c) for c in content)
            else:
                prompt_parts.append(str(content))
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(f"{self.api_base}/api/generate", json=payload)
        result = response.json()

        # Safely extract response or error
        if "response" in result:
            return [result["response"]], 0.0
        elif "error" in result:
            error_msg = result["error"]
            return [f"[OLLAMA ERROR] {error_msg}"], 0.0
        else:
            return ["[OLLAMA ERROR] Unknown response format"], 0.0
