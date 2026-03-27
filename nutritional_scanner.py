import base64
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_prompt() -> str:
    return """
You are a nutrition assistant specialized in PCOS.

From the image:
1. Identify main food items
2. Estimate nutritional values:
   - calories
   - protein
   - carbs
   - sugar
   - fat
   - fiber

3. Give a health score (1-10):
   - high sugar or refined carbs -> lower score
   - high fat/oil -> lower score
   - high protein and fiber -> higher score
   - Do not provide any nutritional info if image is not a food item.

4. Assign color:
   - 8-10 -> GREEN (healthy)
   - 5-7 -> YELLOW (moderate)
   - 1-4 -> RED (avoid)

5. Give a short overall statement (max 12 words)

Return ONLY JSON:
{
  "foods": ["name"],
  "nutrition": {
    "calories": number,
    "protein": number,
    "carbs": number,
    "sugar": number,
    "fat": number,
    "fiber": number
  },
  "score": number,
  "color": "GREEN | YELLOW | RED",
  "overall": "short health summary"
}
"""


def run_nutrition_agent(image_base64: str) -> dict:
    if not API_KEY:
        raise ValueError("NVIDIA_API_KEY is missing")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": "microsoft/phi-3.5-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.2,
        "top_p": 0.7,
        "stream": False
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    raw_output = result["choices"][0]["message"]["content"]
    clean_output = raw_output.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_output)


def scan_nutrition(image_path: str = None, image_base64: str = None) -> dict:
    if image_base64:
        return run_nutrition_agent(image_base64)
    if image_path:
        return run_nutrition_agent(encode_image(image_path))
    raise ValueError("Provide either image_path or image_base64")


if __name__ == "__main__":
    output = scan_nutrition(image_path="002.jpg")
    print(json.dumps(output, indent=2))