"""
Frozen judge for PPO – meta-llama/Llama-3.2-3B-Instruct.
"""

import re, json, torch
from transformers import pipeline
from .prompts import SYSTEM_JUDGE

class LlamaJudge:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    @torch.inference_mode()
    def score(self, note: str, prediction: str, gold: str):
        msgs = [
            {"role": "system", "content": SYSTEM_JUDGE},
            {"role": "user",
             "content": (f"<clinical_note>\n{note}\n</clinical_note>\n\n"
                         f"<ground_truth>\n{gold}\n</ground_truth>\n\n"
                         f"<model_output>\n{prediction}\n</model_output>")}
        ]
        out = self.pipe(msgs, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        js  = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        return js["accuracy"], js["cot_score"]
