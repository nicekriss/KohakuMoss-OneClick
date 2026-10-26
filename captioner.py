"""Built-in captioner for KohakuMoss OneClick, used when LM Studio isn't running.

  python captioner.py <jobs.json> <out.json>

jobs.json: {"prompt": "...", "images": ["a.jpg", ...], "translate": "optional prompt"}
  ->  out.json: {"a.jpg": "caption", ..., "__translate__": "english"}
Runs Qwen3-VL-2B-Instruct with the transformers that ships in the KohakuMoss venv. It is a separate process so its
VRAM is gone before training starts. The first run downloads the model (~4.5 GB) into the Hugging Face cache.
"""
import json, re, sys

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def main():
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    out = {}
    if jobs.get("translate"):  # a Korean description, turned into English with the same model before captioning
        messages = [{"role": "user", "content": [{"type": "text", "text": jobs["translate"]}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        out["__translate__"] = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().strip('".').lower()
        jobs["prompt"] = jobs["prompt"].replace("{cls}", out["__translate__"])
        print(f"  번역: {out['__translate__']}", flush=True)
    for i, path in enumerate(jobs["images"], 1):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((768, 768))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": jobs["prompt"]}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        cap = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out[path] = re.sub(r"\s+", " ", cap).strip().strip('"').rstrip(".")
        print(f"  캡션 {i}/{len(jobs['images'])}", flush=True)
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
