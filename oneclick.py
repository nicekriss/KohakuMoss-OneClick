"""KohakuMoss OneClick — MOSS-Image LoRA from one photo folder: captions, training, and the LoRA in ComfyUI's list.

Runs with the KohakuMoss venv python (KohakuMoss_원클릭학습.bat finds it). Asks the ComfyUI models folder once and keeps it
in settings.json next to this file.

  모드 1 사람        : 옷·배경·표정을 캡션에 적음 → 옷은 프롬프트로 바뀜
  모드 2 캐릭터·의상 : 겉모습(의상)은 안 적음 → 의상까지 트리거에 고정
  모드 3 얼굴        : 얼굴만 잘라 둔 사진을 넣으면 표정·조명만 적음 (자동 크롭은 없음)

Captions: LM Studio's vision model when it is running, else the built-in Qwen3-VL-2B (captioner.py).
Built on KohakuMossTrainer (KohakuMoss License) — it must be installed separately.
"""
import argparse, base64, io, json, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "settings.json"
LM_URL = "http://127.0.0.1:1234/v1"
COMFY_PORTS = (8188, 8000, 8189)
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
HF_REPO = "Comfy-Org/MOSS-Image"
# KohakuMoss looks these names up in a ComfyUI models folder (first match wins); the last one is what we download
NEED = {
    "diffusion_models": ["moss_image_base_int8_convrot.safetensors", "moss_image_base_bf16.safetensors"],
    "text_encoders": ["qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_bf16.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors"],
    "vae": ["flux2-vae_small.safetensors", "flux2-vae.safetensors"],
}
DOWNLOAD = {"diffusion_models": ("moss_image_base_int8_convrot.safetensors", "5.0GB"),
            "text_encoders": ("qwen3.5_9b_w4a8_gptq.safetensors", "6.8GB"),
            "vae": ("flux2-vae_small.safetensors", "0.2GB")}

ASK = {
    "1": "Describe this photo for an image-model training caption, in one plain English sentence under 50 words: "
         "the person's clothing and accessories, pose and expression, camera angle and framing, background, lighting. "
         "Do not describe the face shape, age, ethnicity or identity. No lists, no quotes.",
    "2": "This single still image shows a recurring character: {cls}. Write one plain English phrase under 40 words that describes "
         "this image only: the pose or action, the expression, the camera angle and framing, the background, the lighting, and any "
         "props being held. Do NOT describe the character's own appearance, body or costume, and do not describe any movement over "
         "time. Start with the pose. No lists, no quotes.",
    "3": "This is a close-up crop of a face. In one plain English phrase under 20 words describe only the expression, head angle "
         "and lighting. Do not describe identity, age, hair, or clothing. No lists, no quotes.",
}


def log(msg, f=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if f:
        with open(f, "a", encoding="utf-8") as h:
            h.write(line + "\n")


def ask(prompt, check=lambda v: bool(v)):
    while True:
        v = input(prompt).strip().strip('"')
        if check(v):
            return v


# ---------------------------------------------------------------- setup: paths and models
def is_models_dir(v):
    if v and (Path(v) / "loras").is_dir():
        return True
    print("   그 폴더 안에 loras 폴더가 없어요. 컴피의 models 폴더를 다시 넣어 주세요.")
    return False


def load_settings():
    s = json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}
    if not s.get("models") or not Path(s["models"]).is_dir():
        print("\n 처음 한 번만 물어볼게요. 컴피(ComfyUI)의 models 폴더 경로를 넣어 주세요.")
        print(" (안에 diffusion_models, loras, vae 같은 폴더가 있는 곳. 예: C:\\ComfyUI\\models)")
        s["models"] = ask(" models 폴더: ", is_models_dir)
        SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    s["kohaku"] = str(Path(sys.prefix).parent)  # this runs in <KohakuMossTrainer>\.venv
    return s


def find_models(root):
    """Each needed file anywhere under its folder: {sub: path or None}."""
    found = {}
    for sub, names in NEED.items():
        found[sub] = None
        base = root / sub
        if not base.is_dir():
            continue
        for name in names:
            hits = sorted(base.rglob(name))
            if hits:
                found[sub] = hits[0]
                break
    return found


def download_missing(root, found):
    missing = [sub for sub, p in found.items() if p is None]
    if not missing:
        return found
    print("\n MOSS 학습에 필요한 모델이 없어요:")
    for sub in missing:
        name, size = DOWNLOAD[sub]
        print(f"   - {sub}\\{name} ({size})  https://huggingface.co/{HF_REPO}/resolve/main/{sub}/{name}")
    if ask(" 지금 받을까요? (y/n): ", lambda v: v.lower() in ("y", "n")).lower() != "y":
        raise SystemExit(" 위 링크에서 받아 models 폴더의 같은 이름 폴더에 넣고 다시 실행해 주세요.")
    from huggingface_hub import hf_hub_download
    for sub in missing:
        name, size = DOWNLOAD[sub]
        print(f" 받는 중: {name} ({size})")
        hf_hub_download(HF_REPO, f"{sub}/{name}", local_dir=root)
    return find_models(root)


def weights_dir(root, found):
    """KohakuMoss wants the three files directly under diffusion_models/text_encoders/vae of one folder.
    Files kept in subfolders get hard links (no extra disk space) in <models>\\_kohakumoss_weights."""
    if all(p.parent == root / sub for sub, p in found.items()):
        return root
    link_root = root / "_kohakumoss_weights"
    for sub, p in found.items():
        dst = link_root / sub / p.name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, dst)
        except OSError:
            raise SystemExit(f" {p} 를 {root / sub} 바로 아래로 옮겨 주세요 (하위 폴더가 다른 드라이브라 링크를 못 만들었어요).")
    return link_root


# ---------------------------------------------------------------- captions
def lm_studio_model():
    try:
        ids = [m["id"] for m in json.load(urllib.request.urlopen(LM_URL + "/models", timeout=3))["data"]]
    except Exception:
        return None
    vl = [i for i in ids if "vl" in i.lower()]
    return next((i for i in vl if "instruct" in i.lower()), vl[0] if vl else None)  # instruct answers without a think block


HANGUL = re.compile(r"[가-힣]")
TRANSLATE = ("Translate this short description of a person or character into natural English for an image caption. "
             "Answer with the English phrase only, lowercase, no quotes, no period: {text}")


def translate_lm(model, text):
    body = {"model": model, "max_tokens": 60, "temperature": 0.1,
            "messages": [{"role": "user", "content": TRANSLATE.format(text=text)}]}
    req = urllib.request.Request(LM_URL + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('".').lower()


def unload_lm_studio():
    """Free LM Studio's VRAM before training (its CLI ships with LM Studio)."""
    lms = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    if lms.exists():
        subprocess.run([str(lms), "unload", "--all"], capture_output=True)


def caption_lm(model, img_path, prompt):
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    body = {"model": model, "max_tokens": 160, "temperature": 0.2, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    req = urllib.request.Request(LM_URL + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip().strip('"').rstrip(".")


COSTUME = r"costume|mascot|outfit"


def strip_costume(desc):
    """Mode 2: small captioners describe the costume anyway, and costume words in the caption would keep the trigger
    from owning it. Drop 'wearing ...' clauses; in other clauses cut just the 'a man in a ... costume' phrase."""
    kept = []
    for part in desc.split(","):
        if re.search(rf"\b(wearing|dressed)\b", part, re.I) and re.search(COSTUME, part, re.I) or re.search(r"\bwearing\b", part, re.I):
            continue
        # "(a) man in a green grasshopper mascot costume (with ...)", then a bare "a green grasshopper costume"
        part = re.sub(rf"\b((an?|the)\s+)?(man|woman|person|character|figure)\s+(in|wearing)\s+[\w\s-]*?\b({COSTUME})\b(\s+(costume|suit))?(\s+with\s+[\w\s-]+)?",
                      "", part, flags=re.I)
        part = re.sub(rf"\b(an?|the)\s+[\w\s-]*?\b({COSTUME})\b(\s+(costume|suit))?(\s+with\s+[\w\s-]+)?", "", part, flags=re.I)
        if part.strip():
            kept.append(part)
    return re.sub(r"\s+", " ", ",".join(kept)).strip(" ,") or desc


def captions(imgs, prompt, work, logf, builtin=False):
    model = None if builtin else lm_studio_model()
    if model:
        log(f"캡션: LM 스튜디오 {model}", logf)
        caps = {str(p): caption_lm(model, p, prompt) for p in imgs}
        unload_lm_studio()
        return caps
    log("캡션: 내장 Qwen3-VL-2B (처음엔 모델 약 4.5GB를 받아요)", logf)
    jobs, out = work / "caption_jobs.json", work / "caption_out.json"
    jobs.write_text(json.dumps({"prompt": prompt, "images": [str(p) for p in imgs]}, ensure_ascii=False), encoding="utf-8")
    if subprocess.run([sys.executable, str(HERE / "captioner.py"), str(jobs), str(out)]).returncode != 0:
        raise SystemExit(" 캡션 단계에서 멈췄어요. 위 오류를 확인해 주세요.")
    return json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- training
def run(cmd, logf, cwd=None, total=0):
    log("$ " + " ".join(str(c) for c in cmd), logf)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    t0 = s0 = None
    with open(logf, "a", encoding="utf-8") as h:
        for line in p.stdout:
            h.write(line)
            h.flush()
            # the monitor logs every 10 steps; show that on one console line with a rough ETA
            m = re.search(r"\] step (\d+) \[loss=([\d.]+)", line)
            if m and total:
                s = int(m[1])
                if t0 is None:
                    t0, s0 = time.time(), s
                eta = f" · 남은 시간 약 {round((time.time() - t0) / (s - s0) * (total - s) / 60)}분" if s > s0 else ""
                print(f"\r 학습 {s}/{total} 스텝 · loss {m[2]}{eta}      ", end="", flush=True)
    if total:
        print()
    if p.wait() != 0:
        raise SystemExit(f" 실패 (exit {p.returncode}) — {logf} 를 보세요")


def free_comfy(logf):
    for port in COMFY_PORTS:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
            log(f"컴피 {port}: 올라간 모델을 내렸어요 (컴피는 켜둔 채)", logf)
        except Exception:
            pass


def edit_config(path, every, extra=()):
    c = path.read_text(encoding="utf-8")
    c = c.replace('"artist_from_folder": "name",', '"artist_from_folder": None,').replace('"txt_as": "tags",', '"txt_as": "nl",')
    c = re.sub(r'("include_root": False,\r?\n    \},\r?\n\])', r'\1\nCAPTION = {"template": "nl"}', c, count=1)
    for key, val in (("SAMPLE_EVERY", every), ("SAVE_EVERY", every), ("KEEP_LAST", 8), ("SAVE_MINUTES", 10), *extra):
        c = re.sub(rf"^# {key} = .*$", f"{key} = {val}", c, count=1, flags=re.M)
    path.write_text(c, encoding="utf-8")
    assert '"txt_as": "nl"' in c and 'CAPTION = {"template": "nl"}' in c, "config edit failed"


def sheet(refs, samples_dir, out):
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    H = 380

    def tile(p, label):
        im = Image.open(p).convert("RGB")
        im = im.resize((round(im.width * H / im.height), H))
        c = Image.new("RGB", (im.width, H + 40), (13, 15, 19))
        c.paste(im, (0, 40))
        ImageDraw.Draw(c).text((8, 6), label, fill=(196, 245, 66), font=font)
        return c

    rows = [[tile(p, "학습 사진") for p in refs]]
    for d in sorted(samples_dir.glob("step_*")):
        rows.append([tile(p, f"{int(d.name.split('_')[1])}스텝") for p in sorted(d.glob("*.png"))])
    W = max(sum(t.width + 10 for t in r) for r in rows)
    out_im = Image.new("RGB", (W, sum(H + 50 for _ in rows)), (13, 15, 19))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            out_im.paste(t, (x, y))
            x += t.width + 10
        y += H + 50
    out_im.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="?")
    ap.add_argument("--trigger")
    ap.add_argument("--mode", choices=["1", "2", "3"])
    ap.add_argument("--cls", help="짧은 영어 설명, 예: a young Korean man")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--setup-only", action="store_true", help="경로·모델 확인까지만")
    ap.add_argument("--builtin-caption", action="store_true", help="LM 스튜디오가 켜져 있어도 내장 캡션 사용")
    a = ap.parse_args()
    print("\n ==== KohakuMoss 원클릭 LoRA 학습기 ====")
    s = load_settings()
    models, kohaku = Path(s["models"]), Path(s["kohaku"])
    found = download_missing(models, find_models(models))
    weights = weights_dir(models, found)
    print(f" 컴피 models: {models}\n 학습 모델: " + ", ".join(p.name for p in found.values()))
    if a.setup_only:
        print(f" 학습에 쓸 폴더: {weights}")
        return

    while not a.photos or not Path(a.photos).is_dir():
        a.photos = ask("\n 사진 폴더 경로: ")
    while not a.trigger or not re.sub(r"[^a-z0-9]", "", a.trigger.lower()):
        a.trigger = ask(" 트리거 단어 (영문 소문자, 예: bzsuit): ")
    while a.mode not in ("1", "2", "3"):
        print("\n 종류를 고르세요\n   1 = 사람 (옷은 프롬프트로 바뀜, 얼굴만 고정)\n"
              "   2 = 캐릭터/의상 (의상까지 고정)\n   3 = 얼굴 디테일러용 (얼굴만 잘라 둔 사진)")
        a.mode = ask(" 번호: ")
    while not a.cls:
        a.cls = ask("\n 무엇인지 짧게 영어로 (예: a young Korean man): ")
    print("\n 캡션 - 학습 - 컴피 등록까지 " + ("1~2시간 반" if a.mode == "1" else "1~2시간" if a.mode == "3" else "40분~1시간") + "쯤 걸려요. 창을 닫지 마세요.\n")

    trig = re.sub(r"[^a-z0-9]", "", a.trigger.lower())
    work = HERE / "runs" / trig
    data = work / "dataset" / f"1_{trig}"
    data.mkdir(parents=True, exist_ok=True)
    logf = work / "log.txt"
    log(f"시작: {a.photos} · 트리거 {trig} · 모드 {a.mode} · '{a.cls}'", logf)

    # 1. photos
    for p in sorted(Path(a.photos).iterdir()):
        if p.suffix.lower() in IMG_EXT:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((1536, 1536))
            im.save(data / f"{p.stem}.jpg", quality=95)
    imgs = sorted(data.glob("*.jpg"))
    if len(imgs) < 5:
        raise SystemExit(f" 사진이 {len(imgs)}장뿐이에요 (최소 5장, 권장 15~30장)")
    log(f"사진 {len(imgs)}장 준비", logf)

    # 2. captions
    if HANGUL.search(a.cls):
        m = None if a.builtin_caption else lm_studio_model()
        if not m:
            raise SystemExit(" 명령창에서는 설명을 영어로 써 주세요 (한글은 웹 화면에서 자동 번역돼요)")
        a.cls = translate_lm(m, a.cls)
        log(f"설명 번역: {a.cls}", logf)
    caps = captions(imgs, ASK[a.mode].format(cls=a.cls), work, logf, a.builtin_caption)
    lines = []
    for p in imgs:
        desc = caps[str(p)]
        if a.mode == "2":
            # small captioners describe the costume anyway; costume words in the caption would stop the trigger from owning it
            desc = strip_costume(desc)
        cap = f"{trig}, {a.cls}, {desc}"
        p.with_suffix(".txt").write_text(cap, encoding="utf-8")
        lines.append(f"- `{p.name}` {cap}")
    (work / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"캡션 {len(imgs)}장 완료 (예: {lines[0][:120]})", logf)

    # 3. config — ~80 passes per photo; sample/save every fifth
    # mode 2 trains at 768px with rank 8 (same costume quality as 1024/r32 in our test, about half the time, 47 MB files);
    # a 768px step packs ~4.6 images. Modes 1 and 3 use 1024/r32: ~2 images per step (768/r8 lost on face likeness, 2026-11-10).
    if a.mode == "3":
        # face crops: ~22 passes per crop at ~2 images a step (k40face: 46 crops -> 500 steps was the best checkpoint)
        steps = a.steps or max(300, round(max(600, min(1400, len(imgs) * 22)) / 2 / 50) * 50)
        every = max(50, round(steps / 5 / 50) * 50)
        extra = ()
    elif a.mode == "1":
        steps = a.steps or round(max(600, min(1800, len(imgs) * 80)) / 2 / 10) * 10
        every = max(20, round(steps / 5 / 10) * 10)
        extra = ()
    else:
        steps = a.steps or round(max(600, min(1800, len(imgs) * 80)) / 4.6 / 10) * 10
        every = max(20, round(steps / 5 / 10) * 10)
        extra = (("TARGET_AREA", 589824), ("DIM", 8), ("ALPHA", 8.0),
                 ("SCHEDULE", f'{{"mode": "cosine", "warmup": {round(steps * 0.18)}, "end": -1, "min_value": 0.1}}'))
    name = f"{trig}-{steps}"
    prompt = (f"{trig}, close-up of the face, soft window light, photo" if a.mode == "3"
              else f"{trig}, {a.cls}, standing in a sunny park, smiling, photo")
    bindir = Path(sys.prefix) / "Scripts"
    run([bindir / "kohakumoss-config.exe", "new", "train", "--weights", weights, "--data", data.parent, "--name", name,
         "--steps", steps, "--vram", "auto", "--precision", "bf16", "--prompt", prompt, "--force"], logf, cwd=kohaku)
    edit_config(kohaku / "configs" / "user" / f"{name}.py", every, extra)
    log(f"설정: {steps}스텝, {every}스텝마다 샘플·저장" + (", 768px·rank 8" if extra else ", 1024px·rank 32"), logf)

    # 4. train
    free_comfy(logf)
    kog = bindir / "kogine.exe"
    for script, cfg in (("scripts/data/cache_text.py", f"{name}_data"), ("scripts/data/cache_latents.py", f"{name}_data"), ("scripts/train/moss.py", name)):
        run([kog, "run", script, "--config", f"configs/user/{cfg}.py"], logf, cwd=kohaku, total=steps if "train" in script else 0)
    log("학습 끝", logf)

    # 5. register in ComfyUI + compare sheet
    rundir = kohaku / "runs" / "moss" / name
    out_dir = models / "loras" / "MOSS"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rundir / "export" / f"{name}-comfy.safetensors", out_dir / f"{trig}.safetensors")
    for ck in sorted((rundir / "checkpoints").glob("step_*")):
        f = ck / "export" / f"{name}-comfy.safetensors"
        # only the regular saves; the 10-minute autosaves would just clutter the LoRA list
        if f.exists() and int(ck.name.split("_")[1]) % every == 0:
            shutil.copy2(f, out_dir / f"{trig}-{int(ck.name.split('_')[1])}.safetensors")
    sheet(imgs[:3], rundir / "samples", work / "compare.jpg")
    log(f"완료: 컴피 LoRA 목록 MOSS\\{trig}.safetensors (+스텝별) · 비교 이미지 {work / 'compare.jpg'}", logf)


if __name__ == "__main__":
    main()
