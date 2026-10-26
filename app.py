"""KohakuMoss OneClick — the browser UI. Runs with the KohakuMoss venv python, standard library only (plus what the venv has).

  python app.py            serves http://127.0.0.1:7865 and opens it

One job at a time: photos -> captions (editable) -> training -> LoRA in ComfyUI's list. The pipeline pieces live in
oneclick.py (which also still works as a console tool).
"""
import json, os, re, shutil, subprocess, sys, threading, time, urllib.parse, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

import oneclick as oc

PORT = 7865
HERE = oc.HERE
RUNS = HERE / "runs"
LOCK = threading.Lock()
STATE = {"stage": "idle", "msg": "", "error": "", "cap_done": 0, "cap_total": 0, "step": 0, "total": 0, "loss": None,
         "eta_min": None, "phase": "", "name": "", "lora": "", "download": ""}
PROJECT = {"trigger": "", "mode": "2", "cls": ""}
PROC = {"p": None, "stop": False}
MODE_SEC = {"768": 7.1, "1024": 6.7}  # seconds per step on an RTX 3090 (10-26 measurements)


def settings():
    return json.loads(oc.SETTINGS.read_text(encoding="utf-8")) if oc.SETTINGS.exists() else {}


def kohaku():
    return Path(sys.prefix).parent


def trig_clean(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def data_dir(trig=None):
    t = trig or PROJECT["trigger"]
    return RUNS / t / "dataset" / f"1_{t}"


def photos():
    d = data_dir()
    if not PROJECT["trigger"] or not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.jpg")):
        cap = p.with_suffix(".txt")
        out.append({"name": p.name, "url": f"/run/{PROJECT['trigger']}/dataset/1_{PROJECT['trigger']}/{urllib.parse.quote(p.name)}",
                    "caption": cap.read_text(encoding="utf-8") if cap.exists() else ""})
    return out


STRENGTH = {"light": 0.7, "normal": 1.0, "strong": 1.4}  # "학습 세기" on the page: a factor on the training amount


def plan(n, mode, strength="normal"):
    """Steps, sample interval, config extras and a time estimate — oneclick.py's rules times the strength factor."""
    f = STRENGTH.get(strength, 1.0)
    if mode == "3":
        # face crops: ~22 passes per crop at ~2 images a step (k40face: 46 crops -> 500 steps was the best checkpoint)
        steps = max(150, round(max(600, min(1400, n * 22)) * f / 2 / 50) * 50)
        every = max(50, round(steps / 5 / 50) * 50)
        return steps, every, (), round(steps * MODE_SEC["1024"] / 60 + 5 * 2 + 6), "1024px · rank 32"
    steps = max(100, round(max(600, min(1800, n * 80)) * f / 4.6 / 10) * 10)
    every = max(20, round(steps / 5 / 10) * 10)
    extra = (("TARGET_AREA", 589824), ("DIM", 8), ("ALPHA", 8.0),
             ("SCHEDULE", f'{{"mode": "cosine", "warmup": {round(steps * 0.18)}, "end": -1, "min_value": 0.1}}'))
    return steps, every, extra, round(steps * MODE_SEC["768"] / 60 + 5 * 2 + 6), "768px · rank 8"


def samples():
    if not STATE["name"]:
        return []
    root = kohaku() / "runs" / "moss" / STATE["name"] / "samples"
    out = []
    for d in sorted(root.glob("step_*")):
        for p in sorted(d.glob("*.png")):
            out.append({"step": int(d.name.split("_")[1]), "url": f"/sample/{STATE['name']}/{d.name}/{urllib.parse.quote(p.name)}"})
    return out


def save_project():
    s = settings()
    s["last_project"] = PROJECT  # so reopening the trainer picks up where it was left
    oc.SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def set_state(**kw):
    with LOCK:
        STATE.update(kw)


# ---------------------------------------------------------------- jobs
def stream(cmd, logf, cwd=None, on_line=None):
    with open(logf, "a", encoding="utf-8") as h:
        h.write("$ " + " ".join(str(c) for c in cmd) + "\n")
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                         creationflags=subprocess.CREATE_NO_WINDOW)
    PROC["p"] = p
    with open(logf, "a", encoding="utf-8") as h:
        for line in p.stdout:
            h.write(line)
            h.flush()
            if on_line:
                on_line(line)
    code = p.wait()
    PROC["p"] = None
    if PROC["stop"]:
        raise RuntimeError("멈춤")
    if code != 0:
        raise RuntimeError(f"{Path(str(cmd[2] if len(cmd) > 2 else cmd[0])).name} 실패 (exit {code}) — {logf} 를 보세요")


def job(fn):
    def wrap(*a):
        PROC["stop"] = False
        try:
            fn(*a)
        except Exception as e:  # shown on the page
            set_state(stage="stopped" if PROC["stop"] else "error", error=str(e), msg="")
    threading.Thread(target=wrap, args=(), daemon=True).start()


def caption_job(builtin):
    trig, mode, cls = PROJECT["trigger"], PROJECT["mode"], PROJECT["cls"]
    work, imgs = RUNS / trig, sorted(data_dir().glob("*.jpg"))
    logf = work / "log.txt"
    korean = bool(oc.HANGUL.search(cls))  # a Korean description is translated by the caption model first
    set_state(stage="captioning", cap_done=0, cap_total=len(imgs), error="", msg="캡션 모델 준비 중")
    model = None if builtin else oc.lm_studio_model()
    caps = {}
    if model:
        set_state(msg=f"LM 스튜디오 {model}")
        if korean:
            cls = oc.translate_lm(model, cls)
        prompt = oc.ASK[mode].format(cls=cls)
        for i, p in enumerate(imgs, 1):
            caps[str(p)] = oc.caption_lm(model, p, prompt)
            set_state(cap_done=i)
        oc.unload_lm_studio()
    else:
        set_state(msg="내장 Qwen3-VL-2B (처음 한 번 약 4GB 받는 중일 수 있어요)")
        jobs, out = work / "caption_jobs.json", work / "caption_out.json"
        job_spec = {"prompt": oc.ASK[mode] if korean else oc.ASK[mode].format(cls=cls), "images": [str(p) for p in imgs]}
        if korean:
            job_spec["translate"] = oc.TRANSLATE.format(text=cls)
        jobs.write_text(json.dumps(job_spec, ensure_ascii=False), encoding="utf-8")

        def on_line(line):
            m = re.search(r"캡션 (\d+)/(\d+)", line)
            if m:
                set_state(cap_done=int(m[1]), msg="내장 Qwen3-VL-2B")
        stream([sys.executable, HERE / "captioner.py", jobs, out], logf, on_line=on_line)
        caps = json.loads(out.read_text(encoding="utf-8"))
        cls = caps.pop("__translate__", cls)
    PROJECT["cls_en"] = cls
    save_project()
    for p in imgs:
        desc = caps[str(p)]
        if mode == "2":
            desc = oc.strip_costume(desc)
        p.with_suffix(".txt").write_text(f"{trig}, {cls}, {desc}", encoding="utf-8")
    set_state(stage="captioned", msg="")


def train_job(strength):
    trig, mode, cls = PROJECT["trigger"], PROJECT["mode"], PROJECT.get("cls_en") or PROJECT["cls"]
    s = settings()
    models, kdir = Path(s["models"]), kohaku()
    work, imgs = RUNS / trig, sorted(data_dir().glob("*.jpg"))
    logf = work / "log.txt"
    steps, every, extra, _, _ = plan(len(imgs), mode, strength)
    name = f"{trig}-{steps}"
    set_state(stage="training", phase="설정 만드는 중", step=0, total=steps, loss=None, eta_min=None, name=name, error="", lora="")
    weights = oc.weights_dir(models, oc.find_models(models))
    prompt = (f"{trig}, close-up of the face, soft window light, photo" if mode == "3"
              else f"{trig}, {cls}, standing in a sunny park, smiling, photo")
    bindir = Path(sys.prefix) / "Scripts"
    stream([bindir / "kohakumoss-config.exe", "new", "train", "--weights", weights, "--data", data_dir().parent, "--name", name,
            "--steps", steps, "--vram", "auto", "--precision", "bf16", "--prompt", prompt, "--force"], logf, cwd=kdir)
    oc.edit_config(kdir / "configs" / "user" / f"{name}.py", every, extra)
    oc.free_comfy(logf)
    t0 = {"t": None, "s": 0}

    def on_line(line):
        m = re.search(r"\] step (\d+) \[loss=([\d.]+)", line)
        if m:
            st = int(m[1])
            if t0["t"] is None:
                t0.update(t=time.time(), s=st)
            eta = round((time.time() - t0["t"]) / (st - t0["s"]) * (steps - st) / 60) if st > t0["s"] else None
            set_state(step=st, loss=float(m[2]), eta_min=eta, phase="학습 중")
        elif "steplog" in line or "batch 1/1" in line:
            set_state(phase="중간 샘플 그리는 중")
        elif "prewarm" in line:
            set_state(phase="첫 실행 준비 (몇 분)")
    kog = bindir / "kogine.exe"
    set_state(phase="캡션 읽는 중 (텍스트 캐시)")
    stream([kog, "run", "scripts/data/cache_text.py", "--config", f"configs/user/{name}_data.py"], logf, cwd=kdir)
    set_state(phase="사진 읽는 중 (이미지 캐시)")
    stream([kog, "run", "scripts/data/cache_latents.py", "--config", f"configs/user/{name}_data.py"], logf, cwd=kdir)
    set_state(phase="학습 준비")
    stream([kog, "run", "scripts/train/moss.py", "--config", f"configs/user/{name}.py"], logf, cwd=kdir, on_line=on_line)
    set_state(phase="컴피에 등록 중")
    rundir = kdir / "runs" / "moss" / name
    out_dir = models / "loras" / "MOSS"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rundir / "export" / f"{name}-comfy.safetensors", out_dir / f"{trig}.safetensors")
    for ck in sorted((rundir / "checkpoints").glob("step_*")):
        f = ck / "export" / f"{name}-comfy.safetensors"
        if f.exists() and int(ck.name.split("_")[1]) % every == 0:
            shutil.copy2(f, out_dir / f"{trig}-{int(ck.name.split('_')[1])}.safetensors")
    oc.sheet(imgs[:3], rundir / "samples", work / "compare.jpg")
    set_state(stage="done", phase="", step=steps, eta_min=0, lora=f"MOSS\\{trig}.safetensors")


def download_job():
    from huggingface_hub import hf_hub_download
    models = Path(settings()["models"])
    for sub, p in oc.find_models(models).items():
        if p is None:
            name, size = oc.DOWNLOAD[sub]
            set_state(download=f"{name} ({size}) 받는 중")
            hf_hub_download(oc.HF_REPO, f"{sub}/{name}", local_dir=models)
    set_state(download="")


def stop():
    PROC["stop"] = True
    p = PROC["p"]
    if p:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)


# ---------------------------------------------------------------- http
def api_state():
    s = settings()
    st = {"settings": s, "project": PROJECT, "photos": photos(), "lmstudio": oc.lm_studio_model()}
    if s.get("models") and Path(s["models"]).is_dir():
        found = oc.find_models(Path(s["models"]))
        st["models"] = {k: (v.name if v else None) for k, v in found.items()}
        st["missing"] = [f"{k}\\{oc.DOWNLOAD[k][0]} ({oc.DOWNLOAD[k][1]})" for k, v in found.items() if v is None]
    if st["photos"]:
        st["plan"] = {}
        for k in STRENGTH:
            steps, _, _, minutes, res = plan(len(st["photos"]), PROJECT["mode"], k)
            st["plan"][k] = {"steps": steps, "minutes": minutes, "res": res}
    with LOCK:
        st["job"] = dict(STATE)
    st["job"]["samples"] = samples()
    if (RUNS / PROJECT["trigger"] / "compare.jpg").exists() and STATE["stage"] == "done":
        st["job"]["compare"] = f"/run/{PROJECT['trigger']}/compare.jpg?{int(time.time())}"
    return st


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def file(self, path):
        path = path.resolve()
        if not path.is_file():
            return self.send(404, {"error": "없음"})
        ctype = {".jpg": "image/jpeg", ".png": "image/png", ".html": "text/html; charset=utf-8"}.get(path.suffix.lower(), "application/octet-stream")
        self.send(200, path.read_bytes(), ctype)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(x) for x in u.path.split("/") if x]
        if not parts:
            return self.file(HERE / "ui.html")
        if parts == ["api", "state"]:
            return self.send(200, api_state())
        if parts[0] == "run" and ".." not in parts:
            return self.file(RUNS.joinpath(*parts[1:]))
        if parts[0] == "sample" and ".." not in parts:
            return self.file(kohaku() / "runs" / "moss" / parts[1] / "samples" / parts[2] / parts[3])
        self.send(404, {"error": "없음"})

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        busy = STATE["stage"] in ("captioning", "training")
        try:
            if u.path == "/api/settings":
                d = json.loads(self.body())
                p = Path(d.get("models", "").strip().strip('"'))
                if not (p / "loras").is_dir():
                    return self.send(400, {"error": "그 폴더 안에 loras 폴더가 없어요. 컴피의 models 폴더를 넣어 주세요."})
                s = settings()
                s["models"] = str(p)
                oc.SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
                return self.send(200, {"ok": True})
            if u.path == "/api/download":
                job(download_job)
                return self.send(200, {"ok": True})
            if u.path == "/api/project":
                d = json.loads(self.body())
                if busy:
                    return self.send(409, {"error": "작업 중에는 바꿀 수 없어요"})
                t = trig_clean(d.get("trigger"))
                if t and t != PROJECT["trigger"]:
                    PROJECT["trigger"] = t
                    set_state(stage="idle", error="", name="", lora="")
                for k in ("mode", "cls"):
                    if k in d:
                        PROJECT[k] = str(d[k]).strip()
                if "cls" in d:
                    PROJECT.pop("cls_en", None)  # a new description needs a new translation
                if PROJECT["trigger"]:
                    data_dir().mkdir(parents=True, exist_ok=True)
                save_project()
                return self.send(200, {"ok": True, "trigger": PROJECT["trigger"]})
            if u.path == "/api/photo":
                if not PROJECT["trigger"] or busy:
                    return self.send(409, {"error": "트리거를 먼저 정해 주세요"})
                name = Path(q.get("name", "photo")).stem
                name = re.sub(r'[\\/:*?"<>|]', "_", name)[:80] or "photo"
                im = ImageOps.exif_transpose(Image.open(BytesIO(self.body()))).convert("RGB")
                im.thumbnail((1536, 1536))
                im.save(data_dir() / f"{name}.jpg", quality=95)
                return self.send(200, {"ok": True})
            if u.path == "/api/photo_delete":
                d = json.loads(self.body())
                for ext in (".jpg", ".txt"):
                    (data_dir() / (Path(d["name"]).stem + ext)).unlink(missing_ok=True)
                return self.send(200, {"ok": True})
            if u.path == "/api/caption":
                d = json.loads(self.body())
                if busy or len(photos()) < 5 or not PROJECT["cls"]:
                    return self.send(409, {"error": "사진 5장 이상과 설명이 필요해요"})
                job(lambda: caption_job(bool(d.get("builtin"))))
                return self.send(200, {"ok": True})
            if u.path == "/api/caption_edit":
                d = json.loads(self.body())
                (data_dir() / (Path(d["name"]).stem + ".txt")).write_text(d["text"].strip(), encoding="utf-8")
                return self.send(200, {"ok": True})
            if u.path == "/api/train":
                d = json.loads(self.body())
                if busy:
                    return self.send(409, {"error": "이미 작업 중이에요"})
                if any(not p["caption"] for p in photos()):
                    return self.send(409, {"error": "캡션이 없는 사진이 있어요. 캡션부터 달아 주세요"})
                job(lambda: train_job(d.get("strength", "normal")))
                return self.send(200, {"ok": True})
            if u.path == "/api/stop":
                stop()
                return self.send(200, {"ok": True})
            if u.path == "/api/open":
                d = json.loads(self.body())
                target = Path(settings()["models"]) / "loras" / "MOSS" if d.get("what") == "loras" else RUNS / PROJECT["trigger"]
                os.startfile(target)
                return self.send(200, {"ok": True})
        except Exception as e:
            return self.send(500, {"error": str(e)})
        self.send(404, {"error": "없음"})


def main():
    RUNS.mkdir(exist_ok=True)
    PROJECT.update(settings().get("last_project", {}))
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n KohakuMoss 원클릭 학습기: {url}\n 이 창을 닫으면 학습기가 꺼져요.\n")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
