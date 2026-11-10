"""Finds the main face in each photo for the face mode's auto-crop.

  python facecrop.py <jobs.json> <out.json>

jobs.json: {"images": ["a.jpg", ...]}  ->  out.json: {"a.jpg": [x1, y1, x2, y2] in original pixels, or null when no face}
MTCNN (Zhang et al. 2016) with the weights in facecrop_models/, so nothing extra is installed or downloaded. Runs on the
CPU and leaves the GPU alone. The weights and the model layout come from timesler/facenet-pytorch (MIT).
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn.functional import interpolate, pad, softmax

MODELS = Path(__file__).parent / "facecrop_models"
LONG_SIDE = 1024  # plenty to locate a face, and it keeps a photo to about a second on the CPU
MARGIN = 0.25  # grey border, so a selfie whose face runs off the frame still looks like a whole face to the detector
MIN_FACE = 40  # px at LONG_SIDE
THRESHOLDS = 0.6, 0.7, 0.7


class Net(nn.Module):
    """P-, R- and O-Net differ only in these numbers. Attribute names match the weight files."""

    def __init__(self, file, convs, dense, heads):
        super().__init__()
        self.count, self.dense_at, self.heads = len(convs), len(convs) + 1 if dense else 0, heads
        for i, (a, b, kernel, stride_pool) in enumerate(convs, 1):
            setattr(self, f"conv{i}", nn.Conv2d(a, b, kernel))
            setattr(self, f"prelu{i}", nn.PReLU(b))
            setattr(self, f"pool{i}", nn.MaxPool2d(stride_pool, 2, ceil_mode=True) if stride_pool else nn.Identity())
        last = convs[-1][1]
        if dense:
            setattr(self, f"dense{self.dense_at}", nn.Linear(*dense))
            setattr(self, f"prelu{self.dense_at}", nn.PReLU(dense[1]))
            last = dense[1]
        layer = (lambda n: nn.Linear(last, n)) if dense else (lambda n: nn.Conv2d(last, n, 1))
        for name, n in heads.items():
            setattr(self, name, layer(n))
        self.load_state_dict(torch.load(MODELS / file, map_location="cpu", weights_only=True))
        self.eval()

    def forward(self, x):
        for i in range(1, self.count + 1):
            x = getattr(self, f"pool{i}")(getattr(self, f"prelu{i}")(getattr(self, f"conv{i}")(x)))
        if self.dense_at:
            x = x.permute(0, 3, 2, 1).contiguous().view(x.shape[0], -1)  # the weights expect this flatten order
            x = getattr(self, f"prelu{self.dense_at}")(getattr(self, f"dense{self.dense_at}")(x))
        return [getattr(self, name)(x) for name in self.heads]


def load_nets():
    pnet = Net("pnet.pt", [(3, 10, 3, 2), (10, 16, 3, 0), (16, 32, 3, 0)], None, {"conv4_1": 2, "conv4_2": 4})
    rnet = Net("rnet.pt", [(3, 28, 3, 3), (28, 48, 3, 3), (48, 64, 2, 0)], (576, 128), {"dense5_1": 2, "dense5_2": 4})
    onet = Net("onet.pt", [(3, 32, 3, 3), (32, 64, 3, 3), (64, 64, 3, 2), (64, 128, 2, 0)], (1152, 256),
               {"dense6_1": 2, "dense6_2": 4, "dense6_3": 10})
    return pnet, rnet, onet


def nms(b, scores, limit):
    """Indices of the boxes to keep, best score first, dropping any that overlap a kept one by more than limit."""
    order, keep = scores.argsort(descending=True), []
    area = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    while len(order):
        i, order = order[0], order[1:]
        keep.append(i)
        inter = (torch.min(b[i, 2], b[order, 2]) - torch.max(b[i, 0], b[order, 0])).clamp(0) *                 (torch.min(b[i, 3], b[order, 3]) - torch.max(b[i, 1], b[order, 1])).clamp(0)
        order = order[inter / (area[i] + area[order] - inter).clamp(1e-6) <= limit]
    return torch.stack(keep)


def norm(x):
    return (x - 127.5) * 0.0078125


def regress(b, r):
    w, h = b[:, 2] - b[:, 0] + 1, b[:, 3] - b[:, 1] + 1
    b = b.clone()
    b[:, 0] += r[:, 0] * w
    b[:, 1] += r[:, 1] * h
    b[:, 2] += r[:, 2] * w
    b[:, 3] += r[:, 3] * h
    return b


def squared(b):
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    side = torch.max(w, h)
    b = b.clone()
    b[:, 0] += w * 0.5 - side * 0.5
    b[:, 1] += h * 0.5 - side * 0.5
    b[:, 2] = b[:, 0] + side
    b[:, 3] = b[:, 1] + side
    return b


def patches(x, b, size):
    h, w = x.shape[2:]
    out = []
    for x1, y1, x2, y2 in b[:, :4].int().tolist():
        x1, y1, x2, y2 = max(x1, 1) - 1, max(y1, 1) - 1, min(x2, w), min(y2, h)
        out.append(interpolate(x[:, :, y1:y2, x1:x2], (size, size), mode="area"))
    return norm(torch.cat(out))


def propose(x, pnet):
    """First stage: slide the small net over a pyramid of sizes. Rows are x1, y1, x2, y2, score, then 4 offsets."""
    h, w = x.shape[2:]
    scale, found = 12.0 / MIN_FACE, []
    while min(h, w) * scale >= 12:
        cls, reg = pnet(norm(interpolate(x, (int(h * scale + 1), int(w * scale + 1)), mode="area")))
        score = softmax(cls, 1)[0, 1]
        ys, xs = (score >= THRESHOLDS[0]).nonzero(as_tuple=True)
        if len(ys):
            q1 = ((2 * torch.stack([xs, ys], 1).float() + 1) / scale).floor()
            q2 = ((2 * torch.stack([xs, ys], 1).float() + 12) / scale).floor()
            rows = torch.cat([q1, q2, score[ys, xs].unsqueeze(1), reg[0][:, ys, xs].T], 1)
            found.append(rows[nms(rows[:, :4], rows[:, 4], 0.5)])
        scale *= 0.709
    if not found:
        return None
    rows = torch.cat(found)
    rows = rows[nms(rows[:, :4], rows[:, 4], 0.7)]
    return squared(regress(rows[:, :5], rows[:, 5:9]))


def detect(x, nets):
    pnet, rnet, onet = nets
    b = propose(x, pnet)
    if b is None or not len(b):
        return None
    for net, size, threshold in ((rnet, 24, THRESHOLDS[1]), (onet, 48, THRESHOLDS[2])):
        cls, reg = run(net, patches(x, b, size))
        score = softmax(cls, 1)[:, 1]
        keep = score > threshold
        if not keep.any():
            return None
        b = torch.cat([b[keep, :4], score[keep].unsqueeze(1)], 1)
        reg = reg[keep]
        if size == 24:
            pick = nms(b[:, :4], b[:, 4], 0.7)
            b = squared(regress(b[pick], reg[pick]))
        else:
            b = regress(b, reg)
    return b


def run(net, x):
    """Returns (class logits, box offsets), a chunk at a time so a busy photo can't eat all the memory."""
    parts = [net(c) for c in x.split(256)]
    return tuple(torch.cat([p[i] for p in parts]) for i in (0, 1))


def main_face(path, nets):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(1.0, LONG_SIDE / max(im.size))
    small = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    x = torch.from_numpy(np.array(small)).float().permute(2, 0, 1)[None]
    m = int(max(small.size) * MARGIN)
    x = pad(x, (m, m, m, m), value=127.5)
    b = detect(x, nets)
    if b is None:
        return None
    biggest = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()
    return [(float(v) - m) / s for v in b[biggest, :4]]


def main():
    torch.set_grad_enabled(False)
    images = json.load(open(sys.argv[1], encoding="utf-8"))["images"]
    nets = load_nets()
    out = {}
    for i, path in enumerate(images, 1):
        out[path] = main_face(path, nets)
        print(f"  얼굴 {i}/{len(images)}", flush=True)
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
