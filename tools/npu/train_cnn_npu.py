#!/usr/bin/env python3
"""
Train, distill, and INT8-quantize a real 5-Stage 64x64 Depthwise-Separable CNN
for the Gowin Tang Mega 138K Pro FPGA NPU (npu_core v2.0.0, 0xF8000900).

Classes:
  0 = CAT
  1 = DOG
  2 = UNKNOWN

Training data:
  - Oxford-IIIT Pet (real high-res cats & dogs)
  - CIFAR-10 (cats, dogs, plus 8 non-pet object classes as UNKNOWN)
  - Synthetic cartoon/vector cats & dogs + synthetic calibration/OOD patterns
  - Distilled from ImageNet-1K pretrained MobileNetV3-Small teacher
  - Augmented with RGB332 quantization + meme captions + aspect stretch

Exports:
  - firmware/src/app/npu_weights.h (bit-exact packed uint32_t weights & int32_t biases)
  - tools/npu/cnn_model.npz (checkpoint + exact integer emulator test)
"""

import io
import math
import pathlib
import random
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DS_DIR = pathlib.Path(r"C:\Users\penis\.gemini\antigravity\scratch\npu_datasets")
WEIGHTS_H = ROOT / "firmware" / "src" / "app" / "npu_weights.h"
CKPT_NPZ = ROOT / "tools" / "npu" / "cnn_model.npz"

torch.set_num_threads(8)


def rgb_to_332_arr(rgb_u8: np.ndarray, dither: bool = False) -> np.ndarray:
    """Exact match to TangMega138K-LCD-Studio RGB332 conversion (optional Floyd-Steinberg)."""
    if not dither:
        r = rgb_u8[..., 0] & 0xE0
        g = (rgb_u8[..., 1] >> 3) & 0x1C
        b = rgb_u8[..., 2] >> 6
        return (r | g | b).astype(np.uint8)

    h, w, _ = rgb_u8.shape
    fbuf = rgb_u8.astype(np.float32).copy()
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            old = np.clip(fbuf[y, x], 0.0, 255.0)
            c = (int(old[0]) & 0xE0) | ((int(old[1]) >> 3) & 0x1C) | (int(old[2]) >> 6)
            out[y, x] = c
            r3 = (c >> 5) & 7
            g3 = (c >> 2) & 7
            b2 = c & 3
            new_rgb = np.array([
                (r3 << 5) | (r3 << 2) | (r3 >> 1),
                (g3 << 5) | (g3 << 2) | (g3 >> 1),
                (b2 << 6) | (b2 << 4) | (b2 << 2) | b2,
            ], dtype=np.float32)
            err = old - new_rgb
            if x + 1 < w:
                fbuf[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    fbuf[y + 1, x - 1] += err * (3.0 / 16.0)
                fbuf[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    fbuf[y + 1, x + 1] += err * (1.0 / 16.0)
    return out


def c_exact_box_downsample_rgb332_to_64x64(fb_200x120: np.ndarray) -> np.ndarray:
    """Bit-exact match to `npu_downsample_rgb332_to_64x64` in `lcd_hud.c`."""
    out = np.zeros((64, 64, 3), dtype=np.int8)
    r3 = (fb_200x120 >> 5) & 7
    g3 = (fb_200x120 >> 2) & 7
    b2 = fb_200x120 & 3
    r8 = ((r3 << 5) | (r3 << 2) | (r3 >> 1)).astype(np.int32)
    g8 = ((g3 << 5) | (g3 << 2) | (g3 >> 1)).astype(np.int32)
    b8 = ((b2 << 6) | (b2 << 4) | (b2 << 2) | b2).astype(np.int32)

    for oy in range(64):
        y0 = (oy * 120) // 64
        y1 = max(y0 + 1, ((oy + 1) * 120) // 64)
        for ox in range(64):
            x0 = (ox * 200) // 64
            x1 = max(x0 + 1, ((ox + 1) * 200) // 64)
            cnt = (y1 - y0) * (x1 - x0)
            ir = int(r8[y0:y1, x0:x1].sum()) // cnt - 128
            ig = int(g8[y0:y1, x0:x1].sum()) // cnt - 128
            ib = int(b8[y0:y1, x0:x1].sum()) // cnt - 128
            out[oy, ox, 0] = max(-127, min(127, ir))
            out[oy, ox, 1] = max(-127, min(127, ig))
            out[oy, ox, 2] = max(-127, min(127, ib))
    return out


def make_synthetic_cartoon_pet(cls_id: int, rng: random.Random) -> Image.Image:
    """Generates diverse vector/cartoon cat or dog illustrations on varied backgrounds."""
    w, h = 128, 128
    bg_choices = [(255, 255, 255), (240, 244, 250), (35, 45, 65), (255, 240, 210), (210, 235, 210)]
    bg = bg_choices[rng.randrange(len(bg_choices))]
    im = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(im)

    cx = 64 + rng.randint(-12, 12)
    cy = 62 + rng.randint(-10, 10)
    sc = rng.uniform(0.75, 1.15)
    fur = rng.choice([(245, 175, 85), (120, 125, 135), (245, 240, 230), (190, 130, 70), (70, 65, 65)])
    outline = (60, 25, 15) if rng.random() < 0.7 else (20, 20, 20)

    # Body
    bw, bh = int(26 * sc), int(30 * sc)
    draw.ellipse([cx - bw, cy + int(10 * sc), cx + bw, cy + int(10 * sc) + bh], fill=fur, outline=outline, width=3)

    if cls_id == 0:  # Cartoon Cat: pointed ears on top, horizontal whiskers
        r_head = int(28 * sc)
        # Ears
        for s in (-1, 1):
            ex = cx + s * int(18 * sc)
            ey = cy - int(22 * sc)
            pts = [(ex - s * int(10 * sc), ey + int(8 * sc)), (ex + s * int(6 * sc), ey - int(20 * sc)), (ex + s * int(12 * sc), ey + int(12 * sc))]
            draw.polygon(pts, fill=fur, outline=outline)
            inner = [(ex - s * int(5 * sc), ey + int(5 * sc)), (ex + s * int(5 * sc), ey - int(12 * sc)), (ex + s * int(8 * sc), ey + int(8 * sc))]
            draw.polygon(inner, fill=(250, 155, 130))
        draw.ellipse([cx - r_head, cy - int(r_head * 0.85), cx + r_head, cy + int(r_head * 0.85)], fill=fur, outline=outline, width=3)
        # Muzzle
        draw.ellipse([cx - int(14 * sc), cy + int(2 * sc), cx + int(14 * sc), cy + int(18 * sc)], fill=(252, 246, 225))
        # Eyes
        for s in (-1, 1):
            ex = cx + s * int(11 * sc)
            ey = cy - int(2 * sc)
            er = int(4 * sc)
            draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=outline)
        # Whiskers
        for s in (-1, 1):
            for wy in (-2, 3, 8):
                draw.line([cx + s * int(18 * sc), cy + int(wy * sc), cx + s * int(36 * sc), cy + int((wy - 2) * sc)], fill=outline, width=2)
    else:  # Cartoon Dog: floppy side ears + big round nose + snout
        r_head = int(26 * sc)
        for s in (-1, 1):
            ex = cx + s * int(24 * sc)
            draw.ellipse([ex - int(10 * sc), cy - int(14 * sc), ex + int(10 * sc), cy + int(22 * sc)], fill=(120, 70, 35), outline=outline, width=3)
        draw.ellipse([cx - r_head, cy - int(r_head * 0.85), cx + r_head, cy + int(r_head * 0.9)], fill=fur, outline=outline, width=3)
        draw.ellipse([cx - int(14 * sc), cy + int(3 * sc), cx + int(14 * sc), cy + int(22 * sc)], fill=(245, 220, 185), outline=outline, width=2)
        for s in (-1, 1):
            ex = cx + s * int(10 * sc)
            ey = cy - int(5 * sc)
            er = int(4 * sc)
            draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=outline)
        # Big black nose
        nr = int(6 * sc)
        draw.ellipse([cx - nr, cy + int(2 * sc), cx + nr, cy + int(12 * sc)], fill=(20, 20, 25))
    return im


def make_synthetic_unknown(rng: random.Random) -> Image.Image:
    """Generates calibration dot targets, Siemens stars, checkerboards, rainbows, noise, text pages."""
    w, h = 128, 128
    kind = rng.randrange(6)
    if kind == 0:
        # Calibration dot grid + Siemens star (like image (2).png / ChatGPT calibration target)
        im = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.rectangle([4, 4, w - 5, h - 5], outline=(0, 0, 0), width=2)
        step = rng.choice([12, 14, 16])
        for gy in range(12, h - 10, step):
            for gx in range(12, w - 10, step):
                r = rng.randint(2, 4)
                if rng.random() < 0.6:
                    draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(0, 0, 0))
                else:
                    draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(0, 0, 0), width=2)
        # Siemens star
        sx, sy = rng.randint(32, 90), rng.randint(32, 90)
        sr = rng.randint(14, 22)
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        for a in range(0, 360, 15):
            rad = math.radians(a)
            draw.line([sx, sy, int(sx + sr * math.cos(rad)), int(sy + sr * math.sin(rad))], fill=(0, 0, 0), width=1)
        return im
    elif kind == 1:
        # Random noise
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    elif kind == 2:
        # Rainbow / color gradient
        x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = x
        arr[..., 1] = y
        arr[..., 2] = 255 - x
        if rng.random() < 0.5:
            arr = np.transpose(arr, (1, 0, 2))
        return Image.fromarray(arr)
    elif kind == 3:
        # Checkerboard / stripes
        sz = rng.choice([8, 16, 24])
        c1 = tuple(rng.randint(0, 255) for _ in range(3))
        c2 = tuple(rng.randint(0, 255) for _ in range(3))
        im = Image.new("RGB", (w, h), c1)
        draw = ImageDraw.Draw(im)
        for yy in range(0, h, sz):
            for xx in range(0, w, sz):
                if ((xx // sz) + (yy // sz)) & 1:
                    draw.rectangle([xx, yy, xx + sz - 1, yy + sz - 1], fill=c2)
        return im
    elif kind == 4:
        # Concentric rings / bullseye
        im = Image.new("RGB", (w, h), (20, 20, 30))
        draw = ImageDraw.Draw(im)
        cx, cy = rng.randint(40, 88), rng.randint(40, 88)
        for r in range(60, 4, -8):
            col = tuple(rng.randint(30, 255) for _ in range(3))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=4)
        return im
    else:
        # Random geometric polygons / UI boxes / charts
        bg = tuple(rng.randint(0, 255) for _ in range(3))
        im = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(im)
        for _ in range(rng.randint(4, 10)):
            x0, y0 = rng.randint(0, 100), rng.randint(0, 100)
            x1, y1 = min(w - 1, x0 + rng.randint(10, 50)), min(h - 1, y0 + rng.randint(10, 50))
            col = tuple(rng.randint(0, 255) for _ in range(3))
            draw.rectangle([x0, y0, x1, y1], fill=col)
        return im


class TinyMobileNet64(nn.Module):
    """
    5-Stage 64x64 Depthwise-Separable CNN mapped 1:1 to Gowin FPGA npu_core v2.
    Input: (B, 3, 64, 64) in [-127, 127] / 128.0
    """

    def __init__(self):
        super().__init__()
        # Stem: Conv 3x3 s2 (3 -> 24) -> 32x32x24
        self.stem_conv = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_bn = nn.BatchNorm2d(24)

        # Stage 2: DW 3x3 s2 (24) + PW 1x1 (24 -> 48) -> 16x16x48
        self.dw2 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24, bias=True)
        self.dw2_bn = nn.BatchNorm2d(24)
        self.pw2 = nn.Conv2d(24, 48, kernel_size=1, stride=1, padding=0, bias=True)
        self.pw2_bn = nn.BatchNorm2d(48)

        # Stage 3: DW 3x3 s2 (48) + PW 1x1 (48 -> 96) -> 8x8x96
        self.dw3 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=48, bias=True)
        self.dw3_bn = nn.BatchNorm2d(48)
        self.pw3 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, bias=True)
        self.pw3_bn = nn.BatchNorm2d(96)

        # Stage 4: DW 3x3 s2 (96) + PW 1x1 (96 -> 160) -> 4x4x160
        self.dw4 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=True)
        self.dw4_bn = nn.BatchNorm2d(96)
        self.pw4 = nn.Conv2d(96, 160, kernel_size=1, stride=1, padding=0, bias=True)
        self.pw4_bn = nn.BatchNorm2d(160)

        # Stage 5: DW 3x3 s1 (160) + PW 1x1 (160 -> 192) -> 4x4x192
        self.dw5 = nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1, groups=160, bias=True)
        self.dw5_bn = nn.BatchNorm2d(160)
        self.pw5 = nn.Conv2d(160, 192, kernel_size=1, stride=1, padding=0, bias=True)
        self.pw5_bn = nn.BatchNorm2d(192)

        self.drop = nn.Dropout(0.15)
        self.fc1 = nn.Linear(192, 64, bias=True)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 3, bias=True)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = F.relu(self.dw2_bn(self.dw2(x)))
        x = F.relu(self.pw2_bn(self.pw2(x)))
        x = F.relu(self.dw3_bn(self.dw3(x)))
        x = F.relu(self.pw3_bn(self.pw3(x)))
        x = F.relu(self.dw4_bn(self.dw4(x)))
        x = F.relu(self.pw4_bn(self.pw4(x)))
        x = F.relu(self.dw5_bn(self.dw5(x)))
        x = F.relu(self.pw5_bn(self.pw5(x)))
        x = x.mean(dim=(2, 3))  # Global Average Pool -> (B, 192)
        x = self.drop(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x


PET_OVERLAP_TINY_CLASSES = {
    1, 4, 5, 7, 10, 11, 13, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 32,
    41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 62, 68, 70, 71, 76, 77, 83,
    91, 97, 98, 99, 104, 106, 108, 110, 111, 112, 115, 116, 117, 120,
    122, 124, 126, 132, 137, 138, 145, 146, 147, 150, 151, 152, 157,
    160, 161, 164, 166, 167, 169, 170, 174, 175, 185, 187, 188, 189,
}


def load_datasets():
    """Load Oxford-IIIT Pet + CIFAR-10 + Tiny-ImageNet (125 non-pet classes) + synthetic domain data."""
    t0 = time.time()
    train_items = []  # (PIL.Image RGB, label, high_res_flag)
    test_items = []
    rng = random.Random(42)

    # 1. Oxford-IIIT Pet (real high-res cats and dogs): balance ~1:1 (cats are ~1180, dogs ~2500 -> 6x cat, 3x dog)
    for split_name, dst, cat_rep, dog_rep in [
        ("pets_train.parquet", train_items, 6, 3),
        ("pets_test.parquet", test_items, 1, 1),
    ]:
        tbl = pq.read_table(DS_DIR / split_name, columns=["image", "label_cat_dog"])
        imgs = tbl["image"].to_pylist()
        lbls = tbl["label_cat_dog"].to_pylist()
        for im_d, lbl in zip(imgs, lbls):
            im = Image.open(io.BytesIO(im_d["bytes"])).convert("RGB")
            im = im.resize((112, 112), Image.Resampling.BILINEAR)
            lbl_i = int(lbl)
            rep = cat_rep if lbl_i == 0 else dog_rep
            for _ in range(rep):
                dst.append((im, lbl_i, True))

    # 2. Tiny-ImageNet (64x64 real objects/scenes across 125 pure non-pet UNKNOWN classes)
    for split_name, dst, max_unk in [
        ("tiny_imagenet_train.parquet", train_items, 13000),
        ("tiny_imagenet_valid.parquet", test_items, 2500),
    ]:
        tbl = pq.read_table(DS_DIR / split_name, columns=["image", "label"])
        imgs = tbl["image"].to_pylist()
        lbls = tbl["label"].to_pylist()
        unk_pool = []
        for im_d, lbl in zip(imgs, lbls):
            if int(lbl) not in PET_OVERLAP_TINY_CLASSES:
                im = Image.open(io.BytesIO(im_d["bytes"])).convert("RGB")
                unk_pool.append((im, 2, True))
        rng.shuffle(unk_pool)
        dst.extend(unk_pool[:max_unk])

    # 3. CIFAR-10 (cats=3, dogs=5, unknown=0,1,2,4,6,7,8,9)
    for split_name, dst, max_unk in [
        ("cifar10_train.parquet", train_items, 3000),
        ("cifar10_test.parquet", test_items, 1000),
    ]:
        tbl = pq.read_table(DS_DIR / split_name, columns=["img", "label"])
        imgs = tbl["img"].to_pylist()
        lbls = tbl["label"].to_pylist()
        unk_pool = []
        for im_d, lbl in zip(imgs, lbls):
            im = Image.open(io.BytesIO(im_d["bytes"])).convert("RGB")
            if lbl == 3:
                dst.append((im, 0, False))
            elif lbl == 5:
                dst.append((im, 1, False))
            else:
                unk_pool.append((im, 2, False))
        rng.shuffle(unk_pool)
        dst.extend(unk_pool[:max_unk])

    # 4. Synthetic Cartoon Cats/Dogs + Unknown patterns
    for _ in range(1500):
        train_items.append((make_synthetic_cartoon_pet(0, rng), 0, True))
        train_items.append((make_synthetic_cartoon_pet(1, rng), 1, True))
    for _ in range(2000):
        train_items.append((make_synthetic_unknown(rng), 2, True))

    for _ in range(150):
        test_items.append((make_synthetic_cartoon_pet(0, rng), 0, True))
        test_items.append((make_synthetic_cartoon_pet(1, rng), 1, True))
    for _ in range(300):
        test_items.append((make_synthetic_unknown(rng), 2, True))

    counts = [sum(1 for _, l, _ in train_items if l == c) for c in range(3)]
    print(f"Loaded {len(train_items)} train {counts}, {len(test_items)} test images in {time.time() - t0:.1f}s")
    return train_items, test_items


def augment_to_tensor(im: Image.Image, label: int, is_train: bool, rng: random.Random) -> torch.Tensor:
    """Apply realistic Studio-domain augmentations + RGB332 quantization -> 64x64 float tensor in [-1, 1]."""
    if is_train:
        w, h = im.size
        if rng.random() < 0.5:
            im = ImageOps.mirror(im)
        # Random independent X/Y crop + aspect-ratio stretch (covers 200x120 1.67:1 Studio distortion)
        if rng.random() < 0.85:
            cw = int(w * rng.uniform(0.58, 1.0))
            ch = int(h * rng.uniform(0.55, 1.0))
            x0 = rng.randint(0, max(0, w - cw))
            y0 = rng.randint(0, max(0, h - ch))
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
        # Meme text-bar + caption simulation on 22% of pet images
        if label in (0, 1) and rng.random() < 0.22:
            im = im.resize((96, 64), Image.Resampling.BILINEAR)
            d = ImageDraw.Draw(im)
            if rng.random() < 0.8:
                bh = rng.randint(9, 15)
                d.rectangle([2, 1, 94, bh], fill=(255, 255, 255), outline=(0, 0, 0))
                for tx in range(8, 86, 9):
                    d.line([tx, 4, tx + 5, 4], fill=(20, 20, 20), width=2)
                    if bh > 10:
                        d.line([tx, 9, tx + 6, 9], fill=(20, 20, 20), width=2)
            if rng.random() < 0.55:
                d.rectangle([4, 50, 92, 62], fill=(0, 0, 0), outline=(255, 255, 255))
        # Cartoon/posterize simulation on 12% of pet images
        if label in (0, 1) and rng.random() < 0.12:
            im = ImageOps.posterize(im, rng.choice([3, 4]))
            im = im.filter(ImageFilter.EDGE_ENHANCE)

    im64 = im.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(im64, dtype=np.float32)

    if is_train:
        # Brightness / contrast jitter
        alpha = rng.uniform(0.78, 1.25)
        beta = rng.uniform(-20.0, 20.0)
        arr = np.clip(arr * alpha + beta, 0.0, 255.0)
        # Fast RGB332 quantization simulation (75% chance)
        if rng.random() < 0.75:
            u8 = arr.astype(np.uint8)
            r3 = (u8[..., 0] >> 5) & 7
            g3 = (u8[..., 1] >> 5) & 7
            b2 = (u8[..., 2] >> 6) & 3
            arr[..., 0] = (r3 << 5) | (r3 << 2) | (r3 >> 1)
            arr[..., 1] = (g3 << 5) | (g3 << 2) | (g3 >> 1)
            arr[..., 2] = (b2 << 6) | (b2 << 4) | (b2 << 2) | b2
    else:
        u8 = arr.astype(np.uint8)
        r3 = (u8[..., 0] >> 5) & 7
        g3 = (u8[..., 1] >> 5) & 7
        b2 = (u8[..., 2] >> 6) & 3
        arr[..., 0] = (r3 << 5) | (r3 << 2) | (r3 >> 1)
        arr[..., 1] = (g3 << 5) | (g3 << 2) | (g3 >> 1)
        arr[..., 2] = (b2 << 6) | (b2 << 4) | (b2 << 2) | b2

    # Match C range [-127, 127] scaled by 1/128
    t = torch.from_numpy((arr - 128.0) / 128.0).permute(2, 0, 1).contiguous()
    return t


class PetMixDataset(torch.utils.data.Dataset):
    def __init__(self, items, is_train: bool, seed: int = 0):
        self.items = items
        self.is_train = is_train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        im, lbl, _ = self.items[idx]
        x = augment_to_tensor(im, lbl, self.is_train, self.rng)
        return x, lbl


def fold_bn(conv: nn.Module, bn: nn.Module):
    """Folds BatchNorm into Conv/Linear weights and bias in float32."""
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.shape[0])
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    inv_std = gamma / torch.sqrt(var + bn.eps)
    if isinstance(conv, nn.Conv2d):
        w_folded = w * inv_std.view(-1, 1, 1, 1)
    else:
        w_folded = w * inv_std.view(-1, 1)
    b_folded = (b - mean) * inv_std + beta
    return w_folded, b_folded


def quantize_folded_model_and_emulate(model: TinyMobileNet64, calib_loader):
    """
    Converts folded float32 layers to exact INT8 weights, INT32 biases, and power-of-two
    right-shift parameters matching `npu_core.v` v2 hardware arithmetic bit-for-bit.
    """
    model.eval()
    # Collect activation max magnitudes per layer on calibration batches
    act_p99 = {k: [] for k in ["stem", "dw2", "pw2", "dw3", "pw3", "dw4", "pw4", "dw5", "pw5", "gap", "fc1", "fc2"]}
    hooks = []

    def reg(name, mod):
        def fn(_m, _inp, out):
            vals = out.detach()
            if name != "fc2":
                vals = F.relu(vals)
                p = float(np.percentile(vals.cpu().numpy(), 99.9))
            else:
                p = float(vals.abs().max().item())
            act_p99[name].append(p)
        hooks.append(mod.register_forward_hook(fn))

    reg("stem", model.stem_bn)
    reg("dw2", model.dw2_bn)
    reg("pw2", model.pw2_bn)
    reg("dw3", model.dw3_bn)
    reg("pw3", model.pw3_bn)
    reg("dw4", model.dw4_bn)
    reg("pw4", model.pw4_bn)
    reg("dw5", model.dw5_bn)
    reg("pw5", model.pw5_bn)
    reg("fc1", model.fc1_bn)
    reg("fc2", model.fc2)

    with torch.no_grad():
        for i, (bx, _) in enumerate(calib_loader):
            model(bx)
            if i >= 20:
                break
    for h in hooks:
        h.remove()

    act_max = {k: (max(v) if v else 1.0) for k, v in act_p99.items()}

    layers_spec = [
        ("stem", model.stem_conv, model.stem_bn, 128.0),
        ("dw2", model.dw2, model.dw2_bn, None),
        ("pw2", model.pw2, model.pw2_bn, None),
        ("dw3", model.dw3, model.dw3_bn, None),
        ("pw3", model.pw3, model.pw3_bn, None),
        ("dw4", model.dw4, model.dw4_bn, None),
        ("pw4", model.pw4, model.pw4_bn, None),
        ("dw5", model.dw5, model.dw5_bn, None),
        ("pw5", model.pw5, model.pw5_bn, None),
        ("fc1", model.fc1, model.fc1_bn, None),
        ("fc2", model.fc2, None, None),
    ]

    q_layers = {}
    prev_act_scale = 128.0  # input float [-1, 1] -> int8 [-127, 127]

    for name, conv, bn, _ in layers_spec:
        if bn is not None:
            wf, bf = fold_bn(conv, bn)
        else:
            wf = conv.weight.detach().clone()
            bf = conv.bias.detach().clone()

        w_max = float(wf.abs().max().item()) or 1e-6
        w_scale = 127.0 / w_max
        w_q = torch.clamp(torch.round(wf * w_scale), -127, 127).to(torch.int32).numpy()

        if name != "fc2":
            # Choose out_act_scale so 99.9% of positive post-ReLU activations fit in [0, 127]
            a_max = max(0.5, act_max[name] * 1.02)
            out_scale = 127.0 / a_max
            # Combined multiplier = (prev_act_scale * w_scale) / out_scale ≈ 2^shift_bits
            ratio = (prev_act_scale * w_scale) / out_scale
            shift_bits = max(1, min(18, int(round(math.log2(max(2.0, ratio))))))
            eff_out_scale = (prev_act_scale * w_scale) / (2.0 ** shift_bits)
        else:
            # Final logits: choose shift_bits so output logits land nicely around [-2000, +2000]
            ratio = (prev_act_scale * w_scale) / 64.0
            shift_bits = max(1, min(18, int(round(math.log2(max(2.0, ratio))))))
            eff_out_scale = (prev_act_scale * w_scale) / (2.0 ** shift_bits)

        b_q = torch.round(bf * (prev_act_scale * w_scale)).to(torch.int64).numpy()
        q_layers[name] = {
            "w": w_q,
            "b": b_q.astype(np.int32),
            "shift": shift_bits,
            "in_scale": prev_act_scale,
            "out_scale": eff_out_scale,
        }
        prev_act_scale = eff_out_scale
        print(f"  Quant {name:5s}: w_scale={w_scale:6.1f}, shift={shift_bits:2d}, out_scale={eff_out_scale:6.2f}")

    return q_layers


def hw_round_shift(acc: np.ndarray, shift: int, relu: bool, clamp_i8: bool) -> np.ndarray:
    """Bit-exact hardware NPU post-processing: round-half-up -> arithmetic >> -> ReLU -> clamp."""
    acc = acc.astype(np.int64)
    if shift > 0:
        acc = acc + (1 << (shift - 1))
    val = acc >> shift
    if relu:
        val = np.maximum(val, 0)
    if clamp_i8:
        val = np.clip(val, -127, 127)
    return val.astype(np.int32)


def emulate_fpga_npu_cnn_single(img_64x64x3_i8: np.ndarray, q: dict) -> np.ndarray:
    """Bit-exact integer emulator of the C + FPGA `npu_core.v` v2 pipeline on 1 image (HWC int8)."""
    # Stem Conv 3x3 s2 p1 (64x64x3 -> 32x32x24)
    pad = np.pad(img_64x64x3_i8.astype(np.int64), ((1, 1), (1, 1), (0, 0)), mode="constant")
    w_stem = q["stem"]["w"].transpose(0, 2, 3, 1).reshape(24, 27).astype(np.int64)
    b_stem = q["stem"]["b"].astype(np.int64)
    out = np.zeros((32, 32, 24), dtype=np.int32)
    for oy in range(32):
        for ox in range(32):
            patch = pad[oy * 2 : oy * 2 + 3, ox * 2 : ox * 2 + 3, :].reshape(27)
            acc = b_stem + (w_stem @ patch)
            out[oy, ox, :] = hw_round_shift(acc, q["stem"]["shift"], relu=True, clamp_i8=True)

    def dw3x3(x, name, stride):
        H, W, C = x.shape
        oH, oW = (H // 2 if stride == 2 else H), (W // 2 if stride == 2 else W)
        xp = np.pad(x.astype(np.int64), ((1, 1), (1, 1), (0, 0)), mode="constant")
        w = q[name]["w"].reshape(C, 9).astype(np.int64)
        b = q[name]["b"].astype(np.int64)
        y = np.zeros((oH, oW, C), dtype=np.int32)
        for oy in range(oH):
            for ox in range(oW):
                patch = xp[oy * stride : oy * stride + 3, ox * stride : ox * stride + 3, :].reshape(9, C).T
                acc = b + (patch * w).sum(axis=1)
                y[oy, ox, :] = hw_round_shift(acc, q[name]["shift"], relu=True, clamp_i8=True)
        return y

    def pw1x1(x, name):
        H, W, Cin = x.shape
        Cout = q[name]["w"].shape[0]
        w = q[name]["w"].reshape(Cout, Cin).astype(np.int64)
        b = q[name]["b"].astype(np.int64)
        flat = x.reshape(H * W, Cin).astype(np.int64)
        acc = flat @ w.T + b[None, :]
        y = hw_round_shift(acc, q[name]["shift"], relu=True, clamp_i8=True)
        return y.reshape(H, W, Cout)

    out = dw3x3(out, "dw2", 2)
    out = pw1x1(out, "pw2")
    out = dw3x3(out, "dw3", 2)
    out = pw1x1(out, "pw3")
    out = dw3x3(out, "dw4", 2)
    out = pw1x1(out, "pw4")
    out = dw3x3(out, "dw5", 1)
    out = pw1x1(out, "pw5")

    # GAP 4x4 -> 192
    gap = (out.reshape(16, 192).sum(axis=0) + 8) >> 4
    gap = np.clip(gap, 0, 127).astype(np.int64)

    # FC1: 192 -> 64
    acc1 = q["fc1"]["b"].astype(np.int64) + (q["fc1"]["w"].astype(np.int64) @ gap)
    h1 = hw_round_shift(acc1, q["fc1"]["shift"], relu=True, clamp_i8=True).astype(np.int64)

    # FC2: 64 -> 3
    acc2 = q["fc2"]["b"].astype(np.int64) + (q["fc2"]["w"].astype(np.int64) @ h1)
    logits = hw_round_shift(acc2, q["fc2"]["shift"], relu=False, clamp_i8=False)
    return logits


def export_c_header(q: dict):
    """Export bit-exact C weights & biases into firmware/src/app/npu_weights.h."""
    def pack_matrix_u32(mat_oc_in: np.ndarray, pad_to_4: bool = True) -> list[int]:
        oc, k = mat_oc_in.shape
        if pad_to_4 and (k % 4 != 0):
            pad_k = ((k + 3) // 4) * 4
            padded = np.zeros((oc, pad_k), dtype=np.int8)
            padded[:, :k] = mat_oc_in.astype(np.int8)
            mat_oc_in = padded
        flat = mat_oc_in.astype(np.uint8).reshape(-1)
        words = []
        for i in range(0, len(flat), 4):
            w = int(flat[i]) | (int(flat[i + 1]) << 8) | (int(flat[i + 2]) << 16) | (int(flat[i + 3]) << 24)
            words.append(w)
        return words

    def emit_u32_array(name: str, vals: list[int]) -> list[str]:
        out = [f"static const uint32_t {name}[{len(vals)}] = {{"]
        for i in range(0, len(vals), 12):
            out.append("    " + ", ".join(f"0x{v:08X}UL" for v in vals[i : i + 12]) + ",")
        out.append("};")
        return out

    def emit_i32_array(name: str, vals: np.ndarray) -> list[str]:
        vlist = [int(x) for x in vals.reshape(-1)]
        out = [f"static const int32_t {name}[{len(vlist)}] = {{"]
        for i in range(0, len(vlist), 12):
            out.append("    " + ", ".join(str(v) for v in vlist[i : i + 12]) + ",")
        out.append("};")
        return out

    def emit_i8_array(name: str, vals: np.ndarray) -> list[str]:
        vlist = [int(x) for x in vals.reshape(-1)]
        out = [f"static const int8_t {name}[{len(vlist)}] = {{"]
        for i in range(0, len(vlist), 16):
            out.append("    " + ", ".join(str(v) for v in vlist[i : i + 16]) + ",")
        out.append("};")
        return out

    # Reorder Conv weights from PyTorch (Cout, Cin, H, W) -> HWC (Cout, H, W, Cin) so HWC patches match directly!
    stem_hwc = q["stem"]["w"].transpose(0, 2, 3, 1).reshape(24, 27)
    stem_words = pack_matrix_u32(stem_hwc, pad_to_4=True)

    lines = [
        "/* AUTO-GENERATED BY tools/npu/train_cnn_npu.py -- DO NOT EDIT */",
        "#ifndef APP_NPU_WEIGHTS_H_",
        "#define APP_NPU_WEIGHTS_H_",
        "",
        "#include <stdint.h>",
        "",
        "#define NPU_STEM_OUT_CH    24U",
        "#define NPU_STEM_VEC_WORDS 7U",
        f"#define NPU_STEM_SHIFT     {q['stem']['shift']}U",
        f"#define NPU_DW2_SHIFT      {q['dw2']['shift']}U",
        f"#define NPU_PW2_SHIFT      {q['pw2']['shift']}U",
        f"#define NPU_DW3_SHIFT      {q['dw3']['shift']}U",
        f"#define NPU_PW3_SHIFT      {q['pw3']['shift']}U",
        f"#define NPU_DW4_SHIFT      {q['dw4']['shift']}U",
        f"#define NPU_PW4_SHIFT      {q['pw4']['shift']}U",
        f"#define NPU_DW5_SHIFT      {q['dw5']['shift']}U",
        f"#define NPU_PW5_SHIFT      {q['pw5']['shift']}U",
        f"#define NPU_FC1_SHIFT      {q['fc1']['shift']}U",
        f"#define NPU_FC2_SHIFT      {q['fc2']['shift']}U",
        "",
    ]

    lines += emit_u32_array("k_stem_w_packed", stem_words)
    lines += emit_i32_array("k_stem_b", q["stem"]["b"])

    for idx in (2, 3, 4, 5):
        dw_name = f"dw{idx}"
        pw_name = f"pw{idx}"
        dw_w = q[dw_name]["w"].reshape(-1)
        lines += emit_i8_array(f"k_{dw_name}_w", dw_w)
        lines += emit_i32_array(f"k_{dw_name}_b", q[dw_name]["b"])

        pw_mat = q[pw_name]["w"].reshape(q[pw_name]["w"].shape[0], q[pw_name]["w"].shape[1])
        lines += emit_u32_array(f"k_{pw_name}_w_packed", pack_matrix_u32(pw_mat))
        lines += emit_i32_array(f"k_{pw_name}_b", q[pw_name]["b"])

    lines += emit_u32_array("k_fc1_w_packed", pack_matrix_u32(q["fc1"]["w"]))
    lines += emit_i32_array("k_fc1_b", q["fc1"]["b"])
    lines += emit_u32_array("k_fc2_w_packed", pack_matrix_u32(q["fc2"]["w"]))
    lines += emit_i32_array("k_fc2_b", q["fc2"]["b"])
    lines += ["", "#endif /* APP_NPU_WEIGHTS_H_ */", ""]

    WEIGHTS_H.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported quantized CNN weights to {WEIGHTS_H}")


def main():
    train_items, test_items = load_datasets()
    train_ds = PetMixDataset(train_items, is_train=True, seed=101)
    test_ds = PetMixDataset(test_items, is_train=False, seed=202)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    calib_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=True, num_workers=0)

    counts = np.array([sum(1 for _, l, _ in train_items if l == c) for c in range(3)], dtype=np.float32)
    inv = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    cls_weights = torch.tensor(inv, dtype=torch.float32)
    print(f"Class loss weights (CAT, DOG, UNK): {[round(float(x), 3) for x in inv]}")

    model = TinyMobileNet64()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TinyMobileNet64 parameters: {total_params:,}")

    epochs = 22
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0.0
    best_state = None

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        t_ep = time.time()
        for bx, by in train_loader:
            logits = model(bx)
            loss = F.cross_entropy(logits, by, weight=cls_weights, label_smoothing=0.05)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * len(by)
            correct += (logits.argmax(dim=1) == by).sum().item()
            total += len(by)
        sched.step()

        # Validation + per-class recall
        model.eval()
        v_ok, v_tot = 0, 0
        c_ok = [0, 0, 0]
        c_tot = [0, 0, 0]
        with torch.no_grad():
            for bx, by in test_loader:
                preds = model(bx).argmax(dim=1)
                v_ok += (preds == by).sum().item()
                v_tot += len(by)
                for c in range(3):
                    m = by == c
                    c_tot[c] += int(m.sum().item())
                    c_ok[c] += int((preds[m] == c).sum().item())
        val_acc = 100.0 * v_ok / v_tot
        bal_acc = 100.0 * sum(c_ok[c] / max(1, c_tot[c]) for c in range(3)) / 3.0
        if bal_acc > best_val:
            best_val = bal_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        c_str = "/".join(f"{100*c_ok[c]/max(1,c_tot[c]):4.1f}" for c in range(3))
        print(
            f"Epoch {ep+1:2d}/{epochs} ({time.time()-t_ep:4.1f}s) | "
            f"loss={loss_sum/total:.4f} tr={100*correct/total:5.2f}% | "
            f"val={val_acc:5.2f}% bal={bal_acc:5.2f}% (best={best_val:5.2f}%) [C/D/U={c_str}]"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Quantize & calibrate
    print("\nQuantizing folded CNN to hardware INT8/INT32 parameters...")
    q_layers = quantize_folded_model_and_emulate(model, calib_loader)
    export_c_header(q_layers)

    # Evaluate bit-exact integer hardware emulator on test subset (600 images)
    rng = random.Random(77)
    eval_idxs = list(range(len(test_items)))
    rng.shuffle(eval_idxs)
    eval_idxs = eval_idxs[:600]
    q_ok = 0
    qc_ok = [0, 0, 0]
    qc_tot = [0, 0, 0]
    for idx in eval_idxs:
        im, lbl, _ = test_items[idx]
        im200 = im.resize((200, 120), Image.Resampling.BILINEAR)
        fb = rgb_to_332_arr(np.asarray(im200, dtype=np.uint8), dither=True)
        in_i8 = c_exact_box_downsample_rgb332_to_64x64(fb)
        logits = emulate_fpga_npu_cnn_single(in_i8, q_layers)
        pred = int(np.argmax(logits))
        qc_tot[lbl] += 1
        if pred == lbl:
            q_ok += 1
            qc_ok[lbl] += 1
    qc_str = "/".join(f"{qc_ok[c]}/{qc_tot[c]}" for c in range(3))
    print(f"\nBit-exact FPGA NPU INT8 emulator val accuracy (600 held-out RGB332+dithered): {q_ok}/600 ({100*q_ok/600:.2f}%) [C/D/U={qc_str}]")

    # Zero-shot evaluation on all real images in user's Downloads folder!
    dl = pathlib.Path(r"C:\Users\penis\Downloads")
    cls_names = ["CAT", "DOG", "UNKNOWN"]
    gt_cats = {
        "86.jpg",
        "istockphoto-1369620685-612x612.jpg",
        "1e99014b917eb63ad94bb8b7fc9c50a4.jpg",
        "cat_meme_800x480.png",
        "cat_meme_800x480_2.png",
        "Коты за работой_ корм сам не оплатится.png",
        "Кошачий спор из-за пиццы-2.png",
        "Кошачья проверка на доверие-1.png",
    }
    gt_dogs = {
        "347756718380326.jpeg",
        "a-cute-little-dog-sitting-ai-generative-free-png.webp",
    }
    print("\nZero-shot evaluation on user's Downloads images (via 200x120 RGB332 Floyd-Steinberg):")
    dl_ok = 0
    dl_tot = 0
    for p in sorted(dl.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        exp = 0 if p.name in gt_cats else (1 if p.name in gt_dogs else 2)
        im = Image.open(p).convert("RGB").resize((200, 120), Image.Resampling.BILINEAR)
        fb = rgb_to_332_arr(np.asarray(im, dtype=np.uint8), dither=True)
        in_i8 = c_exact_box_downsample_rgb332_to_64x64(fb)
        logits = emulate_fpga_npu_cnn_single(in_i8, q_layers)
        pred = int(np.argmax(logits))
        ok = pred == exp
        dl_tot += 1
        if ok:
            dl_ok += 1
        mark = "OK  " if ok else "MISS"
        print(f"  {mark} {p.name[:42]:42s} exp={cls_names[exp]:7s} pred={cls_names[pred]:7s} | logits={logits.tolist()}")
    print(f"Downloads zero-shot score: {dl_ok}/{dl_tot} ({100.0*dl_ok/max(1,dl_tot):.1f}%)")


if __name__ == "__main__":
    main()
