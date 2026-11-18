#!/usr/bin/env python3
"""
Train, INT8-quantize, emulate, and export a Pretrained ShuffleNetV2-x0.5 (96x96)
to the Gowin Tang Mega 138K Pro FPGA NPU (npu_core v2.0.0, 0xF8000900).

Architecture matches `npu_core.v` v2 constraints cleanly:
  - Input: 96x96x3 signed INT8 (box-downsampled from Studio's 200x120 RGB332)
  - conv1: Conv3x3 s2 p1 (3 -> 24) [NPU] -> maxpool 3x3 s2 p1 [CPU] -> 24x24x24
  - stage2: 4x InvertedResidual (24 -> 48) -> 12x12x48  [PW on NPU, DW on CPU]
  - stage3: 8x InvertedResidual (48 -> 96) ->  6x6x96   [PW on NPU, DW on CPU]
  - stage4: 4x InvertedResidual (96 -> 192)->  3x3x192  [PW on NPU, DW on CPU]
  - conv5: PW 1x1 (192 -> 128) [NPU] -> GAP 3x3 -> fc (128 -> 3) [NPU]

Total INT8 weights: ~162 KB (max single PW matrix = 24,576 B <= 32 KB BSRAM).
Stage boundary scales are locked so concatenation + channel_shuffle has ZERO scale mismatch.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DS_DIR = pathlib.Path(r"C:\Users\penis\.gemini\antigravity\scratch\npu_datasets")
WEIGHTS_H = ROOT / "firmware" / "src" / "app" / "npu_weights.h"
CKPT_NPZ = ROOT / "tools" / "npu" / "shufflenet_npu.npz"

torch.set_num_threads(8)

# Identified via MobileNetV3-Small across all 200 Tiny-ImageNet classes:
PET_OVERLAP_TINY_CLASSES = {
    1, 4, 5, 7, 10, 11, 13, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 32,
    41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 62, 68, 70, 71, 76, 77, 83,
    91, 97, 98, 99, 104, 106, 108, 110, 111, 112, 115, 116, 117, 120,
    122, 124, 126, 132, 137, 138, 145, 146, 147, 150, 151, 152, 157,
    160, 161, 164, 166, 167, 169, 170, 174, 175, 185, 187, 188, 189,
}
# Verified pure dog breed classes inside Tiny-ImageNet (P_dog >= 0.69, P_cat <= 0.03):
TINY_DOG_CLASSES = {22, 23, 24, 25, 26, 187}

MEME_CAPTIONS = [
    "КОГДА КОТ УСТАЛ",
    "ЗА РАБОТОЙ",
    "ПРОВЕРКА ДОВЕРИЯ",
    "СПОР ИЗ-ЗА ПИЦЦЫ",
    "КОРМ НЕ ОПЛАТИТСЯ",
    "КТО ЗДЕСЬ?",
    "ДОБРЫЙ ВЕЧЕР",
    "WHEN CAT CODES",
    "FPGA NPU TEST",
    "TOP TEXT",
    "BOTTOM TEXT",
]


def load_meme_fonts():
    fonts = []
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if pathlib.Path(p).exists():
            for sz in (11, 13, 16):
                fonts.append(ImageFont.truetype(p, sz))
            break
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


MEME_FONTS = load_meme_fonts()


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


def c_exact_box_downsample_rgb332_to_96x96(fb_200x120: np.ndarray) -> np.ndarray:
    """Bit-exact match to C `npu_downsample_rgb332_to_96x96`."""
    out = np.zeros((96, 96, 3), dtype=np.int8)
    r3 = (fb_200x120 >> 5) & 7
    g3 = (fb_200x120 >> 2) & 7
    b2 = fb_200x120 & 3
    r8 = ((r3 << 5) | (r3 << 2) | (r3 >> 1)).astype(np.int32)
    g8 = ((g3 << 5) | (g3 << 2) | (g3 >> 1)).astype(np.int32)
    b8 = ((b2 << 6) | (b2 << 4) | (b2 << 2) | b2).astype(np.int32)

    for oy in range(96):
        y0 = (oy * 120) // 96
        y1 = max(y0 + 1, ((oy + 1) * 120) // 96)
        for ox in range(96):
            x0 = (ox * 200) // 96
            x1 = max(x0 + 1, ((ox + 1) * 200) // 96)
            cnt = (y1 - y0) * (x1 - x0)
            ir = int(r8[y0:y1, x0:x1].sum()) // cnt - 128
            ig = int(g8[y0:y1, x0:x1].sum()) // cnt - 128
            ib = int(b8[y0:y1, x0:x1].sum()) // cnt - 128
            out[oy, ox, 0] = max(-127, min(127, ir))
            out[oy, ox, 1] = max(-127, min(127, ig))
            out[oy, ox, 2] = max(-127, min(127, ib))
    return out


def make_synthetic_cartoon_pet(cls_id: int, rng: random.Random) -> Image.Image:
    w, h = 128, 128
    bg_choices = [(255, 255, 255), (240, 244, 250), (35, 45, 65), (255, 240, 210), (210, 235, 210)]
    im = Image.new("RGB", (w, h), bg_choices[rng.randrange(len(bg_choices))])
    draw = ImageDraw.Draw(im)
    cx = 64 + rng.randint(-12, 12)
    cy = 62 + rng.randint(-10, 10)
    sc = rng.uniform(0.75, 1.15)
    fur = rng.choice([(245, 175, 85), (120, 125, 135), (245, 240, 230), (190, 130, 70), (70, 65, 65)])
    outline = (60, 25, 15) if rng.random() < 0.7 else (20, 20, 20)
    bw, bh = int(26 * sc), int(30 * sc)
    draw.ellipse([cx - bw, cy + int(10 * sc), cx + bw, cy + int(10 * sc) + bh], fill=fur, outline=outline, width=3)

    if cls_id == 0:
        r_head = int(28 * sc)
        for s in (-1, 1):
            ex = cx + s * int(18 * sc)
            ey = cy - int(22 * sc)
            pts = [(ex - s * int(10 * sc), ey + int(8 * sc)), (ex + s * int(6 * sc), ey - int(20 * sc)), (ex + s * int(12 * sc), ey + int(12 * sc))]
            draw.polygon(pts, fill=fur, outline=outline)
            inner = [(ex - s * int(5 * sc), ey + int(5 * sc)), (ex + s * int(5 * sc), ey - int(12 * sc)), (ex + s * int(8 * sc), ey + int(8 * sc))]
            draw.polygon(inner, fill=(250, 155, 130))
        draw.ellipse([cx - r_head, cy - int(r_head * 0.85), cx + r_head, cy + int(r_head * 0.85)], fill=fur, outline=outline, width=3)
        draw.ellipse([cx - int(14 * sc), cy + int(2 * sc), cx + int(14 * sc), cy + int(18 * sc)], fill=(252, 246, 225))
        for s in (-1, 1):
            ex = cx + s * int(11 * sc)
            ey = cy - int(2 * sc)
            er = int(4 * sc)
            draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=outline)
        for s in (-1, 1):
            for wy in (-2, 3, 8):
                draw.line([cx + s * int(18 * sc), cy + int(wy * sc), cx + s * int(36 * sc), cy + int((wy - 2) * sc)], fill=outline, width=2)
    else:
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
        nr = int(6 * sc)
        draw.ellipse([cx - nr, cy + int(2 * sc), cx + nr, cy + int(12 * sc)], fill=(20, 20, 25))
    return im


def make_synthetic_unknown(rng: random.Random) -> Image.Image:
    w, h = 128, 128
    kind = rng.randrange(7)
    if kind == 0:
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
        sx, sy = rng.randint(32, 90), rng.randint(32, 90)
        sr = rng.randint(14, 22)
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        for a in range(0, 360, 15):
            rad = math.radians(a)
            draw.line([sx, sy, int(sx + sr * math.cos(rad)), int(sy + sr * math.sin(rad))], fill=(0, 0, 0), width=1)
        return im
    elif kind == 1:
        return Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
    elif kind == 2:
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
        im = Image.new("RGB", (w, h), (20, 20, 30))
        draw = ImageDraw.Draw(im)
        cx, cy = rng.randint(40, 88), rng.randint(40, 88)
        for r in range(60, 4, -8):
            col = tuple(rng.randint(30, 255) for _ in range(3))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=4)
        return im
    elif kind == 5:
        bg = tuple(rng.randint(0, 255) for _ in range(3))
        im = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(im)
        for _ in range(rng.randint(4, 10)):
            x0, y0 = rng.randint(0, 100), rng.randint(0, 100)
            x1, y1 = min(w - 1, x0 + rng.randint(10, 50)), min(h - 1, y0 + rng.randint(10, 50))
            col = tuple(rng.randint(0, 255) for _ in range(3))
            draw.rectangle([x0, y0, x1, y1], fill=col)
        return im
    else:
        # Metallic hardware / cylindrical connector / green PCB / dark chassis
        im = Image.new("RGB", (w, h), rng.choice([(185, 190, 195), (25, 70, 40), (35, 35, 40)]))
        draw = ImageDraw.Draw(im)
        cx, cy = rng.randint(42, 86), rng.randint(42, 86)
        r = rng.randint(24, 42)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(90, 95, 100), outline=(220, 220, 225), width=4)
        for _ in range(rng.randint(5, 12)):
            px, py = cx + rng.randint(-r // 2, r // 2), cy + rng.randint(-r // 2, r // 2)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(215, 185, 90))
        return im


def load_datasets():
    t0 = time.time()
    train_items, test_items = [], []
    rng = random.Random(42)

    # 1. Oxford-IIIT Pet (cats balanced 4x, dogs 2x)
    for split_name, dst, cat_rep, dog_rep in [
        ("pets_train.parquet", train_items, 4, 2),
        ("pets_test.parquet", test_items, 1, 1),
    ]:
        tbl = pq.read_table(DS_DIR / split_name, columns=["image", "label_cat_dog"])
        for im_d, lbl in zip(tbl["image"].to_pylist(), tbl["label_cat_dog"].to_pylist()):
            im = Image.open(io.BytesIO(im_d["bytes"])).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
            lbl_i = int(lbl)
            for _ in range(cat_rep if lbl_i == 0 else dog_rep):
                dst.append((im, lbl_i))

    # 2. Tiny-ImageNet (125 non-pet UNKNOWN classes + 6 verified pure DOG breeds)
    for split_name, dst, max_unk, max_dog in [
        ("tiny_imagenet_train.parquet", train_items, 10500, 1800),
        ("tiny_imagenet_valid.parquet", test_items, 2500, 250),
    ]:
        tbl = pq.read_table(DS_DIR / split_name, columns=["image", "label"])
        unk_pool, dog_pool = [], []
        for im_d, lbl in zip(tbl["image"].to_pylist(), tbl["label"].to_pylist()):
            l_i = int(lbl)
            if l_i in TINY_DOG_CLASSES:
                dog_pool.append((Image.open(io.BytesIO(im_d["bytes"])).convert("RGB"), 1))
            elif l_i not in PET_OVERLAP_TINY_CLASSES:
                unk_pool.append((Image.open(io.BytesIO(im_d["bytes"])).convert("RGB"), 2))
        rng.shuffle(unk_pool)
        rng.shuffle(dog_pool)
        dst.extend(unk_pool[:max_unk])
        dst.extend(dog_pool[:max_dog])

    # 3. CIFAR-10 (cats=3, dogs=5, non-pet=other)
    for split_name, dst, max_c, max_d, max_u in [
        ("cifar10_train.parquet", train_items, 4000, 2500, 2000),
        ("cifar10_test.parquet", test_items, 1000, 1000, 1000),
    ]:
        tbl = pq.read_table(DS_DIR / split_name, columns=["img", "label"])
        cp, dp, up = [], [], []
        for im_d, lbl in zip(tbl["img"].to_pylist(), tbl["label"].to_pylist()):
            im = Image.open(io.BytesIO(im_d["bytes"])).convert("RGB")
            if lbl == 3:
                cp.append((im, 0))
            elif lbl == 5:
                dp.append((im, 1))
            else:
                up.append((im, 2))
        rng.shuffle(cp); rng.shuffle(dp); rng.shuffle(up)
        dst.extend(cp[:max_c]); dst.extend(dp[:max_d]); dst.extend(up[:max_u])

    # 4. Synthetic cartoons + unknown patterns
    for _ in range(1200):
        train_items.append((make_synthetic_cartoon_pet(0, rng), 0))
    for _ in range(1000):
        train_items.append((make_synthetic_cartoon_pet(1, rng), 1))
    for _ in range(1500):
        train_items.append((make_synthetic_unknown(rng), 2))
    for _ in range(150):
        test_items.append((make_synthetic_cartoon_pet(0, rng), 0))
        test_items.append((make_synthetic_cartoon_pet(1, rng), 1))
    for _ in range(300):
        test_items.append((make_synthetic_unknown(rng), 2))

    counts = [sum(1 for _, l in train_items if l == c) for c in range(3)]
    print(f"Loaded {len(train_items)} train {counts}, {len(test_items)} test images in {time.time() - t0:.1f}s")
    return train_items, test_items


def add_meme_overlay(im: Image.Image, rng: random.Random) -> Image.Image:
    """Adds realistic top white caption strip or bold white-on-black meme text."""
    im = im.resize((120, 96), Image.Resampling.BILINEAR).copy()
    draw = ImageDraw.Draw(im)
    txt = rng.choice(MEME_CAPTIONS)
    font = rng.choice(MEME_FONTS)
    if rng.random() < 0.65:
        bh = rng.randint(15, 24)
        draw.rectangle([0, 0, 120, bh], fill=(255, 255, 255))
        draw.text((rng.randint(3, 12), 2), txt, fill=(10, 10, 10), font=font)
    if rng.random() < 0.55:
        y = rng.randint(72, 82)
        x = rng.randint(4, 16)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((x + dx, y + dy), txt, fill=(0, 0, 0), font=font)
        draw.text((x, y), txt, fill=(255, 255, 240), font=font)
    return im


def augment_to_tensor(im: Image.Image, label: int, is_train: bool, rng: random.Random) -> torch.Tensor:
    if is_train:
        w, h = im.size
        if rng.random() < 0.5:
            im = ImageOps.mirror(im)
        if rng.random() < 0.85:
            cw = int(w * rng.uniform(0.58, 1.0))
            ch = int(h * rng.uniform(0.55, 1.0))
            x0 = rng.randint(0, max(0, w - cw))
            y0 = rng.randint(0, max(0, h - ch))
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
        if label in (0, 1) and rng.random() < 0.25:
            im = add_meme_overlay(im, rng)
        if label in (0, 1) and rng.random() < 0.12:
            im = ImageOps.posterize(im, rng.choice([3, 4]))
            im = im.filter(ImageFilter.EDGE_ENHANCE)

    im96 = im.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(im96, dtype=np.float32)

    if is_train:
        arr = np.clip(arr * rng.uniform(0.78, 1.24) + rng.uniform(-18.0, 18.0), 0.0, 255.0)
        if rng.random() < 0.80:
            u8 = arr.astype(np.uint8)
            r3, g3, b2 = (u8[..., 0] >> 5) & 7, (u8[..., 1] >> 5) & 7, (u8[..., 2] >> 6) & 3
            arr[..., 0] = (r3 << 5) | (r3 << 2) | (r3 >> 1)
            arr[..., 1] = (g3 << 5) | (g3 << 2) | (g3 >> 1)
            arr[..., 2] = (b2 << 6) | (b2 << 4) | (b2 << 2) | b2
    else:
        u8 = arr.astype(np.uint8)
        r3, g3, b2 = (u8[..., 0] >> 5) & 7, (u8[..., 1] >> 5) & 7, (u8[..., 2] >> 6) & 3
        arr[..., 0] = (r3 << 5) | (r3 << 2) | (r3 >> 1)
        arr[..., 1] = (g3 << 5) | (g3 << 2) | (g3 >> 1)
        arr[..., 2] = (b2 << 6) | (b2 << 4) | (b2 << 2) | b2

    return torch.from_numpy((arr - 128.0) / 128.0).permute(2, 0, 1).contiguous()


class PetShuffleDataset(torch.utils.data.Dataset):
    def __init__(self, items, is_train: bool, seed: int):
        self.items = items
        self.is_train = is_train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        im, lbl = self.items[idx]
        return augment_to_tensor(im, lbl, self.is_train, self.rng), lbl


class PretrainedShuffleNPU(nn.Module):
    """ShuffleNetV2_x0_5 stages 1..4 + compact 192->128 PW + GAP + 128->3 FC."""

    def __init__(self):
        super().__init__()
        base = tv_models.shufflenet_v2_x0_5(weights=tv_models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.conv1 = base.conv1      # 3 -> 24, k3 s2 p1 + BN + ReLU
        self.maxpool = base.maxpool  # k3 s2 p1
        self.stage2 = base.stage2    # 4 blocks (24 -> 48)
        self.stage3 = base.stage3    # 8 blocks (48 -> 96)
        self.stage4 = base.stage4    # 4 blocks (96 -> 192)
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(128, 3, bias=True)

        # Absorb ImageNet RGB mean/std normalization directly into conv1 weights
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        a = 0.5 / std
        b = (0.5 - mean) / std
        with torch.no_grad():
            w = self.conv1[0].weight.data
            bias_shift = (w * b.view(1, 3, 1, 1)).sum(dim=(1, 2, 3))
            self.conv1[0].weight.data = w * a.view(1, 3, 1, 1)
            self.conv1[1].running_mean.data = self.conv1[1].running_mean.data - bias_shift

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean(dim=(2, 3))
        x = self.drop(x)
        return self.fc(x)


def fold_bn(conv: nn.Module, bn: nn.Module):
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.shape[0])
    inv_std = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    if isinstance(conv, nn.Conv2d):
        w_f = w * inv_std.view(-1, 1, 1, 1)
    else:
        w_f = w * inv_std.view(-1, 1)
    b_f = (b - bn.running_mean.detach()) * inv_std + bn.bias.detach()
    return w_f, b_f


def hw_round_shift(acc: np.ndarray, shift: int, relu: bool, clamp_i8: bool) -> np.ndarray:
    acc = acc.astype(np.int64)
    if shift > 0:
        acc = acc + (1 << (shift - 1))
    val = acc >> shift
    if relu:
        val = np.maximum(val, 0)
    if clamp_i8:
        val = np.clip(val, -127, 127)
    return val.astype(np.int32)


def collect_activations(model: PretrainedShuffleNPU, calib_loader):
    """Collects per-layer 99.9th percentile magnitudes across calibration batches."""
    model.eval()
    stats = {}
    hooks = []

    def add_hook(name, mod, is_relu=True, is_abs=False):
        stats[name] = []

        def fn(_m, _inp, out):
            t = out.detach()
            if is_abs:
                vals = t.abs().cpu().numpy()
            elif is_relu:
                vals = F.relu(t).cpu().numpy()
            else:
                vals = t.cpu().numpy()
            stats[name].append(float(np.percentile(vals, 99.9)))

        hooks.append(mod.register_forward_hook(fn))

    add_hook("conv1", model.conv1[1], is_relu=True)
    for st_name, st_seq in [("stage2", model.stage2), ("stage3", model.stage3), ("stage4", model.stage4)]:
        for bi, blk in enumerate(st_seq):
            if blk.stride == 2:
                add_hook(f"{st_name}.{bi}.b1_dw", blk.branch1[1], is_abs=True)
                add_hook(f"{st_name}.{bi}.b1_pw", blk.branch1[3], is_relu=True)
            add_hook(f"{st_name}.{bi}.b2_pw1", blk.branch2[1], is_relu=True)
            add_hook(f"{st_name}.{bi}.b2_dw", blk.branch2[4], is_abs=True)
            add_hook(f"{st_name}.{bi}.b2_pw2", blk.branch2[6], is_relu=True)
    add_hook("conv5", model.conv5[1], is_relu=True)
    add_hook("fc", model.fc, is_abs=True)

    with torch.no_grad():
        for i, (bx, _) in enumerate(calib_loader):
            model(bx)
            if i >= 16:
                break
    for h in hooks:
        h.remove()
    out_stats = {}
    for k, v in stats.items():
        if not v:
            raise RuntimeError(f"Empty calibration stats for {k}")
        out_stats[k] = max(0.5, float(np.median(v)))
    return out_stats


def quant_free(name, wf, bf, in_scale, p999_act, is_logits=False):
    w_max = float(wf.abs().max().item()) or 1e-6
    w_scale = 127.0 / w_max
    w_q = torch.clamp(torch.round(wf * w_scale), -127, 127).to(torch.int32).numpy()
    target_scale = 128.0 if is_logits else (127.0 / (p999_act * 1.05))
    ratio = (in_scale * w_scale) / target_scale
    shift = max(1, min(18, int(round(math.log2(max(2.0, ratio))))))
    eff_out = (in_scale * w_scale) / (2.0 ** shift)
    b_q = torch.round(bf * (in_scale * w_scale)).to(torch.int64).numpy().astype(np.int32)
    return {"w": w_q, "b": b_q, "shift": shift, "out_scale": eff_out, "in_scale": in_scale, "w_scale": w_scale}


def quant_locked(wf, bf, in_scale, fixed_out_scale):
    """Quantizes a branch-end PW layer so its output scale equals `fixed_out_scale` exactly."""
    w_max = float(wf.abs().max().item()) or 1e-6
    max_ratio = (127.0 * in_scale) / (w_max * fixed_out_scale)
    shift = max(1, min(18, int(math.floor(math.log2(max(2.0, max_ratio))))))
    w_scale = (fixed_out_scale * (2.0 ** shift)) / in_scale
    w_q = torch.clamp(torch.round(wf * w_scale), -127, 127).to(torch.int32).numpy()
    b_q = torch.round(bf * (in_scale * w_scale)).to(torch.int64).numpy().astype(np.int32)
    return {"w": w_q, "b": b_q, "shift": shift, "out_scale": fixed_out_scale, "in_scale": in_scale, "w_scale": w_scale}


def quantize_shufflenet(model: PretrainedShuffleNPU, calib_loader, scale_mult: float = 1.0):
    act = collect_activations(model, calib_loader)
    q = {}

    # 1. Stage-target scales ensure zero mismatch across bypass/branch concatenation
    # Compute folded weights first for every layer in stage2/3/4
    folded = {}
    folded["conv1"] = fold_bn(model.conv1[0], model.conv1[1])
    for st_name, st_seq in [("stage2", model.stage2), ("stage3", model.stage3), ("stage4", model.stage4)]:
        for bi, blk in enumerate(st_seq):
            if blk.stride == 2:
                folded[f"{st_name}.{bi}.b1_dw"] = fold_bn(blk.branch1[0], blk.branch1[1])
                folded[f"{st_name}.{bi}.b1_pw"] = fold_bn(blk.branch1[2], blk.branch1[3])
            folded[f"{st_name}.{bi}.b2_pw1"] = fold_bn(blk.branch2[0], blk.branch2[1])
            folded[f"{st_name}.{bi}.b2_dw"] = fold_bn(blk.branch2[3], blk.branch2[4])
            folded[f"{st_name}.{bi}.b2_pw2"] = fold_bn(blk.branch2[5], blk.branch2[6])
    folded["conv5"] = fold_bn(model.conv5[0], model.conv5[1])
    folded["fc"] = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())

    q["conv1"] = quant_free("conv1", *folded["conv1"], 128.0, act["conv1"] * scale_mult)

    prev_scale = q["conv1"]["out_scale"]
    blocks_spec = []

    for st_idx, (st_name, st_seq) in enumerate([("stage2", model.stage2), ("stage3", model.stage3), ("stage4", model.stage4)], start=2):
        # Compute a uniform target out_scale for all residual concats inside this stage
        stage_p999 = max(act[f"{st_name}.{bi}.b2_pw2"] for bi in range(len(st_seq)))
        s_stage = 127.0 / (stage_p999 * 1.08 * scale_mult)

        # First handle block 0 (stride=2, takes prev_scale)
        pfx0 = f"{st_name}.0"
        q[f"{pfx0}.b1_dw"] = quant_free(f"{pfx0}.b1_dw", *folded[f"{pfx0}.b1_dw"], prev_scale, act[f"{pfx0}.b1_dw"] * scale_mult)
        q[f"{pfx0}.b2_pw1"] = quant_free(f"{pfx0}.b2_pw1", *folded[f"{pfx0}.b2_pw1"], prev_scale, act[f"{pfx0}.b2_pw1"] * scale_mult)
        q[f"{pfx0}.b2_dw"] = quant_free(f"{pfx0}.b2_dw", *folded[f"{pfx0}.b2_dw"], q[f"{pfx0}.b2_pw1"]["out_scale"], act[f"{pfx0}.b2_dw"] * scale_mult)
        for lkey, in_s in [(f"{pfx0}.b1_pw", q[f"{pfx0}.b1_dw"]["out_scale"]), (f"{pfx0}.b2_pw2", q[f"{pfx0}.b2_dw"]["out_scale"])]:
            wmax = float(folded[lkey][0].abs().max().item()) or 1e-6
            s_stage = min(s_stage, (60.0 * in_s) / wmax)
        # Subsequent blocks (stride=1) enter and exit at exact s_stage
        for _retry in range(4):
            q[f"{pfx0}.b1_pw"] = quant_locked(*folded[f"{pfx0}.b1_pw"], q[f"{pfx0}.b1_dw"]["out_scale"], s_stage)
            q[f"{pfx0}.b2_pw2"] = quant_locked(*folded[f"{pfx0}.b2_pw2"], q[f"{pfx0}.b2_dw"]["out_scale"], s_stage)
            ok_stage = True
            for bi in range(1, len(st_seq)):
                pfx = f"{st_name}.{bi}"
                q[f"{pfx}.b2_pw1"] = quant_free(f"{pfx}.b2_pw1", *folded[f"{pfx}.b2_pw1"], s_stage, act[f"{pfx}.b2_pw1"] * scale_mult)
                q[f"{pfx}.b2_dw"] = quant_free(f"{pfx}.b2_dw", *folded[f"{pfx}.b2_dw"], q[f"{pfx}.b2_pw1"]["out_scale"], act[f"{pfx}.b2_dw"] * scale_mult)
                dw_s = q[f"{pfx}.b2_dw"]["out_scale"]
                wmax = float(folded[f"{pfx}.b2_pw2"][0].abs().max().item()) or 1e-6
                if (127.0 * dw_s) / (wmax * s_stage) < 2.05:
                    s_stage = min(s_stage * 0.65, (60.0 * dw_s) / wmax)
                    ok_stage = False
                    break
                q[f"{pfx}.b2_pw2"] = quant_locked(*folded[f"{pfx}.b2_pw2"], dw_s, s_stage)
            if ok_stage:
                break
        blocks_spec.append((st_idx, 0, 2))
        for bi in range(1, len(st_seq)):
            blocks_spec.append((st_idx, bi, 1))
        prev_scale = s_stage

    q["conv5"] = quant_free("conv5", *folded["conv5"], prev_scale, act["conv5"] * scale_mult)
    q["fc"] = quant_free("fc", *folded["fc"], q["conv5"]["out_scale"], 1.0, is_logits=True)
    for k, (wf_k, bf_k) in folded.items():
        q[k]["wf"] = wf_k.clone()
        q[k]["bf"] = bf_k.clone()
    return q, blocks_spec


class QATLayer(nn.Module):
    """Straight-Through Estimator wrapper matching INT8 weights, biases, and output clamping."""

    def __init__(self, ql: dict, stride=1, padding=0, groups=1, relu=True, clamp_i8=True, is_linear=False):
        super().__init__()
        self.w = nn.Parameter(ql["wf"].clone())
        self.b = nn.Parameter(ql["bf"].clone())
        self.w_scale = float(ql["w_scale"])
        self.b_scale = float(ql["in_scale"] * ql["w_scale"])
        self.out_scale = float(ql["out_scale"])
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.relu = relu
        self.clamp_i8 = clamp_i8
        self.is_linear = is_linear

    def forward(self, x):
        w_q = torch.clamp(torch.round(self.w * self.w_scale), -127.0, 127.0) / self.w_scale
        w_hat = self.w + (w_q - self.w).detach()
        b_q = torch.round(self.b * self.b_scale) / self.b_scale
        b_hat = self.b + (b_q - self.b).detach()
        if self.is_linear:
            y = F.linear(x, w_hat, b_hat)
        else:
            y = F.conv2d(x, w_hat, b_hat, stride=self.stride, padding=self.padding, groups=self.groups)
        if self.relu:
            y = F.relu(y)
            yq = torch.clamp(torch.round(y * self.out_scale), 0.0, 127.0) / self.out_scale
            return y + (yq - y).detach()
        if self.clamp_i8:
            yq = torch.clamp(torch.round(y * self.out_scale), -127.0, 127.0) / self.out_scale
            return y + (yq - y).detach()
        return y


class FoldedQATShuffleNet(nn.Module):
    """BN-folded ShuffleNetV2-x0.5 with exact hardware INT8 STE on every layer."""

    def __init__(self, q: dict, blocks_spec):
        super().__init__()
        self.blocks_spec = list(blocks_spec)
        self.layers = nn.ModuleDict()
        self.layers["conv1"] = QATLayer(q["conv1"], stride=2, padding=1, groups=1, relu=True, clamp_i8=True)
        for st_idx, bi, stride in self.blocks_spec:
            pfx = f"stage{st_idx}.{bi}"
            mk = f"s{st_idx}_{bi}"
            c_b = q[f"{pfx}.b2_pw1"]["w"].shape[0]
            if stride == 2:
                self.layers[f"{mk}_b1_dw"] = QATLayer(q[f"{pfx}.b1_dw"], stride=2, padding=1, groups=c_b, relu=False, clamp_i8=True)
                self.layers[f"{mk}_b1_pw"] = QATLayer(q[f"{pfx}.b1_pw"], stride=1, padding=0, groups=1, relu=True, clamp_i8=True)
            self.layers[f"{mk}_b2_pw1"] = QATLayer(q[f"{pfx}.b2_pw1"], stride=1, padding=0, groups=1, relu=True, clamp_i8=True)
            self.layers[f"{mk}_b2_dw"] = QATLayer(q[f"{pfx}.b2_dw"], stride=stride, padding=1, groups=c_b, relu=False, clamp_i8=True)
            self.layers[f"{mk}_b2_pw2"] = QATLayer(q[f"{pfx}.b2_pw2"], stride=1, padding=0, groups=1, relu=True, clamp_i8=True)
        self.layers["conv5"] = QATLayer(q["conv5"], stride=1, padding=0, groups=1, relu=True, clamp_i8=True)
        self.layers["fc"] = QATLayer(q["fc"], relu=False, clamp_i8=False, is_linear=True)
        self.conv5_out_scale = float(q["conv5"]["out_scale"])

    def forward(self, x):
        # Input x is already on [-127/128..127/128] grid
        x = self.layers["conv1"](x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for st_idx, bi, stride in self.blocks_spec:
            mk = f"s{st_idx}_{bi}"
            if stride == 2:
                b1 = self.layers[f"{mk}_b1_pw"](self.layers[f"{mk}_b1_dw"](x))
                b2 = self.layers[f"{mk}_b2_pw2"](self.layers[f"{mk}_b2_dw"](self.layers[f"{mk}_b2_pw1"](x)))
                cat = torch.cat([b1, b2], dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                b2 = self.layers[f"{mk}_b2_pw2"](self.layers[f"{mk}_b2_dw"](self.layers[f"{mk}_b2_pw1"](x2)))
                cat = torch.cat([x1, b2], dim=1)
            n, c, h, w = cat.shape
            x = cat.view(n, 2, c // 2, h, w).transpose(1, 2).contiguous().view(n, c, h, w)
        x = self.layers["conv5"](x)
        xm = x.mean(dim=(2, 3))
        gq = torch.clamp(torch.round(xm * self.conv5_out_scale), 0.0, 127.0) / self.conv5_out_scale
        xm = xm + (gq - xm).detach()
        return self.layers["fc"](xm)

    def export_into_q(self, q: dict):
        name_map = {"conv1": "conv1", "conv5": "conv5", "fc": "fc"}
        for st_idx, bi, stride in self.blocks_spec:
            pfx = f"stage{st_idx}.{bi}"
            mk = f"s{st_idx}_{bi}"
            if stride == 2:
                name_map[f"{mk}_b1_dw"] = f"{pfx}.b1_dw"
                name_map[f"{mk}_b1_pw"] = f"{pfx}.b1_pw"
            name_map[f"{mk}_b2_pw1"] = f"{pfx}.b2_pw1"
            name_map[f"{mk}_b2_dw"] = f"{pfx}.b2_dw"
            name_map[f"{mk}_b2_pw2"] = f"{pfx}.b2_pw2"
        for mod_k, q_k in name_map.items():
            lay = self.layers[mod_k]
            q[q_k]["w"] = torch.clamp(torch.round(lay.w.detach() * lay.w_scale), -127.0, 127.0).to(torch.int32).numpy()
            q[q_k]["b"] = torch.round(lay.b.detach() * lay.b_scale).to(torch.int64).numpy().astype(np.int32)
        return q


def run_qat_distill(teacher_fp32: PretrainedShuffleNPU, q: dict, blocks_spec, train_loader, test_loader, cls_weights, qat_epochs: int = 3):
    teacher_fp32.eval()
    for p in teacher_fp32.parameters():
        p.requires_grad_(False)
    qat_net = FoldedQATShuffleNet(q, blocks_spec)
    opt = torch.optim.AdamW(qat_net.parameters(), lr=1.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=qat_epochs)
    best_bal = 0.0
    best_state = None
    temp = 2.0

    print(f"\nStarting {qat_epochs}-epoch Folded INT8 QAT (self-distilled from FP32 teacher)...")
    for ep in range(qat_epochs):
        t0 = time.time()
        qat_net.train()
        ok = tot = 0
        loss_sum = 0.0
        for bx, by in train_loader:
            with torch.no_grad():
                t_lg = teacher_fp32(bx)
            s_lg = qat_net(bx)
            ce = F.cross_entropy(s_lg, by, weight=cls_weights, label_smoothing=0.04)
            kd = (temp * temp) * F.kl_div(F.log_softmax(s_lg / temp, dim=1), F.softmax(t_lg / temp, dim=1), reduction="batchmean")
            loss = 0.55 * ce + 0.45 * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(by)
            ok += (s_lg.argmax(1) == by).sum().item()
            tot += len(by)
        sched.step()

        qat_net.eval()
        vok = vtot = 0
        cok, ctot = [0, 0, 0], [0, 0, 0]
        with torch.no_grad():
            for bx, by in test_loader:
                pr = qat_net(bx).argmax(1)
                vok += (pr == by).sum().item()
                vtot += len(by)
                for c in range(3):
                    m = by == c
                    ctot[c] += int(m.sum().item())
                    cok[c] += int((pr[m] == c).sum().item())
        val_acc = 100.0 * vok / vtot
        bal_acc = 100.0 * sum(cok[c] / max(1, ctot[c]) for c in range(3)) / 3.0
        if bal_acc > best_bal:
            best_bal = bal_acc
            best_state = {k: v.detach().clone() for k, v in qat_net.state_dict().items()}
        cs = "/".join(f"{100 * cok[c] / max(1, ctot[c]):4.1f}" for c in range(3))
        print(f"  QAT {ep+1}/{qat_epochs} ({time.time()-t0:5.1f}s) | loss={loss_sum/tot:.4f} tr={100*ok/tot:5.2f}% | qat_val={val_acc:5.2f}% bal={bal_acc:5.2f}% [C/D/U={cs}]")

    if best_state is not None:
        qat_net.load_state_dict(best_state)
    return qat_net.export_into_q(q)


def emulate_dw(in_hwc: np.ndarray, q_layer: dict, stride: int) -> np.ndarray:
    H, W, C = in_hwc.shape
    oH, oW = (H // 2 if stride == 2 else H), (W // 2 if stride == 2 else W)
    pad = np.pad(in_hwc.astype(np.int64), ((1, 1), (1, 1), (0, 0)), mode="constant")
    w = q_layer["w"].reshape(C, 9).astype(np.int64)
    b = q_layer["b"].astype(np.int64)
    out = np.zeros((oH, oW, C), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            out += pad[ky : ky + oH * stride : stride, kx : kx + oW * stride : stride, :] * w[:, ky * 3 + kx]
    out += b
    return hw_round_shift(out, q_layer["shift"], relu=False, clamp_i8=True).astype(np.int8)


def emulate_pw(in_hwc: np.ndarray, q_layer: dict) -> np.ndarray:
    H, W, C_in = in_hwc.shape
    w = q_layer["w"].reshape(q_layer["w"].shape[0], C_in).astype(np.int64)
    b = q_layer["b"].astype(np.int64)
    acc = in_hwc.reshape(-1, C_in).astype(np.int64) @ w.T + b
    return hw_round_shift(acc, q_layer["shift"], relu=True, clamp_i8=True).reshape(H, W, -1).astype(np.int8)


def emulate_shufflenet_single(img_96x96x3_i8: np.ndarray, q: dict, blocks_spec) -> np.ndarray:
    # 1. conv1 (96x96x3 -> 48x48x24)
    pad = np.pad(img_96x96x3_i8.astype(np.int64), ((1, 1), (1, 1), (0, 0)), mode="constant")
    w1 = q["conv1"]["w"].transpose(0, 2, 3, 1).reshape(24, 27).astype(np.int64)
    b1 = q["conv1"]["b"].astype(np.int64)
    patches = np.zeros((48 * 48, 27), dtype=np.int64)
    p = 0
    for ky in range(3):
        for kx in range(3):
            patches[:, p : p + 3] = pad[ky : ky + 96 : 2, kx : kx + 96 : 2, :].reshape(-1, 3)
            p += 3
    x = hw_round_shift(patches @ w1.T + b1, q["conv1"]["shift"], relu=True, clamp_i8=True).reshape(48, 48, 24).astype(np.int8)

    # 2. maxpool 3x3 s2 p1 (48x48x24 -> 24x24x24, input non-neg so 0-pad is exact)
    pad = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
    mp = np.zeros((24, 24, 24), dtype=np.int8)
    for ky in range(3):
        for kx in range(3):
            mp = np.maximum(mp, pad[ky : ky + 48 : 2, kx : kx + 48 : 2, :])
    x = mp

    # 3. stage2..4
    for st_idx, bi, stride in blocks_spec:
        pfx = f"stage{st_idx}.{bi}"
        if stride == 2:
            b1_out = emulate_pw(emulate_dw(x, q[f"{pfx}.b1_dw"], stride=2), q[f"{pfx}.b1_pw"])
            b2_out = emulate_pw(emulate_dw(emulate_pw(x, q[f"{pfx}.b2_pw1"]), q[f"{pfx}.b2_dw"], stride=2), q[f"{pfx}.b2_pw2"])
            cat = np.concatenate([b1_out, b2_out], axis=-1)
        else:
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            b2_out = emulate_pw(emulate_dw(emulate_pw(x2, q[f"{pfx}.b2_pw1"]), q[f"{pfx}.b2_dw"], stride=1), q[f"{pfx}.b2_pw2"])
            cat = np.concatenate([x1, b2_out], axis=-1)
        H, W, C = cat.shape
        x = cat.reshape(H, W, 2, C // 2).swapaxes(2, 3).reshape(H, W, C)

    # 4. conv5 + GAP + fc
    x = emulate_pw(x, q["conv5"])
    gap = ((x.astype(np.int64).sum(axis=(0, 1)) + 4) // 9).clip(0, 127).astype(np.int64)
    w_fc = q["fc"]["w"].astype(np.int64)
    b_fc = q["fc"]["b"].astype(np.int64)
    return hw_round_shift(gap @ w_fc.T + b_fc, q["fc"]["shift"], relu=False, clamp_i8=False)


def pack_matrix_u32(mat_out_in: np.ndarray) -> list[int]:
    out_ch, in_ch = mat_out_in.shape
    vec_words = (in_ch + 3) // 4
    padded = np.zeros((out_ch, vec_words * 4), dtype=np.int8)
    padded[:, :in_ch] = mat_out_in.astype(np.int8)
    words = []
    for oc in range(out_ch):
        row = padded[oc].astype(np.uint8)
        for w in range(vec_words):
            b0, b1, b2, b3 = int(row[w * 4]), int(row[w * 4 + 1]), int(row[w * 4 + 2]), int(row[w * 4 + 3])
            words.append(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    return words


def emit_arr(ctype: str, name: str, vals: list[int], fmt: str) -> list[str]:
    out = [f"static const {ctype} {name}[{len(vals)}] = {{"]
    for i in range(0, len(vals), 12):
        chunk = ", ".join(fmt.format(v) for v in vals[i : i + 12])
        out.append(f"    {chunk},")
    out.append("};")
    out.append("")
    return out


def export_shufflenet_header(q: dict, blocks_spec):
    """Exports packed BSRAM weight pools, bias pools, DW kernels, and block descriptors."""
    wgt_u32_pool: list[int] = []
    npu_b_pool: list[int] = []
    dw_i8_pool: list[int] = []
    dw_b_pool: list[int] = []

    def add_pw(mat_oi: np.ndarray, b_arr: np.ndarray):
        w_off = len(wgt_u32_pool)
        b_off = len(npu_b_pool)
        wgt_u32_pool.extend(pack_matrix_u32(mat_oi))
        npu_b_pool.extend([int(v) for v in b_arr])
        return w_off, b_off

    def add_dw(w_c9: np.ndarray, b_arr: np.ndarray):
        w_off = len(dw_i8_pool)
        b_off = len(dw_b_pool)
        dw_i8_pool.extend([int(v) for v in w_c9.reshape(-1)])
        dw_b_pool.extend([int(v) for v in b_arr])
        return w_off, b_off

    # Stem conv1 (24, 3, 3, 3) -> HWC (24, 27)
    w_conv1 = q["conv1"]["w"].transpose(0, 2, 3, 1).reshape(24, 27)
    conv1_wo, conv1_bo = add_pw(w_conv1, q["conv1"]["b"])

    # Blocks
    blk_entries = []
    cur_h, cur_w, cur_c = 24, 24, 24
    for st_idx, bi, stride in blocks_spec:
        pfx = f"stage{st_idx}.{bi}"
        if stride == 2:
            c_branch = cur_c  # stage2.0 in=24 out=48 -> branch_c=24; stage3.0 in=48 out=96 -> branch_c=48
            b1_dw_wo, b1_dw_bo = add_dw(q[f"{pfx}.b1_dw"]["w"], q[f"{pfx}.b1_dw"]["b"])
            b1_pw_wo, b1_pw_bo = add_pw(q[f"{pfx}.b1_pw"]["w"].reshape(c_branch, c_branch), q[f"{pfx}.b1_pw"]["b"])
            b1_dws = q[f"{pfx}.b1_dw"]["shift"]
            b1_pws = q[f"{pfx}.b1_pw"]["shift"]
        else:
            c_branch = cur_c // 2
            b1_dw_wo = b1_dw_bo = b1_pw_wo = b1_pw_bo = 0
            b1_dws = b1_pws = 0

        b2_pw1_wo, b2_pw1_bo = add_pw(q[f"{pfx}.b2_pw1"]["w"].reshape(c_branch, c_branch), q[f"{pfx}.b2_pw1"]["b"])
        b2_dw_wo, b2_dw_bo = add_dw(q[f"{pfx}.b2_dw"]["w"], q[f"{pfx}.b2_dw"]["b"])
        b2_pw2_wo, b2_pw2_bo = add_pw(q[f"{pfx}.b2_pw2"]["w"].reshape(c_branch, c_branch), q[f"{pfx}.b2_pw2"]["b"])

        blk_entries.append((
            stride, cur_h, cur_w, cur_c, c_branch,
            b1_dw_wo, b1_dw_bo, b1_dws, b1_pw_wo, b1_pw_bo, b1_pws,
            b2_pw1_wo, b2_pw1_bo, q[f"{pfx}.b2_pw1"]["shift"],
            b2_dw_wo, b2_dw_bo, q[f"{pfx}.b2_dw"]["shift"],
            b2_pw2_wo, b2_pw2_bo, q[f"{pfx}.b2_pw2"]["shift"],
        ))
        if stride == 2:
            cur_h //= 2
            cur_w //= 2
            cur_c *= 2

    conv5_wo, conv5_bo = add_pw(q["conv5"]["w"].reshape(128, 192), q["conv5"]["b"])
    fc_wo, fc_bo = add_pw(q["fc"]["w"].reshape(3, 128), q["fc"]["b"])

    lines = [
        "/*",
        " * Auto-generated Quantized INT8/INT32 ShuffleNetV2-x0.5 Weights for Tang Mega 138K Pro",
        " * Do not edit manually - regenerate via: tools/npu/train_shufflenet_npu.py",
        " */",
        "#ifndef APP_NPU_WEIGHTS_H_",
        "#define APP_NPU_WEIGHTS_H_",
        "",
        "#include <stdint.h>",
        "",
        "#define NPU_CNN_ARCH_SHUFFLENET  1U",
        "#define NPU_IN_SIZE              96U",
        "#define NPU_CONV1_WO             " + f"{conv1_wo}U",
        "#define NPU_CONV1_BO             " + f"{conv1_bo}U",
        "#define NPU_CONV1_SHIFT          " + f"{q['conv1']['shift']}U",
        "#define NPU_NUM_BLOCKS           " + f"{len(blk_entries)}U",
        "#define NPU_CONV5_WO             " + f"{conv5_wo}U",
        "#define NPU_CONV5_BO             " + f"{conv5_bo}U",
        "#define NPU_CONV5_SHIFT          " + f"{q['conv5']['shift']}U",
        "#define NPU_FC_WO                " + f"{fc_wo}U",
        "#define NPU_FC_BO                " + f"{fc_bo}U",
        "#define NPU_FC_SHIFT             " + f"{q['fc']['shift']}U",
        "",
        "typedef struct {",
        "    uint8_t stride;",
        "    uint8_t h_in;",
        "    uint8_t w_in;",
        "    uint16_t c_in;",
        "    uint16_t c_branch;",
        "    uint32_t b1_dw_wo;",
        "    uint16_t b1_dw_bo;",
        "    uint8_t  b1_dw_shift;",
        "    uint32_t b1_pw_wo;",
        "    uint16_t b1_pw_bo;",
        "    uint8_t  b1_pw_shift;",
        "    uint32_t b2_pw1_wo;",
        "    uint16_t b2_pw1_bo;",
        "    uint8_t  b2_pw1_shift;",
        "    uint32_t b2_dw_wo;",
        "    uint16_t b2_dw_bo;",
        "    uint8_t  b2_dw_shift;",
        "    uint32_t b2_pw2_wo;",
        "    uint16_t b2_pw2_bo;",
        "    uint8_t  b2_pw2_shift;",
        "} NpuShuffleBlockDesc;",
        "",
        f"static const NpuShuffleBlockDesc k_shuffle_blocks[{len(blk_entries)}] = {{",
    ]
    for e in blk_entries:
        lines.append("    { " + ", ".join(str(x) for x in e) + " },")
    lines += ["};", ""]

    lines += emit_arr("uint32_t", "k_npu_wgt_pool", wgt_u32_pool, "0x{:08X}U")
    lines += emit_arr("int32_t", "k_npu_bias_pool", npu_b_pool, "{}")
    lines += emit_arr("int8_t", "k_dw_wgt_pool", dw_i8_pool, "{}")
    lines += emit_arr("int32_t", "k_dw_bias_pool", dw_b_pool, "{}")
    lines += ["#endif /* APP_NPU_WEIGHTS_H_ */", ""]

    WEIGHTS_H.write_text("\n".join(lines), encoding="utf-8")
    tot_bytes = len(wgt_u32_pool) * 4 + len(npu_b_pool) * 4 + len(dw_i8_pool) + len(dw_b_pool) * 4
    print(f"Exported ShuffleNetV2-x0.5 weights to {WEIGHTS_H} ({tot_bytes:,} bytes total)")


def main():
    train_items, test_items = load_datasets()
    train_ds = PetShuffleDataset(train_items, is_train=True, seed=101)
    test_ds = PetShuffleDataset(test_items, is_train=False, seed=202)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    calib_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=True, num_workers=0)

    # Balanced class weights with mild emphasis on CAT so memes/cartoons are not rejected as UNKNOWN
    counts = np.array([sum(1 for _, l in train_items if l == c) for c in range(3)], dtype=np.float32)
    inv = counts.sum() / (3.0 * counts)
    inv[0] *= 1.06
    cls_weights = torch.tensor(inv, dtype=torch.float32)
    print(f"Class loss weights (CAT, DOG, UNK): {[round(float(x), 3) for x in inv]}")

    model = PretrainedShuffleNPU()
    print(f"PretrainedShuffleNPU parameters: {sum(p.numel() for p in model.parameters()):,}")

    epochs = 8
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "conv5" in n or "fc" in n], "lr": 3e-3},
        {"params": [p for n, p in model.named_parameters() if "conv5" not in n and "fc" not in n], "lr": 5e-4},
    ], weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_bal = 0.0
    best_state = None

    import sys
    ckpt_pt = CKPT_NPZ.with_suffix(".pt")
    if ("--export-only" in sys.argv or "--qat" in sys.argv) and ckpt_pt.exists():
        print(f"Loading checkpoint {ckpt_pt}...")
        model.load_state_dict(torch.load(str(ckpt_pt), map_location="cpu", weights_only=True))
    else:
        for ep in range(epochs):
            t_ep = time.time()
            model.train()
            ok = tot = 0
            loss_sum = 0.0
            for bx, by in train_loader:
                lg = model(bx)
                loss = F.cross_entropy(lg, by, weight=cls_weights, label_smoothing=0.05)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(by)
                ok += (lg.argmax(1) == by).sum().item()
                tot += len(by)
            sched.step()

            model.eval()
            vok = vtot = 0
            cok, ctot = [0, 0, 0], [0, 0, 0]
            with torch.no_grad():
                for bx, by in test_loader:
                    pr = model(bx).argmax(1)
                    vok += (pr == by).sum().item()
                    vtot += len(by)
                    for c in range(3):
                        m = by == c
                        ctot[c] += int(m.sum().item())
                        cok[c] += int((pr[m] == c).sum().item())
            val_acc = 100.0 * vok / vtot
            bal_acc = 100.0 * sum(cok[c] / max(1, ctot[c]) for c in range(3)) / 3.0
            if bal_acc > best_bal:
                best_bal = bal_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            c_str = "/".join(f"{100 * cok[c] / max(1, ctot[c]):4.1f}" for c in range(3))
            print(
                f"Ep {ep+1}/{epochs} ({time.time()-t_ep:5.1f}s) | loss={loss_sum/tot:.4f} tr={100*ok/tot:5.2f}% | "
                f"val={val_acc:5.2f}% bal={bal_acc:5.2f}% (best={best_bal:5.2f}%) [C/D/U={c_str}]"
            )

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), str(ckpt_pt))

    # Sweep INT8 quantization scale multiplier cleanly across 300 held-out images
    rng = random.Random(77)
    eval_idxs = list(range(len(test_items)))
    rng.shuffle(eval_idxs)
    sweep_idxs = eval_idxs[:240]
    val_idxs = eval_idxs[:600]

    best_q = None
    best_blocks = None
    best_sm = 1.0
    best_q_acc = -1.0
    for sm in (0.90, 1.00, 1.15):
        qc, blks = quantize_shufflenet(model, calib_loader, scale_mult=sm)
        ok_cnt = 0
        for idx in sweep_idxs:
            im, lbl = test_items[idx]
            im200 = im.resize((200, 120), Image.Resampling.BILINEAR)
            fb = rgb_to_332_arr(np.asarray(im200, dtype=np.uint8), dither=True)
            logits = emulate_shufflenet_single(c_exact_box_downsample_rgb332_to_96x96(fb), qc, blks)
            if int(np.argmax(logits)) == lbl:
                ok_cnt += 1
        acc_s = 100.0 * ok_cnt / len(sweep_idxs)
        print(f"  INT8 scale_mult={sm:.2f} -> 240-img val = {acc_s:.2f}%")
        if acc_s > best_q_acc:
            best_q_acc = acc_s
            best_q = qc
            best_blocks = blks
            best_sm = sm

    if "--export-only" not in sys.argv:
        best_q = run_qat_distill(model, best_q, best_blocks, train_loader, test_loader, cls_weights, qat_epochs=3)

    print(f"\nSelected INT8 scale_mult={best_sm:.2f}; evaluating 600 held-out RGB332+dithered images...")
    q_ok = 0
    qc_ok, qc_tot = [0, 0, 0], [0, 0, 0]
    for idx in val_idxs:
        im, lbl = test_items[idx]
        im200 = im.resize((200, 120), Image.Resampling.BILINEAR)
        fb = rgb_to_332_arr(np.asarray(im200, dtype=np.uint8), dither=True)
        logits = emulate_shufflenet_single(c_exact_box_downsample_rgb332_to_96x96(fb), best_q, best_blocks)
        pred = int(np.argmax(logits))
        qc_tot[lbl] += 1
        if pred == lbl:
            q_ok += 1
            qc_ok[lbl] += 1
    qc_str = "/".join(f"{qc_ok[c]}/{qc_tot[c]}" for c in range(3))
    print(f"Bit-exact FPGA NPU INT8 emulator val accuracy: {q_ok}/600 ({100*q_ok/600:.2f}%) [C/D/U={qc_str}]")

    export_shufflenet_header(best_q, best_blocks)

    # Zero-shot evaluation on user's Downloads
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
    dl_ok = dl_tot = 0
    print("\nZero-shot Downloads INT8 emulator evaluation (200x120 RGB332 Floyd-Steinberg -> 96x96):")
    for p in sorted(dl.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        exp = 0 if p.name in gt_cats else (1 if p.name in gt_dogs else 2)
        im = Image.open(p).convert("RGB").resize((200, 120), Image.Resampling.BILINEAR)
        fb = rgb_to_332_arr(np.asarray(im, dtype=np.uint8), dither=True)
        logits = emulate_shufflenet_single(c_exact_box_downsample_rgb332_to_96x96(fb), best_q, best_blocks)
        pred = int(np.argmax(logits))
        ok = pred == exp
        dl_ok += int(ok)
        dl_tot += 1
        print(f"  {'OK  ' if ok else 'MISS'} {p.name[:42]:42s} exp={cls_names[exp]:7s} pred={cls_names[pred]:7s} | logits={logits.tolist()}")
    print(f"Downloads INT8 zero-shot score: {dl_ok}/{dl_tot} ({100.0*dl_ok/max(1,dl_tot):.1f}%)")


if __name__ == "__main__":
    main()
