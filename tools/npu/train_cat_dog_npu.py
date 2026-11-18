#!/usr/bin/env python3
"""
Deterministic training & quantization script for the Tang Mega 138K Pro FPGA NPU
3-class image classifier: CAT (0), DOG (1), UNKNOWN (2).

Architecture executed on FPGA `npu_core` (SIMD-4 INT8xINT8 DSP MAC @ 100 MHz):
  - Input feature map: 4 channels x 8x8 spatial grid = 256 signed INT8 features
      ch0 (0..63):   Mean Luma (centered) in each 25x15 cell of 200x120 RGB332 frame
      ch1 (64..127): Mean |dI/dx| vertical edge energy in each 25x15 cell
      ch2 (128..191):Mean |dI/dy| horizontal edge energy in each 25x15 cell
      ch3 (192..255):Chroma / diagonal structure (R-B and G-B pooled)
  - Layer 1 (NPU Hardware): 256 -> 32 (INT8 weights, INT32 bias, arithmetic >>> 7, HW ReLU, clamp 0..127)
  - Layer 2 (NPU Hardware): 32  -> 3  (INT8 weights, INT32 bias, arithmetic >>> 5, linear logits)

Also generates ready-to-upload test images in `tools/npu/samples/` and exports
`firmware/src/app/npu_weights.h`.
"""

import math
import pathlib
import random
import struct
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = ROOT / "tools" / "npu" / "samples"
WEIGHTS_H = ROOT / "firmware" / "src" / "app" / "npu_weights.h"

FB_W = 200
FB_H = 120


def rgb_to_332(r: int, g: int, b: int) -> int:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return (r & 0xE0) | ((g >> 3) & 0x1C) | (b >> 6)


def rgb332_to_rgb8(c: int):
    r3 = (c >> 5) & 0x07
    g3 = (c >> 2) & 0x07
    b2 = c & 0x03
    r = (r3 << 5) | (r3 << 2) | (r3 >> 1)
    g = (g3 << 5) | (g3 << 2) | (g3 >> 1)
    b = (b2 << 6) | (b2 << 4) | (b2 << 2) | b2
    return r, g, b


def write_png_rgb(path: pathlib.Path, width: int, height: int, rgb332_buf: bytes):
    """Minimal dependency-free PNG writer from a 200x120 RGB332 buffer."""
    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)  # filter type 0 (None)
        for x in range(width):
            r, g, b = rgb332_to_rgb8(rgb332_buf[y * width + x])
            raw_rows.extend((r, g, b))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), 9)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


def extract_features_c_exact(fb: bytes) -> list[int]:
    """
    Bit-exact match of C firmware `npu_extract_features_int8(const uint8_t *fb_200x120, int8_t out_feat[256])`.
    """
    luma = [0] * (FB_W * FB_H)
    rb_diff = [0] * (FB_W * FB_H)
    for i in range(FB_W * FB_H):
        c = fb[i]
        r3 = (c >> 5) & 7
        g3 = (c >> 2) & 7
        b2 = c & 3
        # Luma in 0..255 range: r3*9 + g3*18 + b2*12 (max = 63 + 126 + 36 = 225)
        luma[i] = r3 * 9 + g3 * 18 + b2 * 12
        rb_diff[i] = (r3 * 16) - (b2 * 32)

    feat = [0] * 256
    for gy in range(8):
        y0 = gy * 15
        for gx in range(8):
            x0 = gx * 25
            cell_idx = gy * 8 + gx
            sum_l = 0
            sum_dx = 0
            sum_dy = 0
            sum_rb = 0
            for dy in range(15):
                y = y0 + dy
                row_off = y * FB_W
                y_next = (y + 1) if (y + 1 < FB_H) else y
                row_next = y_next * FB_W
                for dx in range(25):
                    x = x0 + dx
                    x_next = (x + 1) if (x + 1 < FB_W) else x
                    v = luma[row_off + x]
                    vx = luma[row_off + x_next]
                    vy = luma[row_next + x]
                    sum_l += v
                    sum_dx += abs(vx - v)
                    sum_dy += abs(vy - v)
                    sum_rb += rb_diff[row_off + x]

            # 375 pixels per cell
            mean_l = (sum_l // 375) - 112
            mean_dx = (sum_dx * 3) // 375 - 32
            mean_dy = (sum_dy * 3) // 375 - 32
            mean_rb = sum_rb // 375

            feat[0 + cell_idx] = max(-127, min(127, mean_l))
            feat[64 + cell_idx] = max(-127, min(127, mean_dx))
            feat[128 + cell_idx] = max(-127, min(127, mean_dy))
            feat[192 + cell_idx] = max(-127, min(127, mean_rb))

    return feat


def draw_cat(variant: int = 0, dx_off: int = 0, dy_off: int = 0, noise: int = 0, rng=None) -> bytes:
    """Renders a recognizable 200x120 Cat portrait (pointed triangular ears, round head, whiskers, slit eyes)."""
    if rng is None:
        rng = random.Random(variant)
    buf = bytearray(FB_W * FB_H)

    # Background: soft indoor blue/slate or dark teal
    bg_cols = [(24, 40, 72), (32, 48, 64), (20, 36, 56), (40, 32, 60)]
    fur_cols = [
        (210, 140, 60),   # ginger tabby
        (90, 95, 110),    # Russian blue / grey cat
        (230, 175, 95),   # cream/orange cat
        (160, 115, 75),   # brown tabby
    ]
    bg_r, bg_g, bg_b = bg_cols[variant % len(bg_cols)]
    fr, fg, fb_c = fur_cols[variant % len(fur_cols)]

    cx = 100 + dx_off
    cy = 68 + dy_off

    for y in range(FB_H):
        for x in range(FB_W):
            r, g, b = bg_r, bg_g, bg_b

            # Cat body/shoulders at bottom
            if y > cy + 22 and abs(x - cx) < 38 + (y - (cy + 22)) // 2:
                r, g, b = int(fr * 0.85), int(fg * 0.85), int(fb_c * 0.85)

            # Pointed triangular ears at top-left and top-right of head!
            # Left ear: apex at (cx - 34, cy - 48), base at y = cy - 16
            if cy - 48 <= y <= cy - 14:
                t = (y - (cy - 48)) / 34.0  # 0 at tip, 1 at base
                half_w = 2.0 + t * 16.0
                ear_lx = (cx - 34) + t * 6.0
                ear_rx = (cx + 34) - t * 6.0
                if abs(x - ear_lx) <= half_w or abs(x - ear_rx) <= half_w:
                    r, g, b = fr, fg, fb_c
                    # Pink inner ear triangle
                    if t > 0.25 and (abs(x - ear_lx) <= half_w * 0.55 or abs(x - ear_rx) <= half_w * 0.55):
                        r, g, b = 235, 135, 155

            # Round cat head (slightly wider than tall)
            hx = (x - cx) / 38.0
            hy = (y - cy) / 31.0
            if hx * hx + hy * hy <= 1.0:
                r, g, b = fr, fg, fb_c
                # Tabby forehead stripes ('M' / 3 vertical bars at top of head)
                if cy - 26 <= y <= cy - 10 and abs(x - cx) <= 14:
                    if abs(x - cx) <= 2 or abs(abs(x - cx) - 9) <= 2:
                        r, g, b = int(fr * 0.45), int(fg * 0.45), int(fb_c * 0.45)

            # Almond green/gold cat eyes with vertical slit pupils
            for ex_sign in (-1, 1):
                ex = cx + ex_sign * 16
                ey = cy - 5
                edx = (x - ex) / 8.5
                edy = (y - ey) / 5.5
                if edx * edx + edy * edy <= 1.0:
                    r, g, b = 90, 230, 85  # bright emerald cat eye
                    if abs(x - ex) <= 2 and abs(y - ey) <= 4:
                        r, g, b = 15, 15, 20  # vertical slit pupil

            # Small pink inverted-triangle nose
            if cy + 4 <= y <= cy + 10:
                nt = (cy + 10 - y) / 6.0
                if abs(x - cx) <= 1 + nt * 5.0:
                    r, g, b = 245, 125, 145

            # White muzzle pads
            if ((x - (cx - 6)) / 8.0) ** 2 + ((y - (cy + 13)) / 6.0) ** 2 <= 1.0 or (
                ((x - (cx + 6)) / 8.0) ** 2 + ((y - (cy + 13)) / 6.0) ** 2 <= 1.0
            ):
                r, g, b = 240, 235, 225

            # Prominent horizontal/fanned cat whiskers extending wide left and right
            for wy_off, slope in ((-1, -0.10), (3, 0.0), (7, 0.10)):
                if 14 <= abs(x - cx) <= 58:
                    wy = int(cy + 10 + wy_off + (abs(x - cx) - 14) * slope)
                    if abs(y - wy) <= 1:
                        r, g, b = 250, 250, 250

            if noise > 0:
                r += rng.randint(-noise, noise)
                g += rng.randint(-noise, noise)
                b += rng.randint(-noise, noise)

            buf[y * FB_W + x] = rgb_to_332(r, g, b)

    return bytes(buf)


def draw_dog(variant: int = 0, dx_off: int = 0, dy_off: int = 0, noise: int = 0, rng=None) -> bytes:
    """Renders a recognizable 200x120 Dog portrait (hanging floppy ears on sides, long snout, big black nose, tongue)."""
    if rng is None:
        rng = random.Random(100 + variant)
    buf = bytearray(FB_W * FB_H)

    # Background: warm meadow green / sky
    bg_cols = [(35, 95, 45), (45, 110, 55), (30, 80, 50), (55, 100, 40)]
    coat_cols = [
        (195, 145, 70),   # Golden retriever
        (140, 90, 45),    # Chocolate/brown hound
        (215, 175, 110),  # Yellow lab
        (170, 115, 60),   # Beagle/terrier tan
    ]
    ear_cols = [
        (115, 70, 25),
        (75, 45, 20),
        (135, 90, 40),
        (95, 55, 25),
    ]
    bg_r, bg_g, bg_b = bg_cols[variant % len(bg_cols)]
    cr, cg, cb = coat_cols[variant % len(coat_cols)]
    er, eg, eb = ear_cols[variant % len(ear_cols)]

    cx = 100 + dx_off
    cy = 56 + dy_off

    for y in range(FB_H):
        for x in range(FB_W):
            r, g, b = bg_r, bg_g, bg_b

            # Dog neck/chest + red collar at bottom
            if y >= cy + 32 and abs(x - cx) <= 34:
                r, g, b = cr, cg, cb
                if abs(x - cx) <= 16 and y >= cy + 36:
                    r, g, b = 235, 230, 220  # white chest patch
                if cy + 32 <= y <= cy + 37 and abs(x - cx) <= 30:
                    r, g, b = 220, 35, 35  # red dog collar
                    if abs(x - cx) <= 4:
                        r, g, b = 250, 215, 40  # gold collar tag

            # Hanging floppy ears on left and right sides (extending down from cy-12 to cy+34)
            for esign in (-1, 1):
                ex_center = cx + esign * 42
                ey_center = cy + 12
                edx = (x - ex_center) / 14.0
                edy = (y - ey_center) / 26.0
                if edx * edx + edy * edy <= 1.0:
                    r, g, b = er, eg, eb

            # Rounded top skull (no pointed top ears!)
            hx = (x - cx) / 34.0
            hy = (y - cy) / 28.0
            if hx * hx + hy * hy <= 1.0:
                r, g, b = cr, cg, cb

            # Prominent elongated lower snout/muzzle protruding downward
            sx = (x - cx) / 22.0
            sy = (y - (cy + 18)) / 18.0
            if sx * sx + sy * sy <= 1.0:
                r, g, b = min(255, cr + 35), min(255, cg + 35), min(255, cb + 35)

            # Round dark brown/black puppy eyes above snout
            for esign in (-1, 1):
                ex = cx + esign * 14
                ey = cy - 6
                if (x - ex) ** 2 + (y - ey) ** 2 <= 30:
                    r, g, b = 25, 20, 18
                    if (x - (ex - 2)) ** 2 + (y - (ey - 2)) ** 2 <= 4:
                        r, g, b = 250, 250, 250  # eye catchlight

            # Large oval black dog nose in center of snout
            nx = (x - cx) / 10.5
            ny = (y - (cy + 11)) / 7.0
            if nx * nx + ny * ny <= 1.0:
                r, g, b = 20, 20, 25
                if -2 <= y - (cy + 9) <= 0 and abs(x - cx) <= 5:
                    r, g, b = 90, 90, 105  # nose shine

            # Pink dog tongue hanging below snout
            tx = (x - cx) / 7.5
            ty = (y - (cy + 28)) / 8.5
            if y >= cy + 22 and tx * tx + ty * ty <= 1.0:
                r, g, b = 245, 95, 115

            if noise > 0:
                r += rng.randint(-noise, noise)
                g += rng.randint(-noise, noise)
                b += rng.randint(-noise, noise)

            buf[y * FB_W + x] = rgb_to_332(r, g, b)

    return bytes(buf)


def draw_unknown(kind: int = 0, seed: int = 0) -> bytes:
    """Renders diverse 'something unknown' images: noise, geometric shapes, vehicles/buildings, plasma, checkerboards."""
    rng = random.Random(500 + kind * 97 + seed)
    buf = bytearray(FB_W * FB_H)
    mode = kind % 6

    for y in range(FB_H):
        for x in range(FB_W):
            if mode == 0:
                # Random color television static / unknown noise
                r = rng.randint(0, 255)
                g = rng.randint(0, 255)
                b = rng.randint(0, 255)
            elif mode == 1:
                # Built-in Rainbow Gradient (default pattern in TangMega138K-LCD-Studio)
                r = (x * 255) // FB_W
                g = (y * 255) // FB_H
                b = 255 - (x * 255) // FB_W
            elif mode == 2:
                # Checkerboard / calibration grid
                cell = ((x // 20) + (y // 20) + seed) & 1
                r = 240 if cell else 20
                g = 220 if cell else 30
                b = 40 if cell else 200
            elif mode == 3:
                # Abstract geometric / vehicle-like horizontal box + wheels on road
                if y > 88:
                    r, g, b = 50, 50, 55  # road
                else:
                    r, g, b = 15, 25, 45  # night sky
                if 52 <= y <= 84 and 35 <= x <= 165:
                    r, g, b = 220, 40, 30
                if 34 <= y <= 52 and 65 <= x <= 135:
                    r, g, b = 140, 210, 245
                if (x - 65) ** 2 + (y - 84) ** 2 <= 144 or (x - 135) ** 2 + (y - 84) ** 2 <= 144:
                    r, g, b = 20, 20, 20
            elif mode == 4:
                # Concentric neon rings / target
                dist = int(math.hypot(x - 100, y - 60))
                ring = ((dist + seed * 5) // 10) % 3
                r = 255 if ring == 0 else 20
                g = 255 if ring == 1 else 20
                b = 255 if ring == 2 else 20
            else:
                # Diagonal stripes / industrial barcode
                stripe = ((x * 2 + y * 3 + seed * 7) // 16) & 1
                r = 230 if stripe else 15
                g = 90 if stripe else 180
                b = 210 if stripe else 30

            buf[y * FB_W + x] = rgb_to_332(r, g, b)

    return bytes(buf)


def npu_hw_dot_emulate(act_i8: list[int], wgt_i8: list[int], bias_i32: int, shift_bits: int, relu_en: bool) -> int:
    """Bit-exact emulation of `npu_core.v` SIMD-4 INT8xINT8 DSP pipeline."""
    acc = int(bias_i32)
    for a, w in zip(act_i8, wgt_i8):
        acc += int(a) * int(w)
    shifted = acc >> shift_bits
    if relu_en and shifted < 0:
        shifted = 0
    return shifted


def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate canonical sample images for user testing & verification
    samples = {
        "cat_ginger.png": (0, draw_cat(variant=0, dx_off=0, dy_off=0, noise=4)),
        "cat_grey.png": (0, draw_cat(variant=1, dx_off=-4, dy_off=2, noise=4)),
        "dog_golden.png": (1, draw_dog(variant=0, dx_off=0, dy_off=0, noise=4)),
        "dog_brown.png": (1, draw_dog(variant=1, dx_off=4, dy_off=-2, noise=4)),
        "unknown_noise.png": (2, draw_unknown(kind=0, seed=1)),
        "unknown_rainbow.png": (2, draw_unknown(kind=1, seed=0)),
        "unknown_car.png": (2, draw_unknown(kind=3, seed=2)),
        "unknown_rings.png": (2, draw_unknown(kind=4, seed=3)),
    }
    for name, (_, buf) in samples.items():
        write_png_rgb(SAMPLES_DIR / name, FB_W, FB_H, buf)
        (SAMPLES_DIR / name.replace(".png", ".rgb332")).write_bytes(buf)

    # 2. Build training dataset of feature vectors (256 INT8 features each)
    dataset = []
    rng = random.Random(42)
    for v in range(4):
        for dx in (-8, -4, 0, 4, 8):
            for dy in (-4, 0, 4):
                img_cat = draw_cat(variant=v, dx_off=dx, dy_off=dy, noise=8, rng=rng)
                dataset.append((0, extract_features_c_exact(img_cat)))
                img_dog = draw_dog(variant=v, dx_off=dx, dy_off=dy, noise=8, rng=rng)
                dataset.append((1, extract_features_c_exact(img_dog)))

    for k in range(6):
        for s in range(10):
            img_unk = draw_unknown(kind=k, seed=s)
            dataset.append((2, extract_features_c_exact(img_unk)))

    # 3. Quantization-Aware Training (QAT) of 2-layer network (256 -> 32 -> 3)
    # Forward pass uses exact integer NPU emulation (STE for gradients)
    train_rng = random.Random(12345)
    w1_f = [[train_rng.uniform(-18.0, 18.0) for _ in range(256)] for _ in range(32)]
    b1_f = [0.0 for _ in range(32)]
    w2_f = [[train_rng.uniform(-24.0, 24.0) for _ in range(32)] for _ in range(3)]
    b2_f = [0.0, 0.0, 4096.0]

    def quant_w1():
        return [[max(-127, min(127, int(round(x)))) for x in row] for row in w1_f]

    def quant_b1():
        return [int(round(b)) for b in b1_f]

    def quant_w2():
        return [[max(-127, min(127, int(round(x)))) for x in row] for row in w2_f]

    def quant_b2():
        return [int(round(b)) for b in b2_f]

    indices = list(range(len(dataset)))
    for epoch in range(60):
        train_rng.shuffle(indices)
        lr1 = 0.03 * (0.96 ** epoch)
        lr2 = 0.18 * (0.96 ** epoch)
        w1_q = quant_w1()
        b1_q = quant_b1()
        w2_q = quant_w2()
        b2_q = quant_b2()
        for idx in indices:
            lbl, feat = dataset[idx]

            h_out = []
            h_pre = []
            for h in range(32):
                acc = b1_q[h] + sum(feat[i] * w1_q[h][i] for i in range(256))
                s = acc >> 8
                h_pre.append(s)
                h_out.append(max(0, min(127, s)))

            logits = []
            for c in range(3):
                acc = b2_q[c] + sum(h_out[h] * w2_q[c][h] for h in range(32))
                logits.append(acc >> 4)

            # Softmax with temperature 256.0
            m_log = max(logits)
            exps = [math.exp(max(-40.0, min(40.0, (l - m_log) / 256.0))) for l in logits]
            s_exp = sum(exps)
            probs = [e / s_exp for e in exps]

            d_logits = [probs[c] - (1.0 if c == lbl else 0.0) for c in range(3)]
            if abs(d_logits[lbl]) < 0.02:
                continue

            # Backprop to Layer 1 hidden activations
            d_h = [0.0] * 32
            for h in range(32):
                grad_sum = sum(d_logits[c] * w2_f[c][h] for c in range(3))
                if 0 < h_pre[h] < 127 or (h_pre[h] <= 0 and grad_sum < 0):
                    d_h[h] = grad_sum / 16.0

            # Update Layer 2
            for c in range(3):
                g_c = d_logits[c]
                b2_f[c] -= lr2 * 512.0 * g_c
                b2_q[c] = int(round(b2_f[c]))
                for h in range(32):
                    w2_f[c][h] = max(-127.0, min(127.0, w2_f[c][h] - lr2 * g_c * h_out[h]))
                    w2_q[c][h] = int(round(w2_f[c][h]))

            # Update Layer 1
            for h in range(32):
                g_h = d_h[h]
                if abs(g_h) < 1e-4:
                    continue
                b1_f[h] -= lr1 * 1024.0 * g_h
                b1_q[h] = int(round(b1_f[h]))
                row_f = w1_f[h]
                row_q = w1_q[h]
                for i in range(256):
                    v = row_f[i] - lr1 * g_h * feat[i]
                    if v > 127.0:
                        v = 127.0
                    elif v < -127.0:
                        v = -127.0
                    row_f[i] = v
                    row_q[i] = int(round(v))

    w1_i8 = quant_w1()
    b1_i32 = quant_b1()
    w2_i8 = quant_w2()
    b2_i32 = quant_b2()

    def infer_npu_bit_exact(feat256: list[int]):
        h_out = []
        for h in range(32):
            v = npu_hw_dot_emulate(feat256, w1_i8[h], b1_i32[h], shift_bits=8, relu_en=True)
            h_out.append(max(0, min(127, v)))
        logits = []
        for c in range(3):
            v = npu_hw_dot_emulate(h_out, w2_i8[c], b2_i32[c], shift_bits=4, relu_en=False)
            logits.append(v)
        return logits, h_out

    # Verify dataset accuracy
    correct = 0
    for lbl, feat in dataset:
        logits, _ = infer_npu_bit_exact(feat)
        pred = logits.index(max(logits))
        if pred == lbl:
            correct += 1
    acc_pct = 100.0 * correct / len(dataset)
    print(f"Dataset NPU bit-exact accuracy: {correct}/{len(dataset)} ({acc_pct:.2f}%)")

    for name, (lbl, buf) in samples.items():
        feat = extract_features_c_exact(buf)
        logits, _ = infer_npu_bit_exact(feat)
        pred = logits.index(max(logits))
        cls_names = ["CAT", "DOG", "UNKNOWN"]
        print(f"  Sample {name:22s} -> expected={cls_names[lbl]:7s} pred={cls_names[pred]:7s} logits={logits}")
        assert pred == lbl, f"Mismatch on {name}"

    # 4. Export C header `firmware/src/app/npu_weights.h` with pre-packed uint32_t words for `WGT_DATA4_RAW`
    def pack_i8_words(vec_i8: list[int]) -> list[int]:
        words = []
        for i in range(0, len(vec_i8), 4):
            b0 = vec_i8[i + 0] & 0xFF
            b1 = vec_i8[i + 1] & 0xFF
            b2 = vec_i8[i + 2] & 0xFF
            b3 = vec_i8[i + 3] & 0xFF
            words.append(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
        return words

    lines = [
        "/* AUTO-GENERATED BY tools/npu/train_cat_dog_npu.py -- DO NOT EDIT */",
        "#ifndef APP_NPU_WEIGHTS_H_",
        "#define APP_NPU_WEIGHTS_H_",
        "",
        "#include <stdint.h>",
        "",
        "#define NPU_L1_IN_DIM       256U",
        "#define NPU_L1_WORDS        64U",
        "#define NPU_L1_OUT_DIM      32U",
        "#define NPU_L1_SHIFT        8U",
        "",
        "#define NPU_L2_IN_DIM       32U",
        "#define NPU_L2_WORDS        8U",
        "#define NPU_L2_OUT_DIM      3U",
        "#define NPU_L2_SHIFT        4U",
        "",
        "static const int32_t k_npu_l1_bias[NPU_L1_OUT_DIM] = {",
        "    " + ", ".join(str(b) for b in b1_i32),
        "};",
        "",
        "static const uint32_t k_npu_l1_weights_packed[NPU_L1_OUT_DIM][NPU_L1_WORDS] = {",
    ]
    for h in range(32):
        pw = pack_i8_words(w1_i8[h])
        lines.append("    { " + ", ".join(f"0x{w:08X}UL" for w in pw) + " },")
    lines += [
        "};",
        "",
        "static const int32_t k_npu_l2_bias[NPU_L2_OUT_DIM] = {",
        "    " + ", ".join(str(b) for b in b2_i32),
        "};",
        "",
        "static const uint32_t k_npu_l2_weights_packed[NPU_L2_OUT_DIM][NPU_L2_WORDS] = {",
    ]
    for c in range(3):
        pw = pack_i8_words(w2_i8[c])
        lines.append("    { " + ", ".join(f"0x{w:08X}UL" for w in pw) + " },")
    lines += [
        "};",
        "",
        "#endif /* APP_NPU_WEIGHTS_H_ */",
        "",
    ]
    WEIGHTS_H.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {WEIGHTS_H} and {len(samples)} sample images in {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
