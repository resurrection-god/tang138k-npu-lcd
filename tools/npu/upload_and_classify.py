#!/usr/bin/env python3
"""
Ethernet Image Uploader & Hardware FPGA NPU Classifier Client for Tang Mega 138K Pro Dock.

Uploads 200x120 RGB332 images over UDP port 5555 ("LCD1" protocol, compatible with
TangMega138K-LCD-Studio), displays the image + HUD banner on the 800x480 RGB666 LCD,
and queries the Gowin SIMD-4 INT8 DSP NPU Core (0xF8000900) for classification telemetry.

Usage:
  # Run automated demo across Cat, Dog, and Unknown samples:
  python tools/npu/upload_and_classify.py --host 10.0.0.222 --demo

  # Upload a specific image file (.png, .rgb332, or any image via Pillow if installed):
  python tools/npu/upload_and_classify.py --host 10.0.0.222 --image tools/npu/samples/cat_ginger.png
"""

import argparse
import pathlib
import socket
import struct
import sys
import time
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = ROOT / "tools" / "npu" / "samples"

FB_W = 200
FB_H = 120
FB_SIZE = FB_W * FB_H
UDP_PORT = 5555

CLASS_NAMES = {
    0: "CAT (КОШКА)",
    1: "DOG (СОБАКА)",
    2: "UNKNOWN (НЕИЗВЕСТНО)",
}


def rgb_to_332(r: int, g: int, b: int) -> int:
    return (int(r) & 0xE0) | ((int(g) >> 3) & 0x1C) | (int(b) >> 6)


def load_image_to_rgb332(path: pathlib.Path) -> bytes:
    """Loads a .rgb332 file, our uncompressed/zlib 200x120 8-bit RGB PNG, or any image via PIL."""
    if path.suffix.lower() == ".rgb332":
        data = path.read_bytes()
        if len(data) != FB_SIZE:
            raise ValueError(f"Expected {FB_SIZE} bytes in {path}, got {len(data)}")
        return data

    rgb332_sibling = path.with_suffix(".rgb332")
    if rgb332_sibling.exists() and rgb332_sibling.stat().st_size == FB_SIZE:
        return rgb332_sibling.read_bytes()

    # Try Pillow first for arbitrary user PNG/JPG/BMP/WEBP (matches Studio's Bilinear + Floyd-Steinberg RGB332)
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        im = Image.open(path).convert("RGB").resize((FB_W, FB_H), Image.Resampling.BILINEAR)
        fbuf = np.asarray(im, dtype=np.float32).copy()
        out = bytearray(FB_SIZE)
        for y in range(FB_H):
            for x in range(FB_W):
                old = np.clip(fbuf[y, x], 0.0, 255.0)
                c = (int(old[0]) & 0xE0) | ((int(old[1]) >> 3) & 0x1C) | (int(old[2]) >> 6)
                out[y * FB_W + x] = c
                r3 = (c >> 5) & 7
                g3 = (c >> 2) & 7
                b2 = c & 3
                new_rgb = np.array([
                    (r3 << 5) | (r3 << 2) | (r3 >> 1),
                    (g3 << 5) | (g3 << 2) | (g3 >> 1),
                    (b2 << 6) | (b2 << 4) | (b2 << 2) | b2,
                ], dtype=np.float32)
                err = old - new_rgb
                if x + 1 < FB_W:
                    fbuf[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < FB_H:
                    if x > 0:
                        fbuf[y + 1, x - 1] += err * (3.0 / 16.0)
                    fbuf[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < FB_W:
                        fbuf[y + 1, x + 1] += err * (1.0 / 16.0)
        return bytes(out)
    except ImportError:
        pass

    # Built-in dependency-free reader for 200x120 8-bit RGB PNG (filter 0)
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Unsupported image format for {path} (install pillow for arbitrary JPG/PNG)")
    pos = 8
    idat = bytearray()
    w = h = 0
    while pos < len(raw):
        ln = struct.unpack(">I", raw[pos : pos + 4])[0]
        tag = raw[pos + 4 : pos + 8]
        chunk_data = raw[pos + 8 : pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            w, h = struct.unpack(">II", chunk_data[:8])
        elif tag == b"IDAT":
            idat.extend(chunk_data)
        elif tag == b"IEND":
            break
    decomp = zlib.decompress(bytes(idat))
    if w != FB_W or h != FB_H:
        raise RuntimeError(f"PNG {path} is {w}x{h}, expected {FB_W}x{FB_H}")
    out = bytearray(FB_SIZE)
    stride = 1 + w * 3
    for y in range(FB_H):
        row = decomp[y * stride + 1 : (y + 1) * stride]
        for x in range(FB_W):
            r = row[x * 3 + 0]
            g = row[x * 3 + 1]
            b = row[x * 3 + 2]
            out[y * FB_W + x] = rgb_to_332(r, g, b)
    return bytes(out)


def upload_and_query(sock: socket.socket, host: str, port: int, rgb332: bytes) -> dict:
    """Streams 24 KB RGB332 image in 1000-byte UDP chunks with ACK and queries FPGA NPU telemetry."""
    chunk_size = 1000
    t0 = time.perf_counter()
    for offset in range(0, FB_SIZE, chunk_size):
        chunk = rgb332[offset : offset + chunk_size]
        is_last = (offset + len(chunk)) >= FB_SIZE
        flags = 1 if is_last else 0
        pkt = struct.pack("<4sBHHB", b"LCD1", 0x04, offset, len(chunk), flags) + chunk

        ack_ok = False
        for _ in range(5):
            sock.sendto(pkt, (host, port))
            try:
                rep, _ = sock.recvfrom(64)
                if len(rep) >= 8 and rep[:4] == b"LCD1" and rep[4] == 0x84:
                    ack_off = struct.unpack("<H", rep[5:7])[0]
                    if ack_off == offset:
                        ack_ok = True
                        break
            except socket.timeout:
                continue
        if not ack_ok:
            raise TimeoutError(f"Timeout waiting for LCD1 chunk ACK at offset {offset}")

    upload_ms = (time.perf_counter() - t0) * 1000.0

    # Query full NPU telemetry (CMD_NPU_QUERY = 0x07)
    q_pkt = struct.pack("<4sBBBB", b"LCD1", 0x07, 0, 0, 0)
    for _ in range(5):
        sock.sendto(q_pkt, (host, port))
        try:
            rep, _ = sock.recvfrom(64)
            if len(rep) >= 32 and rep[:4] == b"LCD1" and rep[4] == 0x87:
                npu_ok = rep[5]
                pred_cls = rep[6]
                conf_pct = rep[7]
                infer_cnt = struct.unpack("<H", rep[8:10])[0]
                l_cat, l_dog, l_unk, exec_ns, mac_cnt = struct.unpack("<iiiII", rep[12:32])
                return {
                    "npu_ok": bool(npu_ok),
                    "pred_class": pred_cls,
                    "pred_name": CLASS_NAMES.get(pred_cls, f"CLASS_{pred_cls}"),
                    "confidence_pct": conf_pct,
                    "infer_cnt": infer_cnt,
                    "logits": [l_cat, l_dog, l_unk],
                    "exec_time_ns": exec_ns,
                    "mac_count": mac_cnt,
                    "upload_ms": upload_ms,
                }
        except socket.timeout:
            continue
    raise TimeoutError("Timeout waiting for NPU query reply (0x87)")


def main():
    parser = argparse.ArgumentParser(description="Upload image over Ethernet to Tang Mega 138K Pro LCD & FPGA NPU")
    parser.add_argument("--host", default="10.0.0.222", help="Board IPv4 address (default: 10.0.0.222)")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="UDP port (default: 5555)")
    parser.add_argument("--image", type=str, help="Path to a specific image (.png / .rgb332) to upload")
    parser.add_argument("--demo", action="store_true", help="Run automated test across Cat, Dog, and Unknown samples")
    parser.add_argument("--downloads", type=str, help="Evaluate all static images in folder (e.g. C:\\Users\\penis\\Downloads)")
    parser.add_argument("--pause", type=float, default=1.5, help="Pause in seconds between demo images on LCD")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.25)

    if args.image:
        img_path = pathlib.Path(args.image)
        buf = load_image_to_rgb332(img_path)
        res = upload_and_query(sock, args.host, args.port, buf)
        print(
            f"[{img_path.name}] -> NPU: {res['pred_name']} ({res['confidence_pct']}%) | "
            f"logits={res['logits']} | hw_time={res['exec_time_ns']} ns ({res['mac_count']} MACs) | "
            f"udp_tx={res['upload_ms']:.1f} ms"
        )
        return 0

    if args.downloads:
        dl_dir = pathlib.Path(args.downloads)
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
        files = [p for p in sorted(dl_dir.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        print("=====================================================================================================")
        print(f" Tang Mega 138K Pro — Live Hardware Downloads Test ({args.host}:{args.port}, {len(files)} images)")
        print("=====================================================================================================")
        passed = 0
        for idx, fpath in enumerate(files):
            exp = 0 if fpath.name in gt_cats else (1 if fpath.name in gt_dogs else 2)
            buf = load_image_to_rgb332(fpath)
            res = upload_and_query(sock, args.host, args.port, buf)
            ok = (res["pred_class"] == exp) and res["npu_ok"] and (res["mac_count"] == 5419392)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(
                f" [{idx+1:2d}/{len(files)}] {fpath.name[:34]:34s} exp={CLASS_NAMES[exp][:7]:7s} -> "
                f"{res['pred_name'][:7]:7s} ({res['confidence_pct']:2d}%) | logits={str(res['logits']):20s} | "
                f"NPU={res['exec_time_ns']/1000.0:6.1f} us ({res['mac_count']} MAC) | {status}"
            )
            if idx + 1 < len(files) and args.pause > 0:
                time.sleep(args.pause)
        print("-----------------------------------------------------------------------------------------------------")
        print(f" Downloads Live Result: {passed}/{len(files)} ({100.0*passed/max(1,len(files)):.1f}%) on hardware FPGA NPU")
        print("=====================================================================================================")
        return 0 if passed == len(files) else 1

    demo_files = [
        ("cat_ginger.png", 0),
        ("cat_grey.png", 0),
        ("dog_golden.png", 1),
        ("dog_brown.png", 1),
        ("unknown_noise.png", 2),
        ("unknown_car.png", 2),
        ("unknown_rainbow.png", 2),
        ("unknown_rings.png", 2),
        ("cat_ginger.png", 0),  # leave Cat on LCD screen at the end
    ]

    print("========================================================================================")
    print(f" Tang Mega 138K Pro — Ethernet -> LCD + FPGA NPU Classifier ({args.host}:{args.port})")
    print("========================================================================================")
    passed = 0
    for idx, (fname, exp_cls) in enumerate(demo_files):
        fpath = SAMPLES_DIR / fname
        buf = load_image_to_rgb332(fpath)
        res = upload_and_query(sock, args.host, args.port, buf)
        ok = (res["pred_class"] == exp_cls) and res["npu_ok"] and (res["mac_count"] == 5419392)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f" [{idx+1}/{len(demo_files)}] {fname:20s} -> {res['pred_name']:22s} "
            f"conf={res['confidence_pct']:2d}% | logits={str(res['logits']):22s} | "
            f"NPU={res['exec_time_ns']/1000.0:6.1f} us ({res['mac_count']} MAC) | {status}"
        )
        if idx + 1 < len(demo_files) and args.pause > 0:
            time.sleep(args.pause)

    print("----------------------------------------------------------------------------------------")
    print(f" Result: {passed}/{len(demo_files)} verified on hardware FPGA NPU")
    print("========================================================================================")
    return 0 if passed == len(demo_files) else 1


if __name__ == "__main__":
    sys.exit(main())
