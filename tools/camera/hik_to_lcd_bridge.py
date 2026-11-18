#!/usr/bin/env python3
"""Stage 1 bridge: Hikrobot GigE Vision camera -> Tang Mega 138K Pro LCD (UDP :5555).

Receives Mono8 frames over GVSP from a Hikrobot GigE camera (e.g. MV-CS004-10GM),
downscales them to 200x120, applies Bayer 2x2 dithering into RGB332 (with optional
auto-contrast stretch), and streams them to lcd_hud.c on UDP port 5555.
Pure Python stdlib (no numpy dependency required).

Examples:
    python tools/camera/hik_to_lcd_bridge.py --selftest
    python tools/camera/hik_to_lcd_bridge.py --cam-ip 10.0.0.5 --board-ip 10.0.0.88
    python tools/camera/hik_to_lcd_bridge.py --board-ip 10.0.0.88 --test-pattern
"""
import argparse
import signal
import socket
import struct
import sys
import time

GVCP_PORT = 3956
LCD_UDP_PORT = 5555
LCD_W = 200
LCD_H = 120
LCD_FB_BYTES = LCD_W * LCD_H
LCD_CHUNK_BYTES = 1200  # 20 chunks = 24,000 B

# Hikrobot MV-CS004-10GM GenICam register addresses (Hikrobot_GigE_Vision_1_V4_0_35.xml)
REG_GVCP_CCP = 0x00000A00
REG_GVCP_HB_TIMEOUT = 0x00000938
REG_SC_PORT0 = 0x00000D00
REG_SC_PS0 = 0x00000D04
REG_SC_PD0 = 0x00000D08
REG_SC_DA0 = 0x00000D18
REG_WIDTH = 0x00030360
REG_HEIGHT = 0x000303A0
REG_DECIMATION = 0x00030600
REG_PIXEL_FORMAT = 0x00030610
REG_ACQ_START = 0x00030804
REG_ACQ_STOP = 0x00030808
REG_TRIGGER_MODE = 0x00030840

PIXEL_FORMAT_MONO8 = 0x01080001

# Precomputed Bayer 2x2 LUTs: [parity][gray_u8] -> rgb332 byte
_BAYER_WEIGHTS = (0, 2, 3, 1)
_DITHER_LUT = []
for _b in _BAYER_WEIGHTS:
    _row = bytearray(256)
    _d_rg = (_b - 1) * 9
    _d_bl = (_b - 1) * 21
    for _v in range(256):
        _rg = max(0, min(255, _v + _d_rg)) >> 5
        _bl = max(0, min(255, _v + _d_bl)) >> 6
        _row[_v] = (_rg << 5) | (_rg << 2) | _bl
    _DITHER_LUT.append(bytes(_row))

_NODITHER_LUT = bytes((((v >> 5) << 5) | ((v >> 5) << 2) | (v >> 6)) for v in range(256))


def _local_ip_for(target_ip: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_ip, GVCP_PORT))
        return s.getsockname()[0]
    finally:
        s.close()


class GvcpClient:
    """Minimal GigE Vision Control Protocol client."""

    def __init__(self, cam_ip: str, bind_ip: str = "", timeout: float = 1.0):
        self.cam_ip = cam_ip
        self.req_id = 1
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        if bind_ip:
            self.sock.bind((bind_ip, 0))

    def close(self):
        self.sock.close()

    def _next_id(self) -> int:
        rid = self.req_id
        self.req_id = (self.req_id + 1) & 0xFFFF
        if self.req_id == 0:
            self.req_id = 1
        return rid

    def read_reg(self, addr: int) -> int:
        rid = self._next_id()
        pkt = struct.pack(">BBHHHI", 0x42, 0x01, 0x0080, 4, rid, addr)
        self.sock.sendto(pkt, (self.cam_ip, GVCP_PORT))
        data, _ = self.sock.recvfrom(512)
        st, ack, plen, aid = struct.unpack(">HHHH", data[:8])
        if st != 0 or ack != 0x0081 or aid != rid or plen < 4:
            raise RuntimeError(f"GVCP READREG 0x{addr:08X} failed: st=0x{st:04X}")
        return struct.unpack(">I", data[8:12])[0]

    def write_reg(self, addr: int, val: int):
        rid = self._next_id()
        pkt = struct.pack(">BBHHHII", 0x42, 0x01, 0x0082, 8, rid, addr, val & 0xFFFFFFFF)
        self.sock.sendto(pkt, (self.cam_ip, GVCP_PORT))
        data, _ = self.sock.recvfrom(512)
        st, ack, _, aid = struct.unpack(">HHHH", data[:8])
        if st != 0 or ack != 0x0083 or aid != rid:
            raise RuntimeError(f"GVCP WRITEREG 0x{addr:08X}=0x{val:08X} failed: st=0x{st:04X}")


def discover_hikrobot(bind_ip: str = "", timeout: float = 1.0) -> str:
    """Broadcast GVCP discovery and return the first Hikrobot camera IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(timeout)
    if bind_ip:
        s.bind((bind_ip, 0))
    pkt = struct.pack(">BBHHH", 0x42, 0x11, 0x0002, 0, 1)
    s.sendto(pkt, ("255.255.255.255", GVCP_PORT))
    try:
        while True:
            data, addr = s.recvfrom(1024)
            if len(data) >= 8 + 248:
                payload = data[8:]
                manuf = payload[0x48:0x68].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                model = payload[0x68:0x88].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                cur_ip = socket.inet_ntoa(payload[0x24:0x28])
                if "hik" in manuf.lower() or model.startswith("MV-"):
                    return cur_ip or addr[0]
    except socket.timeout:
        pass
    finally:
        s.close()
    raise RuntimeError("No Hikrobot GigE Vision camera found on network")


def build_resampler(src_w: int, src_h: int):
    """Precompute row offsets and x indices mapping src_w x src_h -> 200x120."""
    y_rows = [((y * src_h) // LCD_H) * src_w for y in range(LCD_H)]
    x_cols = [(x * src_w) // LCD_W for x in range(LCD_W)]
    return y_rows, x_cols


def mono8_frame_to_rgb332(
    frame_buf: bytearray,
    y_rows: list,
    x_cols: list,
    auto_contrast: bool = False,
    dither: bool = True,
) -> bytes:
    """Downscale src_w x src_h Mono8 buffer to 200x120 RGB332."""
    sampled = bytearray(LCD_FB_BYTES)
    di = 0
    for row_off in y_rows:
        for xc in x_cols:
            sampled[di] = frame_buf[row_off + xc]
            di += 1

    if auto_contrast:
        lo = min(sampled)
        hi = max(sampled)
        if hi > lo + 8:
            scale = 255.0 / (hi - lo)
            stretch = bytes(max(0, min(255, int((v - lo) * scale))) for v in range(256))
            sampled = bytearray(stretch[v] for v in sampled)

    out = bytearray(LCD_FB_BYTES)
    if dither:
        lut0, lut1, lut2, lut3 = _DITHER_LUT
        for y in range(LCD_H):
            base = y * LCD_W
            la, lb = (lut2, lut3) if (y & 1) else (lut0, lut1)
            for x in range(0, LCD_W, 2):
                p = base + x
                out[p] = la[sampled[p]]
                out[p + 1] = lb[sampled[p + 1]]
    else:
        nd = _NODITHER_LUT
        for i in range(LCD_FB_BYTES):
            out[i] = nd[sampled[i]]
    return bytes(out)


def send_frame_to_lcd(sock: socket.socket, board_ip: str, fb_bytes: bytes, wait_ack: bool = True):
    """Send 24,000 B RGB332 frame to lcd_hud.c over UDP :5555."""
    total = len(fb_bytes)
    offset = 0
    while offset < total:
        chunk_len = min(LCD_CHUNK_BYTES, total - offset)
        is_last = (offset + chunk_len) >= total
        flags = 1 if is_last else 0
        pkt = struct.pack("<4sBHHB", b"LCD1", 0x04, offset, chunk_len, flags) + fb_bytes[offset:offset + chunk_len]
        sock.sendto(pkt, (board_ip, LCD_UDP_PORT))
        if wait_ack:
            try:
                sock.recvfrom(64)
            except socket.timeout:
                pass
        offset += chunk_len


def parse_gvsp_packet(pkt: bytes):
    """Parse standard (8 B) or extended (20 B) GVSP packet header.

    Returns (block_id, pkt_id, fmt, payload_bytes).
    """
    if len(pkt) < 8:
        return None
    flag_byte = pkt[4]
    extended = (flag_byte & 0x80) != 0
    fmt = flag_byte & 0x0F
    if extended:
        if len(pkt) < 20:
            return None
        _, _, _, bid, pid = struct.unpack(">HHB3xQI", pkt[:20])
        return bid, pid, fmt, pkt[20:]
    bid, w2 = struct.unpack(">HI", pkt[2:8])
    pid = w2 & 0x00FFFFFF
    return bid, pid, fmt, pkt[8:]


def run_selftest() -> int:
    """Offline sanity check for resampler, Bayer dither, and GVSP/LCD packetizers."""
    src_w, src_h = 720, 480
    row = bytes((x * 255) // (src_w - 1) for x in range(src_w))
    src = bytearray(row * src_h)
    y_rows, x_cols = build_resampler(src_w, src_h)
    fb = mono8_frame_to_rgb332(src, y_rows, x_cols, auto_contrast=True, dither=True)
    assert len(fb) == LCD_FB_BYTES, f"Bad fb size: {len(fb)}"
    assert fb[0] == 0x00 and fb[LCD_W - 1] == 0xFF, f"Bad gradient bounds: {fb[0]:#x}..{fb[LCD_W-1]:#x}"

    std_hdr = struct.pack(">HHB", 0, 42, 0x03) + struct.pack(">I", 7)[1:]
    bid, pid, fmt, body = parse_gvsp_packet(std_hdr + b"\xAA\xBB")
    assert (bid, pid, fmt, body) == (42, 7, 3, b"\xAA\xBB"), "Standard GVSP parse failed"

    ext_hdr = struct.pack(">HHB3xQI", 0, 0, 0x83, 99, 12)
    bid, pid, fmt, body = parse_gvsp_packet(ext_hdr + b"\xCC")
    assert (bid, pid, fmt, body) == (99, 12, 3, b"\xCC"), "Extended GVSP parse failed"
    print(f"[SELFTEST] PASS: {src_w}x{src_h} -> {LCD_W}x{LCD_H} RGB332 ({len(fb)} B), GVSP std+ext OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cam-ip", default="", help="Hikrobot camera IP (auto-discover if omitted)")
    ap.add_argument("--board-ip", default="10.0.0.88", help="Tang Mega board IP")
    ap.add_argument("--scps", type=int, default=1440, help="GVSP packet size in bytes")
    ap.add_argument("--decim", type=int, choices=[1, 2], default=1, help="Camera 1x1 or 2x2 decimation")
    ap.add_argument("--auto-contrast", action="store_true", help="Stretch min-max grayscale per frame")
    ap.add_argument("--no-dither", action="store_true", help="Disable Bayer 2x2 dithering")
    ap.add_argument("--no-ack", action="store_true", help="Do not wait for per-chunk UDP ACK")
    ap.add_argument("--fps-limit", type=float, default=30.0, help="Max LCD update rate (Hz)")
    ap.add_argument("--test-pattern", action="store_true", help="Stream synthetic ramp to board LCD (no camera)")
    ap.add_argument("--selftest", action="store_true", help="Run offline unit check and exit")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()

    lcd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lcd_sock.settimeout(0.05)

    if args.test_pattern:
        print(f"[BRIDGE] Streaming synthetic 200x120 test pattern to {args.board_ip}:{LCD_UDP_PORT}...")
        y_rows, x_cols = build_resampler(LCD_W, LCD_H)
        phase = 0
        period = 1.0 / max(1.0, args.fps_limit)
        try:
            while True:
                t0 = time.monotonic()
                row = bytes((x + phase) & 0xFF for x in range(LCD_W))
                frame_buf = bytearray(row * LCD_H)
                fb = mono8_frame_to_rgb332(frame_buf, y_rows, x_cols, dither=not args.no_dither)
                send_frame_to_lcd(lcd_sock, args.board_ip, fb, wait_ack=not args.no_ack)
                phase = (phase + 4) & 0xFF
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\n[BRIDGE] Stopped test pattern.")
            return 0

    host_ip = _local_ip_for(args.board_ip)
    cam_ip = args.cam_ip or discover_hikrobot(bind_ip=host_ip)
    print(f"[BRIDGE] Camera={cam_ip}, PC={host_ip}, Board={args.board_ip}:{LCD_UDP_PORT}")

    gvsp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gvsp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    gvsp_sock.bind((host_ip, 0))
    gvsp_sock.settimeout(0.2)
    gvsp_port = gvsp_sock.getsockname()[1]

    gvcp = GvcpClient(cam_ip, bind_ip=host_ip)
    running = True

    def _stop_handler(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)

    try:
        gvcp.write_reg(REG_GVCP_CCP, 2)
        gvcp.write_reg(REG_GVCP_HB_TIMEOUT, 5000)
        gvcp.write_reg(REG_ACQ_STOP, 0)
        gvcp.write_reg(REG_TRIGGER_MODE, 0)
        gvcp.write_reg(REG_PIXEL_FORMAT, PIXEL_FORMAT_MONO8)
        gvcp.write_reg(REG_DECIMATION, 0x00020002 if args.decim == 2 else 0x00010001)
        w = gvcp.read_reg(REG_WIDTH)
        h = gvcp.read_reg(REG_HEIGHT)
        old_ps = gvcp.read_reg(REG_SC_PS0)
        gvcp.write_reg(REG_SC_PS0, (old_ps & 0xFFFF0000) | (args.scps & 0xFFFF))
        gvcp.write_reg(REG_SC_PD0, 400)
        gvcp.write_reg(REG_SC_DA0, struct.unpack(">I", socket.inet_aton(host_ip))[0])
        gvcp.write_reg(REG_SC_PORT0, gvsp_port)
        gvcp.write_reg(REG_ACQ_START, 0x80000000)
        print(f"[BRIDGE] Streaming {w}x{h} Mono8 (SCPS={args.scps}, GVSP port={gvsp_port})")

        y_rows, x_cols = build_resampler(w, h)
        cur_bid = None
        frame_buf = bytearray(w * h)
        stride = 0
        last_hb = time.monotonic()
        last_lcd = 0.0
        min_period = 1.0 / max(1.0, args.fps_limit)
        frames_sent = 0
        t_stat = time.monotonic()

        while running:
            now = time.monotonic()
            if now - last_hb >= 1.0:
                gvcp.read_reg(REG_GVCP_CCP)
                last_hb = now

            try:
                pkt, _ = gvsp_sock.recvfrom(9216)
            except socket.timeout:
                continue

            parsed = parse_gvsp_packet(pkt)
            if not parsed:
                continue
            bid, pid, fmt, body = parsed

            if fmt == 1:  # Leader
                cur_bid = bid
                stride = 0
                if len(body) >= 24:
                    _, _, _, lw, lh = struct.unpack(">2xHQIII", body[:24])
                    if lw != w or lh != h:
                        w, h = lw, lh
                        frame_buf = bytearray(w * h)
                        y_rows, x_cols = build_resampler(w, h)
            elif fmt == 3 and bid == cur_bid:  # Payload
                if stride == 0:
                    stride = len(body)
                off = (pid - 1) * stride
                end = min(off + len(body), len(frame_buf))
                if off < end:
                    frame_buf[off:end] = body[:end - off]
            elif fmt == 2 and bid == cur_bid:  # Trailer
                if (now - last_lcd) >= min_period:
                    fb = mono8_frame_to_rgb332(
                        frame_buf, y_rows, x_cols,
                        auto_contrast=args.auto_contrast,
                        dither=not args.no_dither,
                    )
                    send_frame_to_lcd(lcd_sock, args.board_ip, fb, wait_ack=not args.no_ack)
                    last_lcd = now
                    frames_sent += 1
                    if now - t_stat >= 2.0:
                        fps = frames_sent / (now - t_stat)
                        print(f"[BRIDGE] {w}x{h} -> {LCD_W}x{LCD_H} LCD: {fps:.1f} FPS")
                        frames_sent = 0
                        t_stat = now
                cur_bid = None
    finally:
        try:
            gvcp.write_reg(REG_ACQ_STOP, 0)
            gvcp.write_reg(REG_SC_PORT0, 0)
            gvcp.write_reg(REG_GVCP_CCP, 0)
        except Exception:
            pass
        gvcp.close()
        gvsp_sock.close()
        lcd_sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
