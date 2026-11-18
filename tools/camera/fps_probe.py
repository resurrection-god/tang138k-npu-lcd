"""Хост для отдельного битстрима fpga/fps_probe: частота кадров сенсора по ROI.

Битстрим fps_probe грузится ТОЛЬКО в SRAM (programmer_cli --run 2) — SoC и сети
в нём нет, камера настраивается самой ПЛИС по I2C, а общение идёт по COM7:

    python tools/camera/fps_probe.py                     # таблица по умолчанию
    python tools/camera/fps_probe.py --roi 1080x1280 512x1280 1080x256
    python tools/camera/fps_probe.py --watch             # просто печатать отчёты

Что меряется для каждого ROI (камера в свободном потоке, частота запрошена
заведомо выше предела — сенсор сам встаёт на свой максимум):
  * стробы/с и период строба — прямое время кадра, мимо MIPI вовсе;
  * строки/с, принятые мостом CSI-2 с длиной строки, равной ширине ROI;
  * word count последнего пакета на проводе — какую строку камера реально шлёт;
  * регистры 0x0818/0x081C/0x0814 и 0x0838 (предельная частота для ROI),
    прочитанные самой ПЛИС, до запуска и через 200 мс после.
"""
import argparse
import struct
import sys
import time

import serial

WORDS = ["seq", "w", "h", "fps_req", "exp", "stb", "stb_per", "stb_hi",
         "lines", "hdr", "bad", "wc", "bclk", "id", "rb_w", "rb_h",
         "rb_fps", "rb_maxfps", "rb_fps_run", "rb_maxfps_run", "nacks",
         "wd_resets", "status", "rsv"]
NW = len(WORDS)
CLK_HZ = 50_000_000

DEFAULT_ROI = ["1080x1280", "1080x640", "1080x256", "1080x128",
               "1024x1280", "1016x1280", "512x1280", "256x1280",
               "1016x256", "512x256", "256x256"]


def read_report(port, timeout=3.0):
    """Следующий целый отчёт; кадр 0x5A 0xA5, NW слов BE, байт суммы."""
    t0 = time.time()
    state = 0
    while time.time() - t0 < timeout:
        b = port.read(1)
        if not b:
            continue
        if state == 0:
            state = 1 if b == b"\x5a" else 0
        elif state == 1:
            if b == b"\xa5":
                body = port.read(4 * NW + 1)
                if len(body) != 4 * NW + 1:
                    return None
                if (sum(body[:-1]) & 0xFF) != body[-1]:
                    state = 0
                    continue
                return dict(zip(WORDS, struct.unpack(f">{NW}I", body[:-1])))
            state = 1 if b == b"\x5a" else 0
    return None


def send_cfg(port, w, h, fps_hz, exp_us):
    port.write(b"\xa5" + struct.pack(">IIII", w, h, int(round(fps_hz * 100)), exp_us))


def measure(port, w, h, fps_hz, exp_us, settle=3):
    port.reset_input_buffer()
    send_cfg(port, w, h, fps_hz, exp_us)
    # Отчёт идёт раз в секунду. Ждём, пока ПЛИС примет команду и закончит
    # настройку (бит 0 статуса — занят I2C), и ещё settle целых секунд после.
    good = 0
    last = None
    t0 = time.time()
    while time.time() - t0 < 20:
        r = read_report(port)
        if r is None:
            continue
        if r["w"] != w or r["h"] != h or (r["status"] & 1) or not (r["status"] & 2):
            good = 0
            continue
        good += 1
        last = r
        if good > settle:
            return r
    return last


def fmt(r):
    stb_fps = CLK_HZ / r["stb_per"] if r["stb_per"] else 0.0
    return (f"{r['w']:>5}x{r['h']:<5} | {r['rb_w']:>5}x{r['rb_h']:<5} | "
            f"{r['rb_fps'] / 100:8.2f} {r['rb_fps_run'] / 100:8.2f} | "
            f"{r['rb_maxfps'] / 100:8.2f} {r['rb_maxfps_run'] / 100:8.2f} | "
            f"{r['stb']:>5} {stb_fps:8.1f} | {r['lines']:>7} {r['lines'] / r['h']:8.1f} | "
            f"{r['wc']:>5} | {r['bclk'] * 1024 / 1e6:6.1f} | {r['nacks']:>3} {r['wd_resets']:>3}")


HEAD = (f"{'задано':>11} | {'прочитано':>11} | {'0x0814 до/после':>17} | "
        f"{'0x0838 до/после':>17} | {'стробы, к/с':>14} | {'строк/с, к/с':>16} | "
        f"{'wc':>5} | {'бклк':>6} | nack wd")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM7")
    ap.add_argument("--roi", nargs="*", default=DEFAULT_ROI, help="список WxH")
    ap.add_argument("--fps", type=float, default=3000.0, help="запрашиваемая частота, Гц")
    ap.add_argument("--exp", type=int, default=200, help="экспозиция, мкс")
    ap.add_argument("--watch", action="store_true", help="только печатать отчёты")
    a = ap.parse_args()

    with serial.Serial(a.port, 115200, timeout=0.5) as port:
        if a.watch:
            while True:
                r = read_report(port)
                if r:
                    print(r)
            return
        r = read_report(port, timeout=4.0)
        if r is None:
            print("нет отчётов от fps_probe на", a.port, "— битстрим загружен?")
            return 1
        print(f"камера: id 0x{r['id']:08X} ({'VEYE' if r['id'] == 0x56455945 else 'НЕ VEYE'})")
        print(f"экспозиция {a.exp} мкс, запрошено {a.fps:.0f} к/с (частоты в кадрах/с)")
        print(HEAD)
        print("-" * len(HEAD))
        for roi in a.roi:
            w, h = (int(x) for x in roi.lower().split("x"))
            r = measure(port, w, h, a.fps, a.exp)
            print(fmt(r) if r else f"{w:>5}x{h:<5} | нет устойчивого отчёта")
        send_cfg(port, 1080, 1280, a.fps, a.exp)


if __name__ == "__main__":
    sys.exit(main())
