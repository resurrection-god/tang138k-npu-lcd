"""
Проверка блока синхронизации на плате (docs/architecture/SYNC_BLOCK.md).

  python tools/camera/sync_check.py

Всё — через VMSP 39151 (stp_client.py). Управление блоком — почтовый ящик DDR3
(ПЛИС читает раз в ~10 мс), состояние — телеметрия ПЛИС (обновляется раз в
0.5 с), поэтому между действием и чтением счётчиков здесь паузы.

Что делает (камера 0 подключена: Trigger In на A18, Strobe Out на A17):
  1. 0x0142 CORE + 0x0140 канал 0 -> телеметрия есть, ПЛИС видит APPLY_SEQ;
  2. CORE: камера в аппаратном триггере (0x0112), 20 x 0x0145 -> принято +20,
     камера (0x0418) насчитала 20;
  3. перегрузка: ширина 1 с, два 0x0145 через 100 мс -> перегрузок +1;
  4. TIMER: серия 0x0302 из 10 импульсов -> принято +10, камера +10
     (прежнее поведение платы);
  5. лазер: 0x0140 канал 2 записывается и применяется;
  в конце всё, что проверка трогала, возвращается как было (см. snapshot ниже).
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capture"))
import stp_client as stp  # noqa: E402

TELEMETRY_S = 0.8          # телеметрия ПЛИС раз в 0.5 с
results = []

# Разряды слова CTRL блока синхронизации (firmware/src/app/sync.h).
CTRL_ENABLE    = 0x001
CTRL_SRC_MASK  = 0x00C
CTRL_SRC_SHIFT = 2
CTRL_EXT_FALL  = 0x010
CTRL_LASER_EN  = 0x100


def ticks_to_ns(ticks):
    """Такт lane4_clk — stp.LANE_TICK_NS (10 нс с 29.09.2026, tools/common/memory_map.py).

    0x0300 отдаёт задержки и ширины в тактах, а 0x0140 принимает их в
    наносекундах, так что для восстановления нужен обратный перевод. При такте
    10 нс он точный; округление вверх оставлено с 7.5 нс, где прошивка отбрасывала
    остаток и round-trip через floor терял такт (13 тактов -> 97 нс -> 12).
    """
    return math.ceil(ticks * stp.LANE_TICK_NS)


def snapshot(h):
    """Состояние, которое проверка перезаписывает, — чтобы вернуть его в finally.

    Раньше вместо этого в конце прописывались жёстко заданные значения, и одно
    из них тихо ломало съёмку с подсветкой: канал лазера «восстанавливался» в
    100 нс, а прошивка брала эту ширину из ящика, когда измеренный ПЛИС строб
    непригоден. В кадре 100 нс подсветки неотличимы от фона (ERRORS гл. 31.52).
    """
    st = stp.status(h)
    try:
        cam = stp.get_camera_config(0, host=h)
    except Exception:
        cam = None
    return {"st": st, "cam": cam}


def restore(h, snap):
    """Вернуть каналы, управление и конфигурацию камеры 0 как было."""
    st = snap["st"]
    for ch, d, w in ((0, "cam0_delay", "cam0_width"),
                     (1, "cam1_delay", "cam1_width"),
                     (2, "laser_delay", "laser_width")):
        stp.sync_channel(ch, ticks_to_ns(st[d]), ticks_to_ns(st[w]),
                         laser_mode=st["laser_mode"] if ch == 2 else 0, host=h)
    ctrl = st["sync_ctrl"]
    stp.sync_control(1 if ctrl & CTRL_ENABLE else 0,
                     (ctrl & CTRL_SRC_MASK) >> CTRL_SRC_SHIFT,
                     1 if ctrl & CTRL_LASER_EN else 0,
                     1 if ctrl & CTRL_EXT_FALL else 0,
                     st["ext_filter"], st["polarity"], host=h)
    cam = snap["cam"]
    if cam:
        stp.camera_config(cam["exp_us"], cam["gain_db"], cam["fps"],
                          trigger_mode=cam["trg_mode"], testpat=cam["testpat"], host=h)


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{('  ' + str(detail)) if detail else ''}")


def main():
    h = stp.HOST
    # С 02.10.2026 плата стартует в STANDBY, где блок запрещён и съёмка
    # отказывается: проверка идёт в SERVICE, режим в конце возвращается.
    was_mode = stp.ensure_active(h)
    snap = snapshot(h)
    try:
        return run_checks(h)
    finally:
        # Восстановление в finally, а не в конце run_checks: проверка падает
        # исключением на любом обрыве связи, и раньше плата оставалась с
        # настройками проверки — в том числе с лазером в 100 нс.
        try:
            restore(h, snap)
            if was_mode == stp.MODE_STANDBY:
                stp.set_mode(stp.MODE_STANDBY, h)
        except Exception as e:                      # noqa: BLE001
            print(f"  [WARN] состояние не восстановлено: {e}")


def run_checks(h):
    print("-- присутствие и применение")
    # До 02.10.2026 здесь был discovery; вне STANDBY плата на него молчит.
    check("плата отвечает по TCP, режим не STANDBY", stp.get_mode(h) != stp.MODE_STANDBY)
    stp.trig_series(1, 0, 1, stop=True, host=h)
    stp.sync_control(1, "core", 0, host=h)
    stp.sync_channel(0, 0, 1_000_000, host=h)            # cam0: без задержки, 1 мс
    time.sleep(TELEMETRY_S)
    st = stp.status(h)
    check("телеметрия ПЛИС: слова блока есть", st["telemetry_ok"] == 1, st["sync_status"])
    check("ПЛИС видит последний APPLY_SEQ", st["apply_seq_fpga"] == st["apply_seq_fw"],
          f"{st['apply_seq_fpga']}/{st['apply_seq_fw']}")
    check("блок разрешён, ящик прочитан", (st["sync_status"] & 0x101) == 0x101, hex(st["sync_status"]))
    # Ширина 1 мс в тактах lane4_clk: 100 000 при 10 нс (133 333 при 7.5 нс до 29.09.2026).
    w1ms = 1_000_000 // stp.LANE_TICK_NS
    check(f"канал 0 = 0 + {w1ms} такта", (st["cam0_delay"], st["cam0_width"]) == (0, w1ms),
          (st["cam0_delay"], st["cam0_width"]))

    print("-- CORE: программный триггер камеры")
    stp.camera_config(1000, 6.0, 10, trigger_mode=1, host=h)
    time.sleep(1.5)
    st = stp.status(h)
    a0, c0 = st["accepted"], st["cam_triggers"]
    check("камера во внешнем триггере (0x0404 = 1, 0x0408 = 1)",
          (st["cam_trg_mode"], st["cam_trg_src"]) == (1, 1), (st["cam_trg_mode"], st["cam_trg_src"]))
    for _ in range(20):
        stp.sw_trigger(h)
        time.sleep(0.15)
    time.sleep(TELEMETRY_S)
    st = stp.status(h)
    check("принято +20", st["accepted"] - a0 == 20, f"{a0} -> {st['accepted']}")
    check("ПЛИС видит последний SW_SEQ", st["sw_seq_fpga"] == st["sw_seq_fw"],
          f"{st['sw_seq_fpga']}/{st['sw_seq_fw']}")
    check("камера насчитала 20 импульсов CORE", ((st["cam_triggers"] - c0) & 0xFFFF) == 20,
          f"{c0} -> {st['cam_triggers']}")

    print("-- перегрузка")
    # Ширина 1 с, а не 200 мс: команда по сети иногда идёт сотни миллисекунд
    # (замер 15.09: 9..49 мс, выброс 339 мс), и при 200 мс второй триггер успевал
    # прийти после конца серии — два принятых вместо перегрузки.
    stp.sync_channel(0, 0, 1_000_000_000, host=h)        # 1 с
    time.sleep(0.1)
    o0 = stp.status(h)["overrun"]
    stp.sw_trigger(h)
    time.sleep(0.1)                                      # больше периода опроса ящика
    stp.sw_trigger(h)
    time.sleep(1.2 + TELEMETRY_S)
    st = stp.status(h)
    check("перегрузок +1", st["overrun"] - o0 == 1, f"{o0} -> {st['overrun']}")
    stp.sync_channel(0, 0, 1_000_000, host=h)

    print("-- TIMER: серия генератора, как до блока")
    stp.sync_control(1, "timer", 0, host=h)
    time.sleep(0.1 + TELEMETRY_S)
    st = stp.status(h)
    a0, c0 = st["accepted"], st["cam_triggers"]
    stp.trig_series(10, 10, 1000, host=h)
    time.sleep(1.5 + TELEMETRY_S)
    st = stp.status(h)
    check("принято +10", st["accepted"] - a0 == 10, f"{a0} -> {st['accepted']}")
    check("камера насчитала 10 импульсов TIMER", ((st["cam_triggers"] - c0) & 0xFFFF) == 10,
          f"{c0} -> {st['cam_triggers']}")

    print("-- лазер")
    stp.sync_channel(2, 250, 100, laser_mode=2, host=h)
    st = stp.status(h)
    ld, lw = 250 // stp.LANE_TICK_NS, 100 // stp.LANE_TICK_NS   # 25 + 10 при 10 нс
    check(f"лазер {ld} + {lw} тактов (250 + 100 нс по {stp.LANE_TICK_NS} нс), режим 2 записано",
          (st["laser_delay"], st["laser_width"], st["laser_mode"]) == (ld, lw, 2),
          (st["laser_delay"], st["laser_width"], st["laser_mode"]))
    time.sleep(TELEMETRY_S)
    st = stp.status(h)
    check("ПЛИС применила настройку лазера", st["apply_seq_fpga"] == st["apply_seq_fw"])

    print(f"итого: {sum(results)}/{len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
