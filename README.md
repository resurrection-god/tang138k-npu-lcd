# Tang Mega 138K Pro — Ethernet Image Classifier on Gowin FPGA NPU + RGB666 LCD

Готовый проект классификации изображений (**Кошка / Собака / Неизвестно**) на плате **Sipeed Tang Mega 138K Pro Dock** (`GW5AST-LV138FPG676AC1/I0` + жёсткое ядро RISC-V **Andes AE350**) с загрузкой картинок по **Ethernet (UDP `:5555`)**, аппаратным ускорителем **NPU (`npu_core.v` v2.0.0)** на базе DSP-блоков ПЛИС и выводом картинки с HUD-оверлеем на экран **RGB666 LCD (`800×480`)**.

---

## Архитектура (v2.0.0 — Weight-Stationary CNN NPU + ShuffleNetV2-x0.5 INT8)

1. **Аппаратный NPU в ПЛИС ([npu_core.v](file:///C:/Users/penis/.gemini/antigravity/scratch/tang138k-npu-lcd/fpga/src/npu_core.v)):**
   - **Конвейер:** `SIMD-4` знаковый `INT8 × INT8 -> INT16 -> INT32` DSP MAC-ускоритель на частоте `AHB_CLK = 100 МГц` (`400 MMAC/s`).
   - **Локальная память BSRAM:** `32 КБ` (`8192 × 32 бит`) Weight BSRAM, `256 × 32 бит` Bias BSRAM, `1 КБ` (`256 × 32 бит`) Activation BSRAM и упакованный выходной буфер `256 × 32 бит` / `4 × 64 × 8 бит`.
   - **Аппаратный постпроцессинг:** `ROUND_EN`, `SHIFT_BITS`, `RELU_EN`, `CLAMP_I8`, `OVF`, `EXEC_TIME_NS` и `MAC_COUNT`.
   - **Единый источник истины карты регистров ([regs.toml](file:///C:/Users/penis/.gemini/antigravity/scratch/tang138k-npu-lcd/regs.toml) -> [regmap.py](file:///C:/Users/penis/.gemini/antigravity/scratch/tang138k-npu-lcd/tools/npu/regmap.py)):**
     - Базовый адрес на шине APB: `0xF8000900`
     - `MAP_HASH`: `0xAEE99776` (проверяется прошивкой в `lcd_hud_init()` через регистр `ID` по смещению `0x00`).

2. **Нейросетевая модель и контроллер LCD ([lcd_hud.c](file:///C:/Users/penis/.gemini/antigravity/scratch/tang138k-npu-lcd/firmware/src/app/lcd_hud.c), [train_shufflenet_npu.py](file:///C:/Users/penis/.gemini/antigravity/scratch/tang138k-npu-lcd/tools/npu/train_shufflenet_npu.py)):**
   - Принимает кадр `200×120` (`RGB332`) по протоколу `"LCD1"` (UDP `:5555`, совместимо с `TangMega138K-LCD-Studio`) и пересчитывает integer box-фильтром в `96×96×3` (`int8`).
   - Исполняет предобученную и дообученную через **INT8 QAT с само-дистилляцией** сеть **ShuffleNetV2-x0.5** (`168 355` параметров, `174 548` байт в `npu_weights.h`, все PW/`conv1`/`conv5`/`fc`-слои на FPGA `npu_core`, `5 419 392` аппаратных MAC на кадр).
   - Точность квантованной модели: `85.65%` (balanced `85.66%`) на `QAT val`, `79.33%` (`476/600`) на побитовом эмуляторе `200×120 RGB332 + Floyd-Steinberg`, **`17/20` (`85.0%`) zero-shot** на изображениях пользователя в `Downloads` (`8/8` кошек включая мемы, `2/2` собаки, `7/10` аппаратных и калибровочных фото).

3. **Клиент загрузки и проверки ([upload_and_classify.py](file:///C:/Users/penis/.gemini/antigravity/scratch/tang138k-npu-lcd/tools/npu/upload_and_classify.py)):**
   - Поддерживает `.png`, `.rgb332` (а также `.jpg`, `.jpeg`, `.webp` при запуске из `npu_ml_venv` с `Pillow`), Флойд–Штейнберг дизеринг, флаги `--demo`, `--image` и `--downloads`.

---

## Быстрый запуск

```powershell
# 1. Проверка генератора карты регистров (regs.toml -> generated/*)
python tools\npu\regmap.py --check

# 2. Переобучение/QAT/экспорт весов (опционально)
C:\Users\penis\.gemini\antigravity\scratch\npu_ml_venv\Scripts\python.exe -u tools\npu\train_shufflenet_npu.py --qat

# 3. Загрузка готового битстрима в SRAM платы (JTAG USB location 4641)
powershell -ExecutionPolicy Bypass -File tools\board\run_sram.ps1 -SkipBuild -UsbLocation 4641

# 4. Загрузка прошивки по Ethernet (Dual-Slot Safe OTA в Slot B)
python tools\board\ota_upload.py --host 10.0.0.222 --bin firmware\Release\output\ae350_scanner.bin

# 5. Демо-прогон или проверка папки с картинками на плате
python tools\npu\upload_and_classify.py --host 10.0.0.222 --demo
C:\Users\penis\.gemini\antigravity\scratch\npu_ml_venv\Scripts\python.exe tools\npu\upload_and_classify.py --host 10.0.0.222 --downloads C:\Users\penis\Downloads
```
