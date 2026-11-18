/*
 * lcd_hud.c - High-Speed 60 FPS APB RGB666 200x120 RGB332 Framebuffer,
 *             Hardware FPGA NPU 5-Stage CNN Classifier (Cat / Dog / Unknown) & UDP Control (Port 5555).
 */

#include "lcd_hud.h"
#include "npu_regs.h"
#include "npu_weights.h"
#include "ota.h"
#include "net_config.h"

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#include <stdio.h>
#include <string.h>

#define LCD_REG_MAGIC       (*(volatile uint32_t *)(LCD_HUD_BASE + 0x000U))
#define LCD_REG_CTRL        (*(volatile uint32_t *)(LCD_HUD_BASE + 0x004U))
#define LCD_REG_STATUS      (*(volatile uint32_t *)(LCD_HUD_BASE + 0x008U))
#define LCD_REG_FB_WR       (*(volatile uint32_t *)(LCD_HUD_BASE + 0x010U))
#define LCD_REG_PULSE       (*(volatile uint32_t *)(LCD_HUD_BASE + 0x014U))
#define LCD_REG_FB_ADDR     (*(volatile uint32_t *)(LCD_HUD_BASE + 0x018U))
#define LCD_REG_FB_DATA4    (*(volatile uint32_t *)(LCD_HUD_BASE + 0x01CU))
#define LCD_REG_HW_RX_CFG   (*(volatile uint32_t *)(LCD_HUD_BASE + 0x020U))
#define LCD_REG_HW_RX_PORT  (*(volatile uint32_t *)(LCD_HUD_BASE + 0x024U))
#define LCD_REG_HW_RX_STAT  (*(volatile uint32_t *)(LCD_HUD_BASE + 0x028U))
#define LCD_REG_HW_RX_GEOM  (*(volatile uint32_t *)(LCD_HUD_BASE + 0x02CU))

/* Simple UDP Protocol ("LCD1") */
#define LCD_PROTO_MAGIC     0x3144434CUL /* "LCD1" LE */
#define LCD_CMD_PING        0x01U
#define LCD_CMD_CONFIG      0x02U
#define LCD_CMD_GRAPH       0x03U
#define LCD_CMD_IMAGE_CHUNK 0x04U
#define LCD_CMD_PIXEL_DRAW  0x05U
#define LCD_CMD_NPU_QUERY   0x07U

static bool                 s_active = false;
static bool                 s_npu_ok = false;
static SemaphoreHandle_t    s_mtx = NULL;
static TaskHandle_t         s_render_task_h = NULL;

/* Framebuffers in RAM (24 KB each) */
static uint8_t              s_fb[LCD_FB_SIZE];
static uint8_t              s_img_fb[LCD_FB_SIZE];
static uint8_t              s_cam_fb[LCD_FB_SIZE];
static uint8_t              s_graph_pts[LCD_FB_W];
static volatile uint32_t    s_cam_frame_id = 0U;

/* Hardware NPU classification state */
static volatile uint8_t     s_npu_pred_class = NPU_CLASS_UNKNOWN;
static volatile uint8_t     s_npu_conf_pct   = 99U;
static volatile uint16_t    s_npu_infer_cnt  = 0U;
static volatile int32_t     s_npu_logits[3]  = { 0, 0, 0 };
static volatile uint32_t    s_npu_exec_ns    = 0U;
static volatile uint32_t    s_npu_mac_cnt    = 0U;
static volatile uint32_t    s_npu_hw_hash    = 0U;

/* Screen & Animation configuration */
static volatile lcd_anim_mode_t s_mode = ANIM_MODE_IMAGE;
static volatile uint8_t     s_rot = 2U;          /* 180 deg upright */
static volatile bool        s_mode_800 = true;   /* 800x480 */
static volatile bool        s_show_border = true;
static volatile uint8_t     s_dim = 0U;          /* 0=100%, 1=75%, 2=50%, 3=25% */
static volatile uint8_t     s_speed = 2U;        /* 1..20 */
static volatile uint8_t     s_graph_style = 1U;  /* 0=line, 1=fill, 2=bars */
static volatile uint8_t     s_fg_col = 0x1FU;    /* Bright cyan-green RGB332 */
static volatile uint8_t     s_bg_col = 0x01U;    /* Deep navy RGB332 */
static volatile uint32_t    s_fps_meas = 60U;
static volatile uint32_t    s_udp_rx_cnt = 0U;
static volatile bool        s_dirty = true;
static volatile bool        s_mono8 = false;
static volatile bool        s_hw_gvsp = false;

/* Bouncing box state for Plasma mode */
static int16_t s_box_x = 30;
static int16_t s_box_y = 20;
static int8_t  s_box_dx = 2;
static int8_t  s_box_dy = 1;
static uint8_t s_box_col = 0xE0U;

/* Equalizer state */
#define NUM_EQ_BARS 20U
static uint8_t s_eq_peaks[NUM_EQ_BARS];

static const int8_t s_sin_lut[64] = {
      0,   6,  12,  19,  24,  30,  36,  41,  45,  49,  53,  56,  59,  61,  63,  64,
     64,  64,  63,  61,  59,  56,  53,  49,  45,  41,  36,  30,  24,  19,  12,   6,
      0,  -6, -12, -19, -24, -30, -36, -41, -45, -49, -53, -56, -59, -61, -63, -64,
    -64, -64, -63, -61, -59, -56, -53, -49, -45, -41, -36, -30, -24, -19, -12,  -6
};

static inline uint8_t rgb_to_332(uint8_t r, uint8_t g, uint8_t b)
{
    return (uint8_t)((r & 0xE0U) | ((g >> 3) & 0x1CU) | (b >> 6));
}

static inline uint32_t get_ctrl_val(void)
{
    uint32_t hw_plasma = (s_mode == ANIM_MODE_PLASMA) ? 1U : 0U;
    uint32_t spd = (s_speed > 20U) ? 20U : ((s_speed < 1U) ? 1U : s_speed);
    return (1U << 0)
         | (s_mode_800 ? (1U << 1) : 0U)
         | (1U << 2)
         | (s_show_border ? (1U << 3) : 0U)
         | (((uint32_t)s_rot & 3U) << 4)
         | (((uint32_t)s_dim & 3U) << 6)
         | (hw_plasma << 8)
         | ((spd & 0x1FU) << 9)
         | (s_mono8 ? (1U << 14) : 0U);
}

static void apply_ctrl(void)
{
    if (s_active) {
        LCD_REG_CTRL = get_ctrl_val();
    }
}

static void flush_fb(void)
{
    volatile uint32_t * const reg = (volatile uint32_t *)(LCD_HUD_BASE + 0x01CU);
    const uint32_t *src = (const uint32_t *)s_fb;
    LCD_REG_FB_ADDR = 0U;
    for (uint32_t i = 0; i < (LCD_FB_SIZE / 4U); i += 8U) {
        reg[0] = src[i + 0U];
        reg[0] = src[i + 1U];
        reg[0] = src[i + 2U];
        reg[0] = src[i + 3U];
        reg[0] = src[i + 4U];
        reg[0] = src[i + 5U];
        reg[0] = src[i + 6U];
        reg[0] = src[i + 7U];
    }
}

static inline void fb_put(int16_t x, int16_t y, uint8_t col)
{
    if (x >= 0 && x < (int16_t)LCD_FB_W && y >= 0 && y < (int16_t)LCD_FB_H) {
        s_fb[(uint32_t)y * LCD_FB_W + (uint32_t)x] = col;
    }
}

static void fb_fill_rect(int16_t x0, int16_t y0, int16_t w, int16_t h, uint8_t col)
{
    int16_t y, x;
    for (y = y0; y < y0 + h; ++y) {
        if (y < 0 || y >= (int16_t)LCD_FB_H) continue;
        for (x = x0; x < x0 + w; ++x) {
            if (x >= 0 && x < (int16_t)LCD_FB_W) {
                s_fb[(uint32_t)y * LCD_FB_W + (uint32_t)x] = col;
            }
        }
    }
}

static void fb_draw_line_v(int16_t x, int16_t y0, int16_t y1, uint8_t col)
{
    int16_t y;
    if (x < 0 || x >= (int16_t)LCD_FB_W) return;
    if (y0 > y1) { int16_t t = y0; y0 = y1; y1 = t; }
    if (y0 < 0) y0 = 0;
    if (y1 >= (int16_t)LCD_FB_H) y1 = (int16_t)LCD_FB_H - 1;
    for (y = y0; y <= y1; ++y) {
        s_fb[(uint32_t)y * LCD_FB_W + (uint32_t)x] = col;
    }
}

/* ============================================================================
 * 5x7 Minimal Bitmap Font for LCD HUD Overlay
 * ============================================================================ */
static void get_glyph_5x7(char c, uint8_t rows[7])
{
    memset(rows, 0, 7);
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    switch (c) {
        case '0': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x13; rows[3]=0x15; rows[4]=0x19; rows[5]=0x11; rows[6]=0x0E; break;
        case '1': rows[0]=0x04; rows[1]=0x0C; rows[2]=0x04; rows[3]=0x04; rows[4]=0x04; rows[5]=0x04; rows[6]=0x0E; break;
        case '2': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x01; rows[3]=0x06; rows[4]=0x08; rows[5]=0x10; rows[6]=0x1F; break;
        case '3': rows[0]=0x1F; rows[1]=0x02; rows[2]=0x04; rows[3]=0x02; rows[4]=0x01; rows[5]=0x11; rows[6]=0x0E; break;
        case '4': rows[0]=0x02; rows[1]=0x06; rows[2]=0x0A; rows[3]=0x12; rows[4]=0x1F; rows[5]=0x02; rows[6]=0x02; break;
        case '5': rows[0]=0x1F; rows[1]=0x10; rows[2]=0x1E; rows[3]=0x01; rows[4]=0x01; rows[5]=0x11; rows[6]=0x0E; break;
        case '6': rows[0]=0x06; rows[1]=0x08; rows[2]=0x10; rows[3]=0x1E; rows[4]=0x11; rows[5]=0x11; rows[6]=0x0E; break;
        case '7': rows[0]=0x1F; rows[1]=0x01; rows[2]=0x02; rows[3]=0x04; rows[4]=0x08; rows[5]=0x08; rows[6]=0x08; break;
        case '8': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x0E; rows[4]=0x11; rows[5]=0x11; rows[6]=0x0E; break;
        case '9': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x0F; rows[4]=0x01; rows[5]=0x02; rows[6]=0x0C; break;
        case 'A': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x1F; rows[4]=0x11; rows[5]=0x11; rows[6]=0x11; break;
        case 'B': rows[0]=0x1E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x1E; rows[4]=0x11; rows[5]=0x11; rows[6]=0x1E; break;
        case 'C': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x10; rows[3]=0x10; rows[4]=0x10; rows[5]=0x11; rows[6]=0x0E; break;
        case 'D': rows[0]=0x1C; rows[1]=0x12; rows[2]=0x11; rows[3]=0x11; rows[4]=0x11; rows[5]=0x12; rows[6]=0x1C; break;
        case 'E': rows[0]=0x1F; rows[1]=0x10; rows[2]=0x10; rows[3]=0x1E; rows[4]=0x10; rows[5]=0x10; rows[6]=0x1F; break;
        case 'F': rows[0]=0x1F; rows[1]=0x10; rows[2]=0x10; rows[3]=0x1E; rows[4]=0x10; rows[5]=0x10; rows[6]=0x10; break;
        case 'G': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x10; rows[3]=0x17; rows[4]=0x11; rows[5]=0x11; rows[6]=0x0F; break;
        case 'H': rows[0]=0x11; rows[1]=0x11; rows[2]=0x11; rows[3]=0x1F; rows[4]=0x11; rows[5]=0x11; rows[6]=0x11; break;
        case 'I': rows[0]=0x0E; rows[1]=0x04; rows[2]=0x04; rows[3]=0x04; rows[4]=0x04; rows[5]=0x04; rows[6]=0x0E; break;
        case 'K': rows[0]=0x11; rows[1]=0x12; rows[2]=0x14; rows[3]=0x18; rows[4]=0x14; rows[5]=0x12; rows[6]=0x11; break;
        case 'L': rows[0]=0x10; rows[1]=0x10; rows[2]=0x10; rows[3]=0x10; rows[4]=0x10; rows[5]=0x10; rows[6]=0x1F; break;
        case 'M': rows[0]=0x11; rows[1]=0x1B; rows[2]=0x15; rows[3]=0x15; rows[4]=0x11; rows[5]=0x11; rows[6]=0x11; break;
        case 'N': rows[0]=0x11; rows[1]=0x19; rows[2]=0x15; rows[3]=0x13; rows[4]=0x11; rows[5]=0x11; rows[6]=0x11; break;
        case 'O': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x11; rows[4]=0x11; rows[5]=0x11; rows[6]=0x0E; break;
        case 'P': rows[0]=0x1E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x1E; rows[4]=0x10; rows[5]=0x10; rows[6]=0x10; break;
        case 'R': rows[0]=0x1E; rows[1]=0x11; rows[2]=0x11; rows[3]=0x1E; rows[4]=0x14; rows[5]=0x12; rows[6]=0x11; break;
        case 'S': rows[0]=0x0F; rows[1]=0x10; rows[2]=0x10; rows[3]=0x0E; rows[4]=0x01; rows[5]=0x01; rows[6]=0x1E; break;
        case 'T': rows[0]=0x1F; rows[1]=0x04; rows[2]=0x04; rows[3]=0x04; rows[4]=0x04; rows[5]=0x04; rows[6]=0x04; break;
        case 'U': rows[0]=0x11; rows[1]=0x11; rows[2]=0x11; rows[3]=0x11; rows[4]=0x11; rows[5]=0x11; rows[6]=0x0E; break;
        case 'W': rows[0]=0x11; rows[1]=0x11; rows[2]=0x11; rows[3]=0x15; rows[4]=0x15; rows[5]=0x1B; rows[6]=0x11; break;
        case ':': rows[1]=0x04; rows[2]=0x04; rows[4]=0x04; rows[5]=0x04; break;
        case '%': rows[0]=0x18; rows[1]=0x19; rows[2]=0x02; rows[3]=0x04; rows[4]=0x08; rows[5]=0x13; rows[6]=0x03; break;
        case '?': rows[0]=0x0E; rows[1]=0x11; rows[2]=0x01; rows[3]=0x06; rows[4]=0x04; rows[6]=0x04; break;
        case '-': rows[3]=0x1F; break;
        default: break;
    }
}

static void fb_draw_str(int16_t x0, int16_t y0, const char *s, uint8_t fg, int16_t scale)
{
    int16_t cx = x0;
    while (*s) {
        uint8_t rows[7];
        get_glyph_5x7(*s++, rows);
        for (int16_t r = 0; r < 7; ++r) {
            uint8_t bits = rows[r];
            for (int16_t c = 0; c < 5; ++c) {
                if (bits & (1U << (4 - c))) {
                    if (scale <= 1) {
                        fb_put(cx + c, y0 + r, fg);
                    } else {
                        fb_fill_rect(cx + c * scale, y0 + r * scale, scale, scale, fg);
                    }
                }
            }
        }
        cx = (int16_t)(cx + 6 * scale);
    }
}

/* ============================================================================
 * Hardware FPGA NPU Pretrained ShuffleNetV2-x0.5 Engine (npu_core v2 @ 0xF8000900)
 * ============================================================================ */
static int8_t s_img_96x96[96U * 96U * 3U] __attribute__((aligned(4)));
static int8_t s_feat_a[48U * 48U * 24U]   __attribute__((aligned(4)));
static int8_t s_feat_b[48U * 48U * 24U]   __attribute__((aligned(4)));
static int8_t s_br1[24U * 24U * 24U]      __attribute__((aligned(4)));
static int8_t s_br2[24U * 24U * 24U]      __attribute__((aligned(4)));
static int8_t s_br3[24U * 24U * 24U]      __attribute__((aligned(4)));

/* Downsample 200x120 RGB332 -> 96x96x3 signed INT8 using integer box filter
 * (eliminates Floyd-Steinberg RGB332 dithering speckles sent by Studio). */
static void npu_downsample_rgb332_to_96x96(const uint8_t *fb_200x120, int8_t *out_hwc)
{
    for (uint32_t oy = 0; oy < NPU_IN_SIZE; ++oy) {
        uint32_t y0 = (oy * LCD_FB_H) / NPU_IN_SIZE;
        uint32_t y1 = ((oy + 1U) * LCD_FB_H) / NPU_IN_SIZE;
        if (y1 <= y0) y1 = y0 + 1U;
        for (uint32_t ox = 0; ox < NPU_IN_SIZE; ++ox) {
            uint32_t x0 = (ox * LCD_FB_W) / NPU_IN_SIZE;
            uint32_t x1 = ((ox + 1U) * LCD_FB_W) / NPU_IN_SIZE;
            if (x1 <= x0) x1 = x0 + 1U;

            uint32_t sr = 0U, sg = 0U, sb = 0U, cnt = 0U;
            for (uint32_t y = y0; y < y1; ++y) {
                const uint8_t *row = &fb_200x120[y * LCD_FB_W];
                for (uint32_t x = x0; x < x1; ++x) {
                    uint8_t c = row[x];
                    uint32_t r3 = (c >> 5) & 7U;
                    uint32_t g3 = (c >> 2) & 7U;
                    uint32_t b2 = c & 3U;
                    sr += (r3 << 5) | (r3 << 2) | (r3 >> 1);
                    sg += (g3 << 5) | (g3 << 2) | (g3 >> 1);
                    sb += (b2 << 6) | (b2 << 4) | (b2 << 2) | b2;
                    ++cnt;
                }
            }

            int32_t ir = (int32_t)(sr / cnt) - 128;
            int32_t ig = (int32_t)(sg / cnt) - 128;
            int32_t ib = (int32_t)(sb / cnt) - 128;
            if (ir < -127) ir = -127; else if (ir > 127) ir = 127;
            if (ig < -127) ig = -127; else if (ig > 127) ig = 127;
            if (ib < -127) ib = -127; else if (ib > 127) ib = 127;

            uint32_t dst = (oy * NPU_IN_SIZE + ox) * 3U;
            out_hwc[dst + 0U] = (int8_t)ir;
            out_hwc[dst + 1U] = (int8_t)ig;
            out_hwc[dst + 2U] = (int8_t)ib;
        }
    }
}

static void npu_load_layer(const uint32_t *wgt_words, uint32_t vec_words, uint32_t out_ch, const int32_t *bias)
{
    uint32_t total_w = out_ch * vec_words;
    NPU_REG_DIMS = ((out_ch & 0x1FFU) << NPU_DIMS_OUT_CH_SHIFT)
                 | (vec_words & 0x1FFU);

    NPU_REG_WGT_ADDR = 0U;
    for (uint32_t i = 0; i < total_w; ++i) {
        NPU_REG_WGT_DATA4_RAW = wgt_words[i];
    }

    NPU_REG_BIAS_ADDR = 0U;
    for (uint32_t oc = 0; oc < out_ch; ++oc) {
        NPU_REG_BIAS_DATA_RAW = (uint32_t)bias[oc];
    }
}

static uint32_t npu_exec_pixel_i8(const uint32_t *act_words, uint32_t vec_words,
                                  uint32_t out_ch, uint32_t shift_bits, bool relu_en,
                                  uint32_t *out_packed)
{
    NPU_REG_ACT_ADDR = 0U;
    for (uint32_t i = 0; i < vec_words; ++i) {
        NPU_REG_ACT_DATA4_RAW = act_words[i];
    }

    NPU_REG_CTRL = NPU_CTRL_START_MASK
                 | (relu_en ? NPU_CTRL_RELU_EN_MASK : 0U)
                 | NPU_CTRL_CLAMP_I8_MASK
                 | ((shift_bits & 0x1FU) << NPU_CTRL_SHIFT_BITS_SHIFT)
                 | NPU_CTRL_ROUND_EN_MASK;

    for (uint32_t spin = 0; spin < 65536U; ++spin) {
        if ((NPU_REG_STATUS & NPU_STATUS_DONE_MASK) != 0U) break;
    }

    uint32_t out_words = (out_ch + 3U) / 4U;
    NPU_REG_OUT_ADDR = 0U;
    for (uint32_t i = 0; i < out_words; ++i) {
        out_packed[i] = NPU_REG_OUT_PACKED_RAW;
    }
    return NPU_REG_EXEC_TIME_NS;
}

static uint32_t npu_exec_pixel_i32(const uint32_t *act_words, uint32_t vec_words,
                                   uint32_t out_ch, uint32_t shift_bits, bool relu_en,
                                   int32_t *out_i32)
{
    NPU_REG_ACT_ADDR = 0U;
    for (uint32_t i = 0; i < vec_words; ++i) {
        NPU_REG_ACT_DATA4_RAW = act_words[i];
    }

    NPU_REG_CTRL = NPU_CTRL_START_MASK
                 | (relu_en ? NPU_CTRL_RELU_EN_MASK : 0U)
                 | ((shift_bits & 0x1FU) << NPU_CTRL_SHIFT_BITS_SHIFT)
                 | NPU_CTRL_ROUND_EN_MASK;

    for (uint32_t spin = 0; spin < 65536U; ++spin) {
        if ((NPU_REG_STATUS & NPU_STATUS_DONE_MASK) != 0U) break;
    }

    NPU_REG_OUT_ADDR = 0U;
    for (uint32_t i = 0; i < out_ch; ++i) {
        out_i32[i] = (int32_t)NPU_REG_OUT_DATA_RAW;
    }
    return NPU_REG_EXEC_TIME_NS;
}

/* Conv1 3x3 stride 2 pad 1 (96x96x3 -> 48x48x24) executed on FPGA NPU */
static uint32_t run_conv1_3x3_s2_npu(const int8_t *in_96x96x3, int8_t *out_48x48x24)
{
    uint32_t total_ns = 0U;
    uint32_t patch_words[7];

    npu_load_layer(&k_npu_wgt_pool[NPU_CONV1_WO], 7U, 24U, &k_npu_bias_pool[NPU_CONV1_BO]);

    for (int32_t oy = 0; oy < 48; ++oy) {
        for (int32_t ox = 0; ox < 48; ++ox) {
            uint8_t patch[28];
            uint32_t p = 0U;
            for (int32_t ky = -1; ky <= 1; ++ky) {
                int32_t iy = oy * 2 + ky;
                for (int32_t kx = -1; kx <= 1; ++kx) {
                    int32_t ix = ox * 2 + kx;
                    if (iy >= 0 && iy < 96 && ix >= 0 && ix < 96) {
                        uint32_t idx = (uint32_t)((iy * 96 + ix) * 3);
                        patch[p + 0U] = (uint8_t)in_96x96x3[idx + 0U];
                        patch[p + 1U] = (uint8_t)in_96x96x3[idx + 1U];
                        patch[p + 2U] = (uint8_t)in_96x96x3[idx + 2U];
                    } else {
                        patch[p + 0U] = 0U;
                        patch[p + 1U] = 0U;
                        patch[p + 2U] = 0U;
                    }
                    p += 3U;
                }
            }
            patch[27] = 0U;
            memcpy(patch_words, patch, sizeof(patch));
            total_ns += npu_exec_pixel_i8(
                patch_words, 7U, 24U, NPU_CONV1_SHIFT, true,
                (uint32_t *)&out_48x48x24[(oy * 48 + ox) * 24]
            );
        }
    }
    return total_ns;
}

/* MaxPool2D 3x3 stride 2 pad 1 (48x48x24 -> 24x24x24, input non-neg so 0-pad is exact) */
static void run_maxpool3x3_s2(const int8_t *in_hwc, int8_t *out_hwc)
{
    for (int32_t oy = 0; oy < 24; ++oy) {
        for (int32_t ox = 0; ox < 24; ++ox) {
            int8_t *dst = &out_hwc[(oy * 24 + ox) * 24];
            for (int32_t c = 0; c < 24; ++c) {
                int8_t mx = 0;
                for (int32_t ky = 0; ky < 3; ++ky) {
                    int32_t iy = oy * 2 + ky - 1;
                    if (iy < 0 || iy >= 48) continue;
                    for (int32_t kx = 0; kx < 3; ++kx) {
                        int32_t ix = ox * 2 + kx - 1;
                        if (ix < 0 || ix >= 48) continue;
                        int8_t v = in_hwc[(iy * 48 + ix) * 24 + c];
                        if (v > mx) mx = v;
                    }
                }
                dst[c] = mx;
            }
        }
    }
}

/* Depthwise 3x3 Conv2D (pad 1, optional ReLU, round+shift, signed/unsigned clamp INT8) */
static void run_dw3x3(const int8_t *in_hwc, int32_t H, int32_t W, int32_t C,
                      int32_t stride, const int8_t *dw_w, const int32_t *dw_b,
                      uint32_t shift_bits, bool relu_en, int8_t *out_hwc)
{
    int32_t out_H = (stride == 2) ? (H / 2) : H;
    int32_t out_W = (stride == 2) ? (W / 2) : W;
    int32_t round_bias = (shift_bits > 0U) ? (1 << (shift_bits - 1U)) : 0;

    for (int32_t oy = 0; oy < out_H; ++oy) {
        for (int32_t ox = 0; ox < out_W; ++ox) {
            int8_t *dst = &out_hwc[(oy * out_W + ox) * C];
            for (int32_t c = 0; c < C; ++c) {
                int32_t acc = dw_b[c];
                const int8_t *wc = &dw_w[c * 9];
                for (int32_t ky = 0; ky < 3; ++ky) {
                    int32_t iy = oy * stride + ky - 1;
                    if (iy < 0 || iy >= H) continue;
                    for (int32_t kx = 0; kx < 3; ++kx) {
                        int32_t ix = ox * stride + kx - 1;
                        if (ix < 0 || ix >= W) continue;
                        acc += (int32_t)in_hwc[(iy * W + ix) * C + c] * (int32_t)wc[ky * 3 + kx];
                    }
                }
                int32_t val = (acc + round_bias) >> shift_bits;
                if (relu_en && val < 0) val = 0;
                else if (val < -127) val = -127;
                if (val > 127) val = 127;
                dst[c] = (int8_t)val;
            }
        }
    }
}

/* Pointwise 1x1 Conv2D executed on FPGA NPU (weight-stationary BSRAM!) */
static uint32_t run_pw1x1_npu(const int8_t *in_hwc, int32_t H, int32_t W,
                              int32_t C_in, int32_t C_out,
                              const uint32_t *w_packed, const int32_t *b,
                              uint32_t shift_bits, int8_t *out_hwc)
{
    uint32_t total_ns = 0U;
    uint32_t vec_words = (uint32_t)(C_in / 4);
    int32_t n_px = H * W;

    npu_load_layer(w_packed, vec_words, (uint32_t)C_out, b);

    for (int32_t i = 0; i < n_px; ++i) {
        const uint32_t *act_ptr = (const uint32_t *)&in_hwc[i * C_in];
        uint32_t *out_ptr = (uint32_t *)&out_hwc[i * C_out];
        total_ns += npu_exec_pixel_i8(act_ptr, vec_words, (uint32_t)C_out,
                                      shift_bits, true, out_ptr);
    }
    return total_ns;
}

static void shuffle_concat_2(const int8_t *b1_hwc, const int8_t *b2_hwc,
                             int32_t n_px, int32_t c_branch, int8_t *out_hwc)
{
    int32_t c_out = c_branch * 2;
    for (int32_t p = 0; p < n_px; ++p) {
        const int8_t *s1 = &b1_hwc[p * c_branch];
        const int8_t *s2 = &b2_hwc[p * c_branch];
        int8_t *dst = &out_hwc[p * c_out];
        for (int32_t c = 0; c < c_branch; ++c) {
            dst[2 * c + 0] = s1[c];
            dst[2 * c + 1] = s2[c];
        }
    }
}

static void npu_classify_image(const uint8_t *fb_200x120)
{
    int8_t gap_vec[128] __attribute__((aligned(4)));
    int32_t logits[3];
    uint32_t total_ns = 0U;

    if (!s_npu_ok) return;

    /* 1. Box-filter 200x120 RGB332 -> 96x96x3 signed INT8 */
    npu_downsample_rgb332_to_96x96(fb_200x120, s_img_96x96);

    /* Clear NPU MAC counter */
    NPU_REG_CTRL = NPU_CTRL_CLEAR_MASK;

    /* 2. Conv1 3x3 s2 (96x96x3 -> 48x48x24) on NPU + MaxPool 3x3 s2 -> 24x24x24 */
    total_ns += run_conv1_3x3_s2_npu(s_img_96x96, s_feat_a);
    run_maxpool3x3_s2(s_feat_a, s_feat_b);

    int8_t *cur_in  = s_feat_b;
    int8_t *cur_out = s_feat_a;

    /* 3. Execute all 16 ShuffleNetV2 blocks (Stage 2, Stage 3, Stage 4) */
    for (uint32_t bi = 0; bi < NPU_NUM_BLOCKS; ++bi) {
        const NpuShuffleBlockDesc *d = &k_shuffle_blocks[bi];
        int32_t H = (int32_t)d->h_in;
        int32_t W = (int32_t)d->w_in;
        int32_t C_b = (int32_t)d->c_branch;

        if (d->stride == 2U) {
            int32_t oH = H / 2;
            int32_t oW = W / 2;

            /* Branch 1: DW3x3 s2 (no ReLU) -> PW1x1 (ReLU) */
            run_dw3x3(cur_in, H, W, C_b, 2,
                      &k_dw_wgt_pool[d->b1_dw_wo], &k_dw_bias_pool[d->b1_dw_bo],
                      d->b1_dw_shift, false, s_br2);
            total_ns += run_pw1x1_npu(s_br2, oH, oW, C_b, C_b,
                                      &k_npu_wgt_pool[d->b1_pw_wo], &k_npu_bias_pool[d->b1_pw_bo],
                                      d->b1_pw_shift, s_br1);

            /* Branch 2: PW1x1 (ReLU) -> DW3x3 s2 (no ReLU) -> PW1x1 (ReLU) */
            total_ns += run_pw1x1_npu(cur_in, H, W, C_b, C_b,
                                      &k_npu_wgt_pool[d->b2_pw1_wo], &k_npu_bias_pool[d->b2_pw1_bo],
                                      d->b2_pw1_shift, s_br2);
            run_dw3x3(s_br2, H, W, C_b, 2,
                      &k_dw_wgt_pool[d->b2_dw_wo], &k_dw_bias_pool[d->b2_dw_bo],
                      d->b2_dw_shift, false, s_br3);
            total_ns += run_pw1x1_npu(s_br3, oH, oW, C_b, C_b,
                                      &k_npu_wgt_pool[d->b2_pw2_wo], &k_npu_bias_pool[d->b2_pw2_bo],
                                      d->b2_pw2_shift, s_br2);

            shuffle_concat_2(s_br1, s_br2, oH * oW, C_b, cur_out);
        } else {
            int32_t n_px = H * W;
            int32_t C_in = (int32_t)d->c_in;

            for (int32_t p = 0; p < n_px; ++p) {
                memcpy(&s_br1[p * C_b], &cur_in[p * C_in], (size_t)C_b);
                memcpy(&s_br2[p * C_b], &cur_in[p * C_in + C_b], (size_t)C_b);
            }

            total_ns += run_pw1x1_npu(s_br2, H, W, C_b, C_b,
                                      &k_npu_wgt_pool[d->b2_pw1_wo], &k_npu_bias_pool[d->b2_pw1_bo],
                                      d->b2_pw1_shift, s_br3);
            run_dw3x3(s_br3, H, W, C_b, 1,
                      &k_dw_wgt_pool[d->b2_dw_wo], &k_dw_bias_pool[d->b2_dw_bo],
                      d->b2_dw_shift, false, s_br2);
            total_ns += run_pw1x1_npu(s_br2, H, W, C_b, C_b,
                                      &k_npu_wgt_pool[d->b2_pw2_wo], &k_npu_bias_pool[d->b2_pw2_bo],
                                      d->b2_pw2_shift, s_br3);

            shuffle_concat_2(s_br1, s_br3, n_px, C_b, cur_out);
        }

        int8_t *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
    }

    /* 4. Conv5 1x1 PW (3x3x192 -> 3x3x128) on NPU */
    total_ns += run_pw1x1_npu(cur_in, 3, 3, 192, 128,
                              &k_npu_wgt_pool[NPU_CONV5_WO], &k_npu_bias_pool[NPU_CONV5_BO],
                              NPU_CONV5_SHIFT, cur_out);

    /* 5. Global Average Pooling (3x3x128 -> 128) */
    for (int32_t c = 0; c < 128; ++c) {
        int32_t sum = 0;
        for (int32_t p = 0; p < 9; ++p) {
            sum += (int32_t)cur_out[p * 128 + c];
        }
        int32_t avg = (sum + 4) / 9;
        if (avg < 0) avg = 0;
        else if (avg > 127) avg = 127;
        gap_vec[c] = (int8_t)avg;
    }

    /* 6. Head FC: 128 -> 3 (linear logits) on NPU */
    npu_load_layer(&k_npu_wgt_pool[NPU_FC_WO], 32U, 3U, &k_npu_bias_pool[NPU_FC_BO]);
    total_ns += npu_exec_pixel_i32((const uint32_t *)gap_vec, 32U, 3U, NPU_FC_SHIFT, false, logits);

    s_npu_logits[0] = logits[0];
    s_npu_logits[1] = logits[1];
    s_npu_logits[2] = logits[2];
    s_npu_exec_ns   = total_ns;
    s_npu_mac_cnt   = NPU_REG_MAC_COUNT;
    ++s_npu_infer_cnt;

    /* Pick winner + softmax-approximate confidence */
    uint8_t best_c = NPU_CLASS_CAT;
    int32_t best_v = logits[0];
    if (logits[1] > best_v) { best_c = NPU_CLASS_DOG; best_v = logits[1]; }
    if (logits[2] > best_v) { best_c = NPU_CLASS_UNKNOWN; best_v = logits[2]; }

    int32_t second_v = -30000;
    for (uint8_t c = 0; c < 3U; ++c) {
        if (c != best_c && logits[c] > second_v) {
            second_v = logits[c];
        }
    }
    int32_t margin = best_v - second_v;
    int32_t conf = 52 + (margin / 12);
    if (conf < 50) conf = 50;
    if (conf > 99) conf = 99;

    s_npu_pred_class = best_c;
    s_npu_conf_pct   = (uint8_t)conf;

    const char *cname = (best_c == NPU_CLASS_CAT) ? "CAT" :
                        ((best_c == NPU_CLASS_DOG) ? "DOG" : "UNKNOWN");
    printf("[NPU-SHUFFLE] Classified: %s (%u%%) | logits=[%ld, %ld, %ld] | hw=%lu us (%lu MACs)\r\n",
           cname, (unsigned int)s_npu_conf_pct,
           (long)logits[0], (long)logits[1], (long)logits[2],
           (unsigned long)(s_npu_exec_ns / 1000U), (unsigned long)s_npu_mac_cnt);
}

static void draw_npu_hud_overlay(void)
{
    char top_txt[32];
    char bot_txt[32];
    uint8_t badge_col;
    const char *cls_str;

    if (s_npu_pred_class == NPU_CLASS_CAT) {
        badge_col = 0x1CU; /* Bright green */
        cls_str   = "CAT";
    } else if (s_npu_pred_class == NPU_CLASS_DOG) {
        badge_col = 0xFCU; /* Bright gold/orange */
        cls_str   = "DOG";
    } else {
        badge_col = 0xE3U; /* Bright red/magenta */
        cls_str   = "UNKNOWN";
    }

    /* Top HUD Banner (y = 2..15) */
    fb_fill_rect(2, 2, 196, 13, 0x00U);
    fb_fill_rect(3, 3, 58, 11, badge_col);
    fb_draw_str(6, 5, cls_str, 0x00U, 1);

    snprintf(top_txt, sizeof(top_txt), "CONF:%u%% %luMS",
             (unsigned int)s_npu_conf_pct,
             (unsigned long)((s_npu_exec_ns + 500000U) / 1000000U));
    fb_draw_str(66, 5, top_txt, 0xFFU, 1);

    /* Bottom HUD Score Bars (y = 106..117) */
    fb_fill_rect(2, 106, 196, 12, 0x00U);
    snprintf(bot_txt, sizeof(bot_txt), "CNN#%u", (unsigned int)s_npu_infer_cnt);
    fb_draw_str(5, 109, bot_txt, 0x1FU, 1);

    /* Draw 3 class confidence bars: C (Cat), D (Dog), U (Unknown) */
    int32_t min_l = s_npu_logits[0];
    if (s_npu_logits[1] < min_l) min_l = s_npu_logits[1];
    if (s_npu_logits[2] < min_l) min_l = s_npu_logits[2];
    int32_t max_l = s_npu_logits[0];
    if (s_npu_logits[1] > max_l) max_l = s_npu_logits[1];
    if (s_npu_logits[2] > max_l) max_l = s_npu_logits[2];
    int32_t span = max_l - min_l;
    if (span < 100) span = 100;

    const char labels[3] = { 'C', 'D', 'U' };
    const uint8_t cols[3] = { 0x1CU, 0xFCU, 0xE3U };
    for (uint8_t c = 0; c < 3U; ++c) {
        int16_t bx = (int16_t)(52 + c * 48);
        char lbl[2] = { labels[c], '\0' };
        fb_draw_str(bx, 109, lbl, 0xFFU, 1);
        fb_fill_rect(bx + 8, 110, 34, 5, 0x25U);
        int32_t v = s_npu_logits[c] - min_l;
        int16_t bar_w = (int16_t)(2 + (v * 32) / span);
        if (bar_w > 34) bar_w = 34;
        fb_fill_rect(bx + 8, 110, bar_w, 5, (c == s_npu_pred_class) ? cols[c] : 0x49U);
    }
}

/* ============================================================================
 * Renderers (200x120 RGB332)
 * ============================================================================ */

static void render_plasma(uint32_t phase)
{
    uint32_t y, x;
    uint8_t p1 = (uint8_t)phase;
    uint8_t p2 = (uint8_t)(phase * 2U);

    for (y = 0; y < LCD_FB_H; ++y) {
        uint8_t *row = &s_fb[y * LCD_FB_W];
        uint8_t gy = (uint8_t)((y * 2U + p2) & 0x7FU);
        uint8_t tg = (gy & 0x40U) ? (uint8_t)(63U - (gy & 0x3FU)) : (gy & 0x3FU);
        for (x = 0; x < LCD_FB_W; ++x) {
            uint8_t rx = (uint8_t)((x + p1) & 0x7FU);
            uint8_t tr = (rx & 0x40U) ? (uint8_t)(63U - (rx & 0x3FU)) : (rx & 0x3FU);
            uint8_t bz = (uint8_t)(((x / 2U) - y + p1) & 0x7FU);
            uint8_t tb = (bz & 0x40U) ? (uint8_t)(63U - (bz & 0x3FU)) : (bz & 0x3FU);
            row[x] = (uint8_t)(((tr >> 3) << 5) | ((tg >> 3) << 2) | (tb >> 4));
        }
    }

    s_box_x += s_box_dx;
    s_box_y += s_box_dy;
    if (s_box_x <= 2) { s_box_x = 2; s_box_dx = 2; s_box_col += 0x49U; }
    else if (s_box_x >= (int16_t)(LCD_FB_W - 34)) { s_box_x = (int16_t)(LCD_FB_W - 34); s_box_dx = -2; s_box_col += 0x49U; }
    if (s_box_y <= 2) { s_box_y = 2; s_box_dy = 1; s_box_col += 0x25U; }
    else if (s_box_y >= (int16_t)(LCD_FB_H - 22)) { s_box_y = (int16_t)(LCD_FB_H - 22); s_box_dy = -1; s_box_col += 0x25U; }

    fb_fill_rect(s_box_x, s_box_y, 32, 20, 0xFFU);
    fb_fill_rect(s_box_x + 2, s_box_y + 2, 28, 16, s_box_col | 0x24U);
}

static void render_equalizer(uint32_t phase)
{
    uint32_t b;
    memset(s_fb, s_bg_col, sizeof(s_fb));

    for (uint32_t y = 0; y < LCD_FB_H; y += 20U) {
        memset(&s_fb[y * LCD_FB_W], 0x25U, LCD_FB_W);
    }

    for (b = 0; b < NUM_EQ_BARS; ++b) {
        uint8_t ph1 = (uint8_t)((phase * 2U + b * 6U) & 0x3FU);
        uint8_t ph2 = (uint8_t)((phase + b * 11U) & 0x3FU);
        int32_t v = (s_sin_lut[ph1] + s_sin_lut[ph2]) / 2;
        uint8_t h = (uint8_t)(((v + 64) * 96) / 128 + 6);
        if (h > 110U) h = 110U;

        if (h > s_eq_peaks[b]) s_eq_peaks[b] = h;
        else if (s_eq_peaks[b] > 1U) s_eq_peaks[b] -= 1U;

        int16_t x0 = (int16_t)(4U + b * 10U);
        int16_t y_top = (int16_t)(116U - h);
        for (int16_t y = y_top; y < 116; ++y) {
            uint8_t col = (y < 35) ? 0xE0U : ((y < 70) ? 0xFCU : 0x1CU);
            fb_fill_rect(x0, y, 7, 1, col);
        }
        fb_fill_rect(x0, (int16_t)(116U - s_eq_peaks[b]), 7, 2, 0xFFU);
    }
}

static void render_graph(void)
{
    uint32_t x, y;
    uint8_t bg = s_bg_col;
    uint8_t fg = s_fg_col;
    uint8_t grid_col = (uint8_t)(bg ^ 0x25U);

    memset(s_fb, bg, sizeof(s_fb));

    for (y = 0; y < LCD_FB_H; y += 20U) {
        memset(&s_fb[y * LCD_FB_W], grid_col, LCD_FB_W);
    }
    for (x = 0; x < LCD_FB_W; x += 25U) {
        for (y = 0; y < LCD_FB_H; ++y) {
            s_fb[y * LCD_FB_W + x] = grid_col;
        }
    }

    for (x = 0; x < LCD_FB_W; ++x) {
        int16_t py = (int16_t)(LCD_FB_H - 1U - (s_graph_pts[x] % LCD_FB_H));
        if (s_graph_style == 2U) {
            fb_draw_line_v((int16_t)x, py, (int16_t)(LCD_FB_H - 1U), fg);
        } else if (s_graph_style == 1U) {
            uint8_t fill_col = (uint8_t)((fg >> 1) & 0x6DU);
            fb_draw_line_v((int16_t)x, py, (int16_t)(LCD_FB_H - 1U), fill_col);
            fb_put((int16_t)x, py, fg);
            fb_put((int16_t)x, py - 1, 0xFFU);
        } else {
            int16_t prev_y = (x > 0U) ? (int16_t)(LCD_FB_H - 1U - (s_graph_pts[x - 1U] % LCD_FB_H)) : py;
            fb_draw_line_v((int16_t)x, prev_y, py, fg);
            fb_put((int16_t)x, py, 0xFFU);
        }
    }
}

static void render_calibration(void)
{
    uint32_t x, y;
    memset(s_fb, 0x00U, sizeof(s_fb));

    for (y = 0; y < LCD_FB_H; y += 10U) {
        memset(&s_fb[y * LCD_FB_W], 0x25U, LCD_FB_W);
    }
    for (x = 0; x < LCD_FB_W; x += 10U) {
        for (y = 0; y < LCD_FB_H; ++y) {
            s_fb[y * LCD_FB_W + x] = 0x25U;
        }
    }

    fb_fill_rect(90, 59, 20, 2, 0xFCU);
    fb_fill_rect(99, 50, 2, 20, 0xFCU);
}

/* ============================================================================
 * Dedicated 60 FPS FreeRTOS Render Task
 * ============================================================================ */
static void lcd_render_task(void *arg)
{
    uint32_t phase = 0U;
    TickType_t fps_tick = xTaskGetTickCount();
    uint32_t fps_cnt = 0U;
    uint16_t prev_hw_fc = (uint16_t)(LCD_REG_STATUS >> 16);

    (void)arg;

    for (;;) {
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(16U));

        if (!s_active || s_mtx == NULL) continue;

        if (xSemaphoreTake(s_mtx, pdMS_TO_TICKS(10U)) == pdTRUE) {
            bool did_flush = false;
            phase += s_speed;

            switch (s_mode) {
                case ANIM_MODE_PLASMA:
                    break;
                case ANIM_MODE_EQUALIZER:
                    render_equalizer(phase);
                    flush_fb();
                    did_flush = true;
                    break;
                case ANIM_MODE_GRAPH:
                    if (s_dirty) {
                        render_graph();
                        flush_fb();
                        s_dirty = false;
                        did_flush = true;
                    }
                    break;
                case ANIM_MODE_IMAGE:
                    if (s_dirty) {
                        memcpy(s_fb, s_img_fb, sizeof(s_fb));
                        draw_npu_hud_overlay();
                        flush_fb();
                        s_dirty = false;
                        did_flush = true;
                    }
                    break;
                case ANIM_MODE_CALIBRATION:
                    if (s_dirty) {
                        render_calibration();
                        flush_fb();
                        s_dirty = false;
                        did_flush = true;
                    }
                    break;
                case ANIM_MODE_SOLID:
                    if (s_dirty) {
                        memset(s_fb, s_fg_col, sizeof(s_fb));
                        flush_fb();
                        s_dirty = false;
                        did_flush = true;
                    }
                    break;
                case ANIM_MODE_CAMERA:
                    if (!s_hw_gvsp && s_dirty) {
                        memcpy(s_fb, s_cam_fb, sizeof(s_fb));
                        flush_fb();
                        s_dirty = false;
                        did_flush = true;
                    }
                    break;
                default:
                    break;
            }
            xSemaphoreGive(s_mtx);

            if (did_flush) {
                ++fps_cnt;
            }
        }

        TickType_t now = xTaskGetTickCount();
        if ((now - fps_tick) >= pdMS_TO_TICKS(1000U)) {
            uint16_t hw_fc = (uint16_t)(LCD_REG_STATUS >> 16);
            uint16_t hw_fps = (uint16_t)(hw_fc - prev_hw_fc);
            prev_hw_fc = hw_fc;
            s_fps_meas = (s_mode == ANIM_MODE_PLASMA) ? hw_fps : ((fps_cnt > 0U) ? fps_cnt : hw_fps);
            fps_cnt = 0U;
            fps_tick = now;
        }
    }
}

/* ============================================================================
 * Simple UDP Server Task (Port 5555)
 * ============================================================================ */
static void lcd_udp_task(void *arg)
{
    static uint8_t rx_buf[1460];
    int sock;
    struct sockaddr_in addr, cli_addr;
    socklen_t cli_len;

    (void)arg;

    for (;;) {
        sock = lwip_socket(AF_INET, SOCK_DGRAM, 0);
        if (sock >= 0) break;
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    int rcvbuf = 65536;
    lwip_setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(LCD_UDP_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (lwip_bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        lwip_close(sock);
        vTaskDelete(NULL);
        return;
    }

    printf("[LCD_UDP] Listening on UDP port %u (NPU 5-Stage CNN enabled)\r\n", (unsigned int)LCD_UDP_PORT);

    for (;;) {
        cli_len = sizeof(cli_addr);
        int n = lwip_recvfrom(sock, rx_buf, sizeof(rx_buf), 0,
                              (struct sockaddr *)&cli_addr, &cli_len);
        if (n < 8) continue;

        uint32_t magic = (uint32_t)rx_buf[0]
                       | ((uint32_t)rx_buf[1] << 8)
                       | ((uint32_t)rx_buf[2] << 16)
                       | ((uint32_t)rx_buf[3] << 24);
        if (magic != LCD_PROTO_MAGIC) continue;

        uint8_t cmd = rx_buf[4];
        ++s_udp_rx_cnt;

        if (cmd == LCD_CMD_PING) {
            uint8_t rep[16];
            memset(rep, 0, sizeof(rep));
            rep[0] = 'L'; rep[1] = 'C'; rep[2] = 'D'; rep[3] = '1';
            rep[4] = 0x81U;
            rep[5] = (uint8_t)s_mode;
            rep[6] = s_rot;
            rep[7] = s_show_border ? 1U : 0U;
            rep[8] = s_dim;
            rep[9] = s_speed;
            rep[10] = (uint8_t)s_fps_meas;
            rep[11] = s_graph_style;
            rep[12] = (uint8_t)(s_udp_rx_cnt & 0xFFU);
            rep[13] = (uint8_t)((s_udp_rx_cnt >> 8) & 0xFFU);
            rep[14] = s_npu_pred_class;
            rep[15] = s_npu_conf_pct;
            lwip_sendto(sock, rep, sizeof(rep), 0, (struct sockaddr *)&cli_addr, cli_len);
        } else if (cmd == LCD_CMD_CONFIG && n >= 16) {
            if (xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50)) == pdTRUE) {
                uint8_t m = rx_buf[5];
                if (m < (uint8_t)ANIM_MODE_COUNT) s_mode = (lcd_anim_mode_t)m;
                s_rot = (uint8_t)(rx_buf[6] & 3U);
                s_show_border = (rx_buf[7] != 0U);
                s_dim = (uint8_t)(rx_buf[8] & 3U);
                s_speed = (rx_buf[9] >= 1U && rx_buf[9] <= 20U) ? rx_buf[9] : 2U;
                s_graph_style = (uint8_t)(rx_buf[10] % 3U);
                s_fg_col = rgb_to_332(rx_buf[11], rx_buf[12], rx_buf[13]);
                s_bg_col = rgb_to_332(rx_buf[14], rx_buf[15], (n >= 17) ? rx_buf[16] : 0U);
                apply_ctrl();
                s_dirty = true;
                xSemaphoreGive(s_mtx);
                if (s_render_task_h) xTaskNotifyGive(s_render_task_h);
            }
            uint8_t ack[8] = { 'L', 'C', 'D', '1', 0x82U, (uint8_t)s_mode, s_rot, 0U };
            lwip_sendto(sock, ack, sizeof(ack), 0, (struct sockaddr *)&cli_addr, cli_len);
        } else if (cmd == LCD_CMD_GRAPH && n >= 8) {
            uint16_t cnt = (uint16_t)rx_buf[5] | ((uint16_t)rx_buf[6] << 8);
            uint8_t style = rx_buf[7];
            if (cnt > LCD_FB_W) cnt = LCD_FB_W;
            if ((int)(8U + cnt) <= n) {
                if (xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50)) == pdTRUE) {
                    s_graph_style = (uint8_t)(style % 3U);
                    for (uint16_t i = 0; i < cnt; ++i) {
                        s_graph_pts[i] = rx_buf[8U + i];
                    }
                    if (s_mode != ANIM_MODE_GRAPH) {
                        s_mode = ANIM_MODE_GRAPH;
                        apply_ctrl();
                    }
                    s_dirty = true;
                    xSemaphoreGive(s_mtx);
                    if (s_render_task_h) xTaskNotifyGive(s_render_task_h);
                }
            }
        } else if (cmd == LCD_CMD_IMAGE_CHUNK && n >= 10) {
            uint16_t offset = (uint16_t)rx_buf[5] | ((uint16_t)rx_buf[6] << 8);
            uint16_t len    = (uint16_t)rx_buf[7] | ((uint16_t)rx_buf[8] << 8);
            uint8_t  flags  = rx_buf[9];
            if ((int)(10U + len) <= n && (uint32_t)offset + len <= LCD_FB_SIZE) {
                if (xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50)) == pdTRUE) {
                    memcpy(&s_img_fb[offset], &rx_buf[10], len);
                    if (s_mode != ANIM_MODE_IMAGE) {
                        s_mode = ANIM_MODE_IMAGE;
                        apply_ctrl();
                    }
                    if ((flags & 1U) || (offset + len >= LCD_FB_SIZE)) {
                        npu_classify_image(s_img_fb);
                        s_dirty = true;
                        if (s_render_task_h) xTaskNotifyGive(s_render_task_h);
                    }
                    xSemaphoreGive(s_mtx);
                }
                uint8_t ack[8] = {
                    'L', 'C', 'D', '1',
                    0x84U,
                    (uint8_t)(offset & 0xFFU),
                    (uint8_t)((offset >> 8) & 0xFFU),
                    (uint8_t)((flags & 1U) ? ((s_npu_pred_class << 6) | (s_npu_conf_pct & 0x3FU)) : flags)
                };
                lwip_sendto(sock, ack, sizeof(ack), 0, (struct sockaddr *)&cli_addr, cli_len);
            }
        } else if (cmd == LCD_CMD_NPU_QUERY) {
            uint8_t rep[32];
            memset(rep, 0, sizeof(rep));
            rep[0] = 'L'; rep[1] = 'C'; rep[2] = 'D'; rep[3] = '1';
            rep[4] = 0x87U;
            rep[5] = s_npu_ok ? 1U : 0U;
            rep[6] = s_npu_pred_class;
            rep[7] = s_npu_conf_pct;
            uint16_t ic = s_npu_infer_cnt;
            int32_t l0 = s_npu_logits[0];
            int32_t l1 = s_npu_logits[1];
            int32_t l2 = s_npu_logits[2];
            uint32_t ns = s_npu_exec_ns;
            uint32_t mc = s_npu_mac_cnt;
            memcpy(&rep[8],  &ic, 2);
            memcpy(&rep[12], &l0, 4);
            memcpy(&rep[16], &l1, 4);
            memcpy(&rep[20], &l2, 4);
            memcpy(&rep[24], &ns, 4);
            memcpy(&rep[28], &mc, 4);
            lwip_sendto(sock, rep, sizeof(rep), 0, (struct sockaddr *)&cli_addr, cli_len);
        }
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

void lcd_hud_init(void)
{
    uint32_t magic;

    if (ota_current_slot() == 0U) {
        s_active = false;
        return;
    }

    magic = LCD_REG_MAGIC;
    if (magic != LCD_HUD_MAGIC_VAL && magic != LCD_HUD_MAGIC_V1) {
        s_active = false;
        return;
    }

    /* Mandatory regmap-codegen MAP_HASH verification between firmware and bitstream */
    s_npu_hw_hash = NPU_REG_ID;
    if (s_npu_hw_hash != NPU_MAP_HASH) {
        printf("[NPU] ERROR: RegmapMismatch! HW=0x%08lX != FW=0x%08lX\r\n",
               (unsigned long)s_npu_hw_hash, (unsigned long)NPU_MAP_HASH);
        s_npu_ok = false;
    } else {
        s_npu_ok = true;
        printf("[NPU] Gowin v2 Weight-Stationary CNN NPU Ready (MAP_HASH=0x%08lX verified)\r\n",
               (unsigned long)s_npu_hw_hash);
    }

    s_mtx = xSemaphoreCreateMutex();
    if (s_mtx == NULL) {
        s_active = false;
        return;
    }

    for (uint32_t y = 0; y < LCD_FB_H; ++y) {
        for (uint32_t x = 0; x < LCD_FB_W; ++x) {
            s_img_fb[y * LCD_FB_W + x] = rgb_to_332((uint8_t)(x * 255U / LCD_FB_W),
                                                    (uint8_t)(y * 255U / LCD_FB_H),
                                                    (uint8_t)(255U - x * 255U / LCD_FB_W));
        }
    }
    for (uint32_t x = 0; x < LCD_FB_W; ++x) {
        int32_t sv = s_sin_lut[(x * 2U) & 0x3FU];
        s_graph_pts[x] = (uint8_t)(60 + (sv * 45) / 64);
    }

    if (s_npu_ok) {
        npu_classify_image(s_img_fb);
    }

    s_active = true;
    apply_ctrl();
    s_dirty = true;

    xTaskCreate(lcd_render_task, "lcd_rnd", 1536U, NULL, tskIDLE_PRIORITY + 2U, &s_render_task_h);
    xTaskCreate(lcd_udp_task,    "lcd_udp", 1536U, NULL, tskIDLE_PRIORITY + 3U, NULL);
    printf("[LCD] 60 FPS Render Task + NPU CNN + UDP :%u Ready\r\n", (unsigned int)LCD_UDP_PORT);
}

bool lcd_hud_is_active(void) { return s_active; }

void lcd_hud_tick(uint32_t rx_pkts, uint32_t tx_pkts)
{
    (void)rx_pkts;
    (void)tx_pkts;
}

void lcd_hud_set_pc_state(const char *state, uint8_t color) { (void)state; (void)color; }
void lcd_hud_log_exchange(uint8_t color, const char *proto, const char *msg) { (void)color; (void)proto; (void)msg; }
void lcd_hud_pulse_rx(void) { if (s_active) LCD_REG_PULSE = 1U; }

void lcd_hud_submit_camera_frame(const uint8_t *fb_200x120, uint32_t frame_id, bool auto_switch)
{
    if (!s_active || s_hw_gvsp || fb_200x120 == NULL || s_mtx == NULL) {
        return;
    }
    if (xSemaphoreTake(s_mtx, pdMS_TO_TICKS(5U)) == pdTRUE) {
        memcpy(s_cam_fb, fb_200x120, sizeof(s_cam_fb));
        s_cam_frame_id = frame_id;
        if (auto_switch && s_mode != ANIM_MODE_CAMERA) {
            s_mode = ANIM_MODE_CAMERA;
            apply_ctrl();
        }
        if (s_mode == ANIM_MODE_CAMERA) {
            s_dirty = true;
            if (s_render_task_h) {
                xTaskNotifyGive(s_render_task_h);
            }
        }
        xSemaphoreGive(s_mtx);
    }
}

bool lcd_hud_has_hw_gvsp(void)
{
    if (!s_active) {
        return false;
    }
    return (LCD_REG_HW_RX_PORT & 0xFFFFU) != 0U;
}

void lcd_hud_set_hw_gvsp_mode(bool hw_en, bool mono8)
{
    if (!s_active || s_mtx == NULL) {
        return;
    }
    if (xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50U)) == pdTRUE) {
        s_hw_gvsp = hw_en;
        s_mono8 = hw_en ? mono8 : false;
        LCD_REG_HW_RX_CFG = hw_en ? 1U : 0U;
        if (hw_en) {
            s_mode = ANIM_MODE_CAMERA;
        }
        apply_ctrl();
        s_dirty = !hw_en;
        xSemaphoreGive(s_mtx);
    }
}

uint32_t lcd_hud_get_hw_gvsp_stat(void)
{
    if (!s_active) {
        return 0U;
    }
    return LCD_REG_HW_RX_STAT;
}

void lcd_hud_set_link_status(bool link_ok, uint32_t cam_ip)
{
    (void)link_ok;
    (void)cam_ip;
}

uint32_t lcd_hud_get_hw_gvsp_drops(void)
{
    return 0U;
}

void lcd_hud_anim_next_mode(void)
{
    lcd_hud_anim_set_mode((lcd_anim_mode_t)((s_mode + 1U) % ANIM_MODE_COUNT));
}

void lcd_hud_anim_set_mode(lcd_anim_mode_t mode)
{
    if (mode >= ANIM_MODE_COUNT) mode = ANIM_MODE_PLASMA;
    if (s_mtx && xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50U)) == pdTRUE) {
        s_mode = mode;
        apply_ctrl();
        s_dirty = true;
        xSemaphoreGive(s_mtx);
        if (s_render_task_h) xTaskNotifyGive(s_render_task_h);
    }
}

lcd_anim_mode_t lcd_hud_anim_get_mode(void) { return s_mode; }

const char *lcd_hud_anim_get_mode_name(lcd_anim_mode_t mode)
{
    switch (mode) {
        case ANIM_MODE_PLASMA:      return "Plasma + Bounce";
        case ANIM_MODE_EQUALIZER:   return "Spectrum Equalizer";
        case ANIM_MODE_GRAPH:       return "Live Graph";
        case ANIM_MODE_IMAGE:       return "NPU CNN Classifier";
        case ANIM_MODE_CALIBRATION: return "Calibration Grid";
        case ANIM_MODE_SOLID:       return "Solid Color";
        case ANIM_MODE_CAMERA:      return "Hikrobot Camera";
        default:                    return "Unknown";
    }
}

void lcd_hud_cycle_rotation(void)
{
    if (s_mtx && xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50U)) == pdTRUE) {
        s_rot = (uint8_t)((s_rot + 1U) & 3U);
        apply_ctrl();
        xSemaphoreGive(s_mtx);
    }
}

void lcd_hud_toggle_resolution(void)
{
    if (s_mtx && xSemaphoreTake(s_mtx, pdMS_TO_TICKS(50U)) == pdTRUE) {
        s_mode_800 = !s_mode_800;
        apply_ctrl();
        xSemaphoreGive(s_mtx);
    }
}
