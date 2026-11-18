// ============================================================================
// Module      : npu_core (v2.0.0 - Weight-Stationary Multi-Filter CNN NPU)
// Open-Source SIMD-4 INT8 DSP Neural Processing Unit (NPU) for Gowin Arora V
// Clocked on AHB_CLK (100 MHz, 10 ns period).
//
// Features:
//   * Single-source-of-truth register map (regs.toml -> npu_regs.vh)
//   * Read-only ID register at 0x00 returning NPU_MAP_HASH
//   * 32 KB Weight BSRAM (8192 x 32-bit = 32768 INT8 weights) -> stays on-chip per layer!
//   * 256 x 32-bit Bias BSRAM (256 x INT32 channel biases)
//   * 1 KB Activation BSRAM (256 x 32-bit = 1024 INT8 inputs)
//   * Output BSRAM (256 x INT32) + Packed INT8 Output Banks (4 channels per read)
//   * 4-lane parallel signed INT8xINT8 DSP MAC pipeline (400 MMAC/s @ 100 MHz)
//   * Hardware round-half-up, arithmetic shift, ReLU, INT8 clamp, nanosecond timer
// ============================================================================

`timescale 1ns / 1ps

module npu_core (
    input  wire        clk,        // AHB_CLK (100 MHz, 10 ns period)
    input  wire        rst_n,
    input  wire        reg_wr,     // 1-cycle write strobe
    input  wire        reg_rd,     // 1-cycle read strobe
    input  wire [7:0]  reg_addr,   // byte offset 0x00..0xFC (inside 0xF8000900 window)
    input  wire [31:0] reg_wdata,
    output reg  [31:0] reg_rdata
);

`include "npu_regs.vh"

    // ---- On-Chip BSRAM Memories ----
    (* syn_ramstyle = "block_ram" *) reg [31:0] act_mem     [0:255];
    (* syn_ramstyle = "block_ram" *) reg [31:0] wgt_mem     [0:8191];
    (* syn_ramstyle = "block_ram" *) reg [31:0] bias_mem    [0:255];
    (* syn_ramstyle = "block_ram" *) reg [31:0] out_i32_mem [0:255];

    // Packed INT8 output banks (4 banks of 64 bytes = 256 INT8 channels)
    (* syn_ramstyle = "distributed_ram" *) reg [7:0] out_p0 [0:63];
    (* syn_ramstyle = "distributed_ram" *) reg [7:0] out_p1 [0:63];
    (* syn_ramstyle = "distributed_ram" *) reg [7:0] out_p2 [0:63];
    (* syn_ramstyle = "distributed_ram" *) reg [7:0] out_p3 [0:63];

    reg [7:0]  act_wr_ptr;
    reg [12:0] wgt_wr_ptr;
    reg [7:0]  bias_wr_ptr;
    reg [7:0]  out_rd_ptr;

    reg [8:0]  cfg_vec_words;
    reg [8:0]  cfg_out_ch;
    reg        cfg_relu_en;
    reg        cfg_clamp_i8;
    reg [4:0]  cfg_shift_bits;
    reg        cfg_round_en;

    wire act_we  = reg_wr && (reg_addr == ADDR_ACT_DATA4_RAW);
    wire wgt_we  = reg_wr && (reg_addr == ADDR_WGT_DATA4_RAW);
    wire bias_we = reg_wr && (reg_addr == ADDR_BIAS_DATA_RAW);

    reg [7:0]  act_rd_idx;
    reg [12:0] wgt_rd_idx;
    reg [7:0]  oc_idx;

    reg [31:0] act_rd_word;
    reg [31:0] wgt_rd_word;
    reg [31:0] bias_rd_word;
    reg [31:0] out_i32_q;
    reg [31:0] out_pack_q;

    always @(posedge clk) begin
        if (act_we) act_mem[act_wr_ptr] <= reg_wdata;
        act_rd_word <= act_mem[act_rd_idx];
    end

    always @(posedge clk) begin
        if (wgt_we) wgt_mem[wgt_wr_ptr] <= reg_wdata;
        wgt_rd_word <= wgt_mem[wgt_rd_idx];
    end

    always @(posedge clk) begin
        if (bias_we) bias_mem[bias_wr_ptr] <= reg_wdata;
        bias_rd_word <= bias_mem[oc_idx];
    end

    // ---- Control & Pointer Registers ----
    reg        start_pulse;
    reg        clear_pulse;

    wire       out_rd_inc = reg_rd && ((reg_addr == ADDR_OUT_DATA_RAW) || (reg_addr == ADDR_OUT_PACKED_RAW));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_wr_ptr     <= 8'd0;
            wgt_wr_ptr     <= 13'd0;
            bias_wr_ptr    <= 8'd0;
            out_rd_ptr     <= 8'd0;
            cfg_vec_words  <= 9'd64;
            cfg_out_ch     <= 9'd1;
            cfg_relu_en    <= 1'b0;
            cfg_clamp_i8   <= 1'b0;
            cfg_shift_bits <= 5'd0;
            cfg_round_en   <= 1'b0;
            start_pulse    <= 1'b0;
            clear_pulse    <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            clear_pulse <= 1'b0;

            if (out_rd_inc) begin
                out_rd_ptr <= out_rd_ptr + 8'd1;
            end

            if (reg_wr) begin
                case (reg_addr)
                    ADDR_CTRL: begin
                        start_pulse    <= reg_wdata[0];
                        clear_pulse    <= reg_wdata[1];
                        cfg_relu_en    <= reg_wdata[2];
                        cfg_clamp_i8   <= reg_wdata[3];
                        cfg_shift_bits <= reg_wdata[8:4];
                        cfg_round_en   <= reg_wdata[9];
                    end
                    ADDR_DIMS: begin
                        cfg_vec_words  <= (reg_wdata[8:0] == 9'd0) ? 9'd1 :
                                          (reg_wdata[8:0] > 9'd256) ? 9'd256 : reg_wdata[8:0];
                        cfg_out_ch     <= (reg_wdata[24:16] == 9'd0) ? 9'd1 :
                                          (reg_wdata[24:16] > 9'd256) ? 9'd256 : reg_wdata[24:16];
                    end
                    ADDR_ACT_ADDR:      act_wr_ptr  <= reg_wdata[7:0];
                    ADDR_ACT_DATA4_RAW: act_wr_ptr  <= act_wr_ptr + 8'd1;
                    ADDR_WGT_ADDR:      wgt_wr_ptr  <= reg_wdata[12:0];
                    ADDR_WGT_DATA4_RAW: wgt_wr_ptr  <= wgt_wr_ptr + 13'd1;
                    ADDR_BIAS_ADDR:     bias_wr_ptr <= reg_wdata[7:0];
                    ADDR_BIAS_DATA_RAW: bias_wr_ptr <= bias_wr_ptr + 8'd1;
                    ADDR_OUT_ADDR:      out_rd_ptr  <= reg_wdata[7:0];
                    default: ;
                endcase
            end
        end
    end

    // ---- Multi-Filter Pipelined SIMD-4 INT8xINT8 DSP Engine ----
    localparam [2:0] ST_IDLE      = 3'd0;
    localparam [2:0] ST_LOAD_BIAS = 3'd1;
    localparam [2:0] ST_RUN       = 3'd2;
    localparam [2:0] ST_ROUND     = 3'd3;
    localparam [2:0] ST_STORE     = 3'd4;

    reg [2:0]  state;
    reg        status_busy;
    reg        status_done;
    reg        status_ovf;
    reg [8:0]  rem_words;
    reg [8:0]  rem_oc;
    reg [12:0] wgt_base;
    reg [3:0]  pipe_v;

    reg signed [7:0]  a0_s1, a1_s1, a2_s1, a3_s1;
    reg signed [7:0]  w0_s1, w1_s1, w2_s1, w3_s1;

    (* syn_multstyle = "dsp" *) wire signed [15:0] p0_w = a0_s1 * w0_s1;
    (* syn_multstyle = "dsp" *) wire signed [15:0] p1_w = a1_s1 * w1_s1;
    (* syn_multstyle = "dsp" *) wire signed [15:0] p2_w = a2_s1 * w2_s1;
    (* syn_multstyle = "dsp" *) wire signed [15:0] p3_w = a3_s1 * w3_s1;

    reg signed [15:0] m0_s2, m1_s2, m2_s2, m3_s2;
    reg signed [17:0] sum4_s3;
    reg signed [31:0] acc_r;
    reg signed [31:0] acc_shifted_q;
    reg [31:0]        cycle_cnt;
    reg [31:0]        exec_time_ns;
    reg [31:0]        mac_count;

    wire signed [31:0] sum4_ext = {{14{sum4_s3[17]}}, sum4_s3};
    wire signed [32:0] next_acc_wide = {acc_r[31], acc_r} + {sum4_ext[31], sum4_ext};
    wire               acc_ovf_step = (next_acc_wide[32] != next_acc_wide[31]);

    wire signed [31:0] round_bias = (cfg_round_en && (cfg_shift_bits != 5'd0))
                                    ? ($signed(32'sd1) <<< (cfg_shift_bits - 5'd1))
                                    : 32'sd0;
    wire signed [31:0] acc_rounded = acc_r + round_bias;

    wire signed [31:0] post_relu = (cfg_relu_en && acc_shifted_q[31]) ? 32'sd0 : acc_shifted_q;
    wire signed [31:0] post_clamp = cfg_clamp_i8
                                    ? ((post_relu > 32'sd127) ? 32'sd127 :
                                       (post_relu < -32'sd127) ? -32'sd127 : post_relu)
                                    : post_relu;
    wire [7:0] post_i8 = post_clamp[7:0];

    reg        out_we_q;
    reg [7:0]  out_wa_q;
    reg [31:0] out_wd_i32_q;
    reg [7:0]  out_wd_i8_q;

    always @(posedge clk) begin
        if (out_we_q) begin
            out_i32_mem[out_wa_q] <= out_wd_i32_q;
            case (out_wa_q[1:0])
                2'd0: out_p0[out_wa_q[7:2]] <= out_wd_i8_q;
                2'd1: out_p1[out_wa_q[7:2]] <= out_wd_i8_q;
                2'd2: out_p2[out_wa_q[7:2]] <= out_wd_i8_q;
                2'd3: out_p3[out_wa_q[7:2]] <= out_wd_i8_q;
            endcase
        end
        out_i32_q  <= out_i32_mem[out_rd_ptr];
        out_pack_q <= {
            out_p3[out_rd_ptr[5:0]],
            out_p2[out_rd_ptr[5:0]],
            out_p1[out_rd_ptr[5:0]],
            out_p0[out_rd_ptr[5:0]]
        };
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            status_busy   <= 1'b0;
            status_done   <= 1'b0;
            status_ovf    <= 1'b0;
            oc_idx        <= 8'd0;
            rem_oc        <= 9'd0;
            wgt_base      <= 13'd0;
            act_rd_idx    <= 8'd0;
            wgt_rd_idx    <= 13'd0;
            rem_words     <= 9'd0;
            pipe_v        <= 4'b0000;
            a0_s1 <= 8'sd0; a1_s1 <= 8'sd0; a2_s1 <= 8'sd0; a3_s1 <= 8'sd0;
            w0_s1 <= 8'sd0; w1_s1 <= 8'sd0; w2_s1 <= 8'sd0; w3_s1 <= 8'sd0;
            m0_s2 <= 16'sd0; m1_s2 <= 16'sd0; m2_s2 <= 16'sd0; m3_s2 <= 16'sd0;
            sum4_s3       <= 18'sd0;
            acc_r         <= 32'sd0;
            acc_shifted_q <= 32'sd0;
            out_we_q      <= 1'b0;
            out_wa_q      <= 8'd0;
            out_wd_i32_q  <= 32'd0;
            out_wd_i8_q   <= 8'd0;
            cycle_cnt     <= 32'd0;
            exec_time_ns  <= 32'd0;
            mac_count     <= 32'd0;
        end else if (clear_pulse) begin
            state         <= ST_IDLE;
            status_busy   <= 1'b0;
            status_done   <= 1'b0;
            status_ovf    <= 1'b0;
            pipe_v        <= 4'b0000;
            out_we_q      <= 1'b0;
            acc_r         <= 32'sd0;
            cycle_cnt     <= 32'd0;
            exec_time_ns  <= 32'd0;
            mac_count     <= 32'd0;
        end else begin
            out_we_q <= 1'b0;

            if (status_busy) begin
                cycle_cnt <= cycle_cnt + 32'd1;
            end

            case (state)
                ST_IDLE: begin
                    if (start_pulse) begin
                        status_busy <= 1'b1;
                        status_done <= 1'b0;
                        status_ovf  <= 1'b0;
                        oc_idx      <= 8'd0;
                        rem_oc      <= cfg_out_ch;
                        wgt_base    <= 13'd0;
                        cycle_cnt   <= 32'd0;
                        state       <= ST_LOAD_BIAS;
                    end
                end

                ST_LOAD_BIAS: begin
                    // bias_mem[oc_idx] is being registered into bias_rd_word on this clock
                    act_rd_idx <= 8'd0;
                    wgt_rd_idx <= wgt_base;
                    rem_words  <= cfg_vec_words;
                    pipe_v     <= 4'b0000;
                    state      <= ST_RUN;
                end

                ST_RUN: begin
                    // Latch bias on the very first cycle of ST_RUN (when pipe_v == 0 and rem_words == cfg_vec_words)
                    if ((pipe_v == 4'b0000) && (rem_words == cfg_vec_words)) begin
                        acc_r <= $signed(bias_rd_word);
                    end else if (pipe_v[3]) begin
                        acc_r <= acc_r + sum4_ext;
                        if (acc_ovf_step) status_ovf <= 1'b1;
                    end

                    // Stage 0: Feed BSRAM read
                    if (rem_words != 9'd0) begin
                        pipe_v[0]  <= 1'b1;
                        act_rd_idx <= act_rd_idx + 8'd1;
                        wgt_rd_idx <= wgt_rd_idx + 13'd1;
                        rem_words  <= rem_words - 9'd1;
                        mac_count  <= mac_count + 32'd4;
                    end else begin
                        pipe_v[0] <= 1'b0;
                    end

                    // Stage 1: AREG / BREG
                    pipe_v[1] <= pipe_v[0];
                    a0_s1 <= $signed(act_rd_word[ 7: 0]);
                    a1_s1 <= $signed(act_rd_word[15: 8]);
                    a2_s1 <= $signed(act_rd_word[23:16]);
                    a3_s1 <= $signed(act_rd_word[31:24]);
                    w0_s1 <= $signed(wgt_rd_word[ 7: 0]);
                    w1_s1 <= $signed(wgt_rd_word[15: 8]);
                    w2_s1 <= $signed(wgt_rd_word[23:16]);
                    w3_s1 <= $signed(wgt_rd_word[31:24]);

                    // Stage 2: MREG
                    pipe_v[2] <= pipe_v[1];
                    m0_s2 <= p0_w;
                    m1_s2 <= p1_w;
                    m2_s2 <= p2_w;
                    m3_s2 <= p3_w;

                    // Stage 3: PREG
                    pipe_v[3] <= pipe_v[2];
                    sum4_s3   <= ($signed({m0_s2[15], m0_s2}) + $signed({m1_s2[15], m1_s2})) +
                                 ($signed({m2_s2[15], m2_s2}) + $signed({m3_s2[15], m3_s2}));

                    if ((rem_words == 9'd0) && (pipe_v == 4'b0000)) begin
                        state <= ST_ROUND;
                    end
                end

                ST_ROUND: begin
                    acc_shifted_q <= acc_rounded >>> cfg_shift_bits;
                    state         <= ST_STORE;
                end

                ST_STORE: begin
                    out_we_q     <= 1'b1;
                    out_wa_q     <= oc_idx;
                    out_wd_i32_q <= $unsigned(post_clamp);
                    out_wd_i8_q  <= post_i8;

                    if (rem_oc > 9'd1) begin
                        oc_idx   <= oc_idx + 8'd1;
                        rem_oc   <= rem_oc - 9'd1;
                        wgt_base <= wgt_base + {4'd0, cfg_vec_words};
                        state    <= ST_LOAD_BIAS;
                    end else begin
                        status_busy  <= 1'b0;
                        status_done  <= 1'b1;
                        exec_time_ns <= (cycle_cnt << 3) + (cycle_cnt << 1);
                        state        <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // ---- Register Read Mux (registered into prdata by parent APB slave on apb_rd) ----
    always @(*) begin
        case (reg_addr)
            ADDR_ID:             reg_rdata = NPU_MAP_HASH;
            ADDR_CTRL:           reg_rdata = {22'd0, cfg_round_en, cfg_shift_bits, cfg_clamp_i8, cfg_relu_en, 2'b00};
            ADDR_STATUS:         reg_rdata = {29'd0, status_ovf, status_done, status_busy};
            ADDR_DIMS:           reg_rdata = {7'd0, cfg_out_ch, 7'd0, cfg_vec_words};
            ADDR_ACT_ADDR:       reg_rdata = {24'd0, act_wr_ptr};
            ADDR_ACT_DATA4_RAW:  reg_rdata = act_rd_word;
            ADDR_WGT_ADDR:       reg_rdata = {19'd0, wgt_wr_ptr};
            ADDR_WGT_DATA4_RAW:  reg_rdata = wgt_rd_word;
            ADDR_BIAS_ADDR:      reg_rdata = {24'd0, bias_wr_ptr};
            ADDR_BIAS_DATA_RAW:  reg_rdata = bias_rd_word;
            ADDR_OUT_ADDR:       reg_rdata = {24'd0, out_rd_ptr};
            ADDR_OUT_DATA_RAW:   reg_rdata = out_i32_q;
            ADDR_OUT_PACKED_RAW: reg_rdata = out_pack_q;
            ADDR_EXEC_TIME_NS:   reg_rdata = exec_time_ns;
            ADDR_MAC_COUNT:      reg_rdata = mac_count;
            default:             reg_rdata = 32'd0;
        endcase
    end

endmodule
