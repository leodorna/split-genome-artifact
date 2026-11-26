# Nsight Compute metrics shared by ncu_roofline_v3.sh and ncu_ceilings_v3.sh
# (sourced). Names checked with `ncu --query-metrics --chip gb202`.
#   time, clocks           gpu__time_duration, sm__/dram__cycles_elapsed per second
#   traffic                DRAM read/write bytes, L2 (lts) and L1 bytes, L2 hit rate
#   instructions           warp and thread instructions; per pipe (alu, fma,
#                          fmaheavy = IMAD, xu = MUFU, lsu); FP32 ops for FLOPs
#   utilisation            issue slots, achieved occupancy, pipe busy, top stalls
METRICS="gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,l1tex__t_bytes.sum,\
smsp__inst_executed.sum,smsp__thread_inst_executed.sum,\
smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fmaheavy.sum,\
smsp__inst_executed_pipe_xu.sum,smsp__inst_executed_pipe_lsu.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct"
