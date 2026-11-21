# 推理执行优化：同权重、同物理预算

本文保留第一轮实验结果。候选能量、约束投影与普通 GPU 聚合的后续实现及最新验证见 [第二轮推理优化](inference_execution_v26.md)。

本轮从六例机器人基线出发，保留 v24 / cloth40 权重、648 维主干、名义每帧三子步，以及每子步两轮解析预测和一次网络校正。没有新增训练，没有减少接触/材料检查，没有调用原生 VBD 或 Featherstone。

## 已实现的执行策略

统一入口增加 `--execution-policy`，默认仍为 `reference`。

| 策略 | 行为 |
|---|---|
| `reference` | 原来的 eager 张量计算 |
| `assembly` | 解析轮不组装未使用的网络关系特征；物理度量只计算需要的弯曲 Jacobian |
| `fused` | 包含 `assembly`，另用 TorchInductor 融合接触本构、单轴关节坐标/力/能量和 mimic 坐标的小算子 |

`fused` 保留 FP64 物理输入与原本的公式；网络仍为 FP32。多轴关节坐标保留原路径；CPU 不启用 CUDA 编译；梯度开启时回到原始组装/算子路径。融合会改变浮点运算顺序，因此不承诺长时间线逐位相同。

```bash
# 相同权重和场景参数，分别选择执行策略；输出目录必须不存在
.venv/bin/python scripts/run_robot_validation.py --scene cloth \
  --execution-policy fused --output outputs/cloth_fused --render
.venv/bin/python scripts/run_robot_validation.py --scene cloth \
  --execution-policy reference --output outputs/cloth_reference

# 六例完整动作与视频
.venv/bin/python scripts/run_robot_demo_suite.py \
  --execution-policy fused --output outputs/my_fused_suite
```

首次遇到某种张量形状、关节布局或步长时需要编译；新接触形状可能再次触发编译。完整录制的 wall time 包含该成本。不能把去掉编译成本的短窗延迟当作完整 demo FPS。

## 局部交错测试

从已完成的本模型录制恢复最终状态和真实接受历史，固定控制，在同一进程交错运行参考/融合版本。每样本推进三个真实物理步，每例五个样本/策略；各策略均三次网络调用、零重试。这里测量求解器，不包含控制器、视频渲染和逐帧序列化。

| 场景 | 参考中位秒数 | 融合中位秒数 | 延迟降低 | 吞吐比 |
|---|---:|---:|---:|---:|
| Franka 软块保持 | 1.6176 | 1.3341 | 17.5% | 1.21× |
| Franka 抓布保持 | 1.8224 | 1.3926 | 23.6% | 1.31× |
| W1 装袋末态 | 2.3015 | 2.1700 | 5.7% | 1.06× |

GPU 同时有其他用户作业，没有暂停它们。小样本与并行负载带来波动，尤其不能把 W1 的 5.7% 当作稳健的全场景结论。软块和布的保存位置在这些短窗中相同，W1 刚体位置最大绝对差约 `2.3e-13`，速度差不超过 `4.7e-10`。

本机原始结果保存在 `.local/inference_speed_v25/fused_franka_v3/`、`fused_cloth/`、`fused_w1_v2/`。可用自己的完整录制重做：

```bash
export WP_VBD_NEWTON_ROOT="$PWD/.local/newton_pinned_25298"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.local/p0/torchinductor"
.venv/bin/python scripts/benchmark_robot_step.py --case franka \
  --recording outputs/my_robot_suite/recordings/franka \
  --output .local/my_interleaved_benchmark --repeats 5 \
  --variants reference fused
```

基准脚本保留冷启动时间、样本、网络/物理步计数、位置/速度差及源码校验摘要；新运行会检查数值误差与预算是否匹配。输入必须是相同权重的本模型完整录制，不能用教师中间状态替代。

## 完整动作与数值一致性的区别

完整六例结果目录为 `outputs/inference_speed_v25/`。逐例任务/几何检查与严格数值回归分开报告。对照脚本保留原定的 0.1 mm 位置、0.1 度姿态和 0.001 速度分量门槛，不因看到结果而放宽。

已完成的 Franka 软块全程 270 帧、810 次网络调用、零重试。相对旧录制，顶点最大位置差约 1.98 mm，刚体位置最大差 0.156 mm，因此**严格数值回归未通过**。差异在夹紧接触后增大；这不能单凭末态短窗归因于某一个算子。独立几何检查中，最大机器人/软体表面重叠约 0.146 mm，最小四面体体积比 0.623，保持阶段整个软块距桌面至少约 113 mm，抓取与保持仍完成。

六例现已完成，共 1602 输出帧、4819 次网络调用，全部零重试、零训练更新，原任务/保存帧几何门槛均通过。Franka 保持整块悬空，W1 袋内包含检查通过，抓布末段宽度/法向一致性为 0.617/0.733，Piper 完成软块堆放，W1 双手没有滑落，刚体放布最终有 4 个支撑接触、布中心新增下垂约 5.5 mm。

| 完整任务 | 当前实测 FPS，含冷启动和记录 | 网络调用 | 原任务检查 |
|---|---:|---:|---|
| Franka 软块 | 0.579 | 810 | PASS |
| W1 装袋 | 0.525 | 877 | PASS |
| Franka 抓布 | 0.756 | 504 | PASS |
| Piper 堆放 | 0.763 | 936 | PASS |
| W1 双手 | 0.704 | 540 | PASS |
| 刚体放布 | 0.632 | 1152 | PASS |

**六例的严格数值回归均未通过**。例如抓布单点最大位置差约 69 mm，整条轨迹顶点坐标 RMS 差约 8.9 mm；它仍通过展开、拉伸、接触和抓取检查，但不能说长轨迹数值等价。完整数值差异保存在 `numerical_comparison.json`，独立任务验收与计时保存在 `performance_summary.json`，同目录保留六例完整视频。

完整计时与历史录制的 CPU/GPU 背景负载不同，本次还让纯 CPU 渲染与后续 GPU 推理重叠。历史 FPS 比值只能描述观察结果，**目前未证明完整 demo 整体显著提速**。默认策略仍为 `reference`，`fused` 通过显式参数选择。

```bash
.venv/bin/python scripts/compare_robot_execution.py \
  --candidate outputs/inference_speed_v25/recordings \
  --report outputs/inference_speed_v25/numerical_comparison.json
```

## 性能判断与下一段工作

调用统计显示，物理组装、受保护粗空间/关节运动方向、解析材料校正、回溯线搜索中存在大量小张量操作。网络响应函数还包含物理投影，不能把这个函数的计时全部算作神经网络耗时。`ExecutionProfile` 已单独标记共享主干，以及物理组装、粗空间、材料校正、线搜索、碰撞和原生状态读写。cProfile/事件插桩会显著增大耗时，范围又有嵌套，不能相加或用作 GPU 忙碌时间比例。

支持额外一次关闭 cProfile 的 NVTX 采样，配合 Nsight 查看实际 CUDA kernel 与同步。该采样在正常交错计时结束后进行，不能拿采样中的耗时替换性能基准。

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o .local/robot_timeline \
  .venv/bin/python scripts/benchmark_robot_step.py --case franka \
  --recording outputs/my_robot_suite/recordings/franka \
  --output .local/robot_timeline_benchmark --repeats 1 \
  --variants reference fused --cuda-profiler
nsys stats --report nvtx_sum,nvtx_kern_sum,cuda_gpu_kern_sum,cuda_api_sum \
  .local/robot_timeline.nsys-rep
```

NVTX 标记分别带 `reference.` / `fused.` 前缀。`wp.complete_window` 包含最终同步，可按时间窗统计本进程所有线程的 kernel；单个嵌套 NVTX 范围的 kernel 关联按发起线程统计，不能把它们简单相加，也不能把范围跨度当作 kernel 忙碌时长。

本机 Nsight 已完成一次 Franka 保持段对照：三个物理步相当于一个输出帧，参考路径发射 212,274 个 CUDA kernel，融合路径 158,900 个，减少约 25.1%。融合路径共享主干三次前向的 kernel 总时间约 10.1 ms，对应 CPU 标注范围累计约 47.6 ms；整帧标注窗口约 2.07 s，仍有大量 Python/算子调度开销。GPU→CPU 拷贝 3184 次，其中 3028 次不超过 8 字节；问题主要在频繁小回读和发射粒度，不能仅以传输字节数判断。

融合路径中，回溯/候选检查范围关联 51,315 个 kernel，物理组装 19,449 个，关节运动方向 13,842 个，解析材料校正 9405 个，受保护基 6464 个。统计有嵌套，仅用于定位，不把它们相加计算占比。这里是单场景采样，含追踪开销与并行负载，kernel 时间和也不是硬件占用率。压缩原始 trace 与摘要在 `.local/inference_speed_v25/nsight_franka.nsys-rep`、`cuda_timeline_summary.json`。

第一轮优化收益有限，尚未解决约 0.7 FPS 的整体问题。下一段优先合并回溯中的候选构造、接触/材料能量和约束投影，保持原来的射线顺序与首个可接受候选语义；随后合并粗空间正交化、重复稀疏矩阵作用和材料校正的连续张量段，批量汇总可合并的小回读。继续保留现有物理预算与全部安全检查。

之后再研究两轮解析预测减为一轮，届时需在更少计算预算和不同步长下微调网络并重验完整动作；直接删迭代、缩网络或减子步不能视为保持效果的优化。

另提供单物理步、版本绑定且有内存上限的可选运动学缓存作对照实验，默认关闭；当前收益不足以推荐为主要策略。缓存不保存残差、材料响应或接受决定。
