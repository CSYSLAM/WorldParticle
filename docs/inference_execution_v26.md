# 第二轮执行优化：候选能量、投影与 GPU 聚合

本轮仍用 v24 / cloth40 同一份权重，SHA256 为 `1aac9909419fbb8dda7beef4f0d533050d4ce4191849ebd54ba37dd6a85a2132`。没有训练更新，网络仍是 648 维共享主干、46,884,109 个活跃参数。名义每帧三物理步，每步两轮解析预测和一次网络调用。W1 装袋保留原来的自适应细分。

## 新增实现

- 材料势能、刚体接触势能、加权软接触势能及其真实 owner 分配合并为一个编译段。端口的有限性与姿态检查在该段入口执行。
- 候选位置/姿态更新和整个线性材料路径的方向检查编译执行；仍检查途中坍塌，不能只检查两个端点。
- mimic 的两轮质量投影，以及物理更新中的原始/对偶投影合并张量计算。重复 owner 仍先求和，保留小矩阵秩选择、SPD 检查、修正范围和最终约束误差门槛。
- 原来的射线尝试顺序、首个可接受候选、摩擦上界、材料参数、接受历史及失败重试规则均保留。
- 执行策略在原生状态读入前同步，避免交错基准的首次 roundoff 投影沿用上一策略。

`fused_search` 保留确定性聚合。`fused_fast` 另在无梯度物理调用内允许普通 GPU 并行聚合，并关闭未初始化内存的调试填充；FP64 物理和 FP32 网络精度均保留。它允许浮点求和顺序变化，不能承诺逐位复现。该模式没有关闭实际有限性、接触、约束或变形检查。

这些 PyTorch 设置属于进程全局设置；统一 CLI 使用独立推理进程，设置只在调用期间生效，退出或异常后恢复。模型/场景初始化、外部控制器、记录及梯度开启的训练使用调用者原来的设置。公开默认策略仍是 `reference`。

| 执行策略 | 适用范围 |
|---|---|
| `reference` | 原来的 eager 路径，保留调用者的确定性设置 |
| `assembly` | 解析轮跳过未使用的神经特征组装 |
| `fused` | 第一轮接触本构及关节坐标融合 |
| `fused_search` | 再融合候选能量、运动、路径检查和 mimic 投影，保留调用者的确定性设置 |
| `fused_fast` | 包含上述融合，并在无梯度物理调用内采用普通并行聚合、关闭未初始化内存调试填充 |

同一进程中不要同时运行独立训练线程与快速推理；调用期间的全局设置也会影响其他线程。需要同时训练时使用不同进程。

```mermaid
flowchart LR
    A[原控制器 / 资产及物理参数] --> B[保存调用者的执行设置]
    B --> C[GPU 状态读入与碰撞查询]
    C --> D[两轮解析预测 + 一次共享网络校正]
    D --> E[融合的候选能量 / 投影 / 路径检查]
    E --> F[按原规则接受状态与历史]
    F --> G[恢复调用者设置]
    G --> H[关节读出 / 记录 / 窗口显示]
```

## 检查与局部测时

CPU 与 GPU 的材料/接触能量分配、能量梯度、投影对偶性、重复约束的秩处理，以及大幅修正拒绝均通过检查。包含一个端点均有效、但中途会坍塌的路径，融合版本仍拒绝它。快速模式 GPU 势能误差最大约 `8.3e-17`。另外检查了嵌套调用、异常恢复、训练路径，以及精度设置没有改变。

交错基准每策略七个样本，同进程、同接受状态和历史；每样本三个物理步和三次网络调用，零重试。硬件为 RTX 5090 D v2，PyTorch 2.7.1 / CUDA 12.8。Franka 的帧循环额外执行原 IK 控制器、原生关节读出与逐帧内存记录，并检查各策略的实际控制目标完全相同。它是保持阶段测时，不是完整任务 FPS，也不包含视频渲染和文件压缩。

| 采样 | reference 中位秒 | 上轮 fused 中位秒 | fused_fast 中位秒 | 相对 reference |
|---|---:|---:|---:|---:|
| Franka 保持帧循环 | 1.4524 | 1.2674 | 0.8542 | 1.70× |
| 抓布求解器短窗 | 1.2583 | 1.0935 | 0.7843 | 1.60× |
| W1 装袋求解器短窗 | 1.8969 | 1.6567 | 1.1135 | 1.70× |

Franka 保持帧循环中位数约从 0.689 FPS 到 1.171 FPS。GPU 还有其他进程，绝对数值会波动；只把同进程交错对照用于比较，不用不同时段的旧视频推断加速幅度。

额外拆分试验发现：仅关闭内存调试填充约有 8% 收益，普通并行聚合还有额外收益，因此 `fused_fast` 明确采用后者。相关原始结果在 `.local/inference_speed_v26/atomic_frame_probe/`、`fill_frame_probe/`；它们是消融实验，非默认配置。

## 完整验证与使用

最终 `fused_fast` 候选的六例完整推理已完成，原任务与保存帧几何检查均通过。结果在 `outputs/inference_speed_v26/`，共 1602 输出帧、4819 次网络调用；全部零重试、零训练更新、零原生 VBD/Featherstone 调用、零教师轨迹读取。六例记录中的材料、摩擦、步长、支撑倍率及解析轮数等配置均与原录制一致，逐项结果见 `configuration_comparison.json`。W1 装袋仍为 877 个接受物理步，其余每帧恰好三步；网络调用数均与物理步数一致。

| 完整任务 | FPS，含冷启动与记录 | 网络调用 | 原任务检查 |
|---|---:|---:|---|
| Franka 软块 | 0.804 | 810 | PASS |
| W1 装袋 | 0.741 | 877 | PASS |
| Franka 抓布 | 0.936 | 504 | PASS |
| Piper 堆放 | 1.161 | 936 | PASS |
| W1 双手 | 0.955 | 540 | PASS |
| 刚体放布 | 1.088 | 1152 | PASS |

Franka 保持阶段整个软块距桌面至少约 112 mm，最大爪软体重叠约 0.26 mm。W1 的最终方块包含检查、袋子非刚性变形和持续双侧抓袋接触均通过。抓布末段宽度比/法向一致性为 0.596/0.693，高于原门槛 0.55/0.60，最大爪布重叠约 0.34 mm。Piper 完成堆放，最大爪软体重叠约 1.56 mm，末段最大顶点速度约 0.59 mm/s。W1 双手保持圆柱及软球，最小四面体体积比约 0.969，没有滑落。刚体放布最终有 4 个布面支撑接触，布中心新增下垂约 5.5 mm，刚体已基本静止。

**六例严格数值回归均未通过**，原定门槛仍为 0.1 mm 位置、0.1 度姿态及 0.001 速度分量。Franka 顶点最大位置差约 2.39 mm，W1 装袋约 4.67 mm；抓布最大约 140 mm、全轨迹顶点坐标 RMS 约 20.4 mm。抓布的末帧和阶段预览保持了面板展开，但不能将它称为与旧版数值等价。任务与几何结果保存在 `performance_summary.json` / `audits/`，严格对照保存在 `numerical_comparison.json`。

完整 FPS 不含离线视频渲染，也不是开窗渲染 FPS。Franka 的第一帧约需 50.15 秒，整段 335.74 秒，已经计入首次编译与后续形状编译的成本。保持段热运行的 1.17 FPS 不能替代整段 0.80 FPS。历史录制与本轮背景负载不同；本轮还有一个 CPU 渲染 worker 与后续推理重叠，因此只用前面的同进程交错结果判断局部加速，不据历史 FPS 比值宣布完整任务加速。

```bash
# 单例开窗；默认使用 GPU 和当前统一权重
.venv/bin/python scripts/run_robot_validation.py \
  --scene franka --execution-policy fused_fast --window

# 六例推理串行使用一个 CUDA 槽；纯 CPU 渲染在一个后台 worker 排队
.venv/bin/python scripts/run_robot_demo_suite.py \
  --execution-policy fused_fast --parallel-render --output outputs/my_fast_suite

# 控制器/原生关节读出/逐帧记录都参与的保持段交错测时
export WP_VBD_NEWTON_ROOT="$PWD/.local/newton_pinned_25298"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.local/p0/torchinductor"
.venv/bin/python scripts/benchmark_robot_step.py --case franka \
  --recording outputs/current_weight_demos_20261120/recordings/franka_soft \
  --output .local/my_fast_frame_benchmark --frame-loop --repeats 7 \
  --variants reference fused fused_fast

# 能量、梯度、投影、路径和全局设置的检查
.venv/bin/python scripts/verify_candidate_kernels.py --device cuda:0 \
  --fast-inference --report .local/my_candidate_check.json
.venv/bin/python scripts/verify_inference_settings.py \
  --report .local/my_settings_check.json
```

目录须为新目录。首次编译和新的接触形状会带来冷启动成本。这次没有减少子步、网络调用或原物理检查，性能与任务结论仅覆盖本轮检查的六例。

## GPU 时间线与下一段优化

Nsight 对同一 Franka 保持状态采样了 reference / fused_fast 各三个物理步，即一个名义输出帧。按包含最终同步的完整 CPU 时间窗统计本进程所有线程：kernel 发射从 213,124 个降到 71,097 个，减少约 66.6%；kernel 持续时间之和约从 481 ms 降到 167 ms。标注窗口为 1.708 / 0.918 秒，带追踪和事件开销，不能替代前面的未插桩交错基准，也不能把 kernel 时间和当作硬件占用率。

| 快速路径的标注范围 | 关联 kernel 数 | kernel 时间和 |
|---|---:|---:|
| 物理组装 | 13,406 | 25.1 ms |
| 候选回溯检查 | 11,280 | 18.7 ms |
| 关节运动方向 | 7,560 | 21.4 ms |
| 受保护粗空间基 | 5,272 | 11.3 ms |
| 解析材料校正 | 4,140 | 19.2 ms |
| 三次纯共享主干前向 | 1,659 | 7.6 ms |

这些 NVTX 范围有嵌套，不能相加。共享主干仍不是当前主要耗时来源；减少网络宽度不会解决大量细粒度物理算子调度。

快速路径还有 3184 次 GPU→CPU 拷贝，其中 3028 次不超过 8 字节；reference 对应为 3310 / 3154 次。因此下一段优先批量汇总可合并的有效性回读，把粗空间与材料校正中的连续张量计算继续融合，再研究固定容量/掩码的候选检查以减少主机逐候选决策。仍须保留首个可接受候选的顺序、材料路径与接触检查，并重验完整任务。后续若减少解析轮数或增大步长，需要单独训练与验收。

压缩原始 trace、各范围 CSV 与摘要保存在 `.local/inference_speed_v26/nsight_fast.nsys-rep`、`nsight_summary_*.csv`、`cuda_timeline_summary.json`。六例任务、交错基准与 GPU 摘要也已合并到 `outputs/inference_speed_v26/performance_summary.json`。
