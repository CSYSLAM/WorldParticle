# WorldParticle · Shared VBD Research

本分支保存跨类型共享动力学模型的研究实现。共享网络处理软体、布料、刚体、关节与 cable/rod；解析物理算子与网络协同推进。原 WorldParticle 实现仍在 `models/`、`trajectory_model.py` 等文件，说明见 [原始项目](docs/ORIGINAL_WORLDPARTICLE.md)。跨类型共享属于新增研究能力。

当前发布版为 **v24 / cloth fine-tune step40**。同一份权重完成六个机器人任务，名义每帧三子步，每物理步两轮解析预测和一次网络校正。共享特征 648 维、12 头注意力、6 层编码 / 4 层解码，活跃参数 46,884,109。详细结构见 [当前架构](docs/CURRENT_ARCHITECTURE.md)。

本分支还保存最新 **v32 / A1+N1 / step64 研究快照**，复用共享特征学习材料补偿，活跃参数 47,137,509。六项固定 GPU 任务已完成各自检查；Piper 局部单元压缩仍需复核，因此默认稳定入口保持 v24。最新权重与精确参数见 [v32 模型目录](model_releases/shared_vbd_v32_a1/README.md)，实际覆盖与材料质量边界见 [六例扩展验证](docs/v32_six_scene_validation.md)。

## 模型与环境

[当前模型目录](model_releases/shared_vbd_v24_cloth40/README.md)包含精确配置、校验摘要和六个不超过 32 MiB 的权重分块；与本次源代码一起提交。完整权重 188,146,967 字节，SHA256 `1aac9909419fbb8dda7beef4f0d533050d4ce4191849ebd54ba37dd6a85a2132`。

```bash
git clone --depth 1 --single-branch --branch research/shared-vbd-v32-20261121 \
  https://github.com/CSYSLAM/WorldParticle.git
cd WorldParticle
# 在已准备好的 Python / Torch / Warp 环境中执行
python -m pip install -r requirements-vbd-assets.txt
python scripts/fetch_model_snapshot.py
python scripts/verify_model_snapshot.py
```

恢复并运行最新研究快照（不改变稳定版默认值）：

```bash
python scripts/fetch_model_snapshot.py --snapshot model_releases/shared_vbd_v32_a1
python scripts/verify_model_snapshot.py --snapshot model_releases/shared_vbd_v32_a1
python scripts/run_robot_validation.py --scene cloth --recipe configs/robot_validation_v32.json --window
python scripts/run_robot_demo_suite.py --recipe configs/robot_validation_v32.json \
  --output outputs/my_v32_six_demos --parallel-render
```

v32 检查点为六个不超过 32 MiB 的普通 Git 分块，原始 SHA256 为 `b73dbd02f92adcf7ee9b82587fa937c967b1d8f165ecb1974d10cc206c4b5c60`。本发布提供完整推理权重、配置和训练摘要；配套优化器/RNG 状态保留在原工作区，未重复放入 Git。外部资产和原始训练数据不随权重发布，因此不宣称提供精确训练复现或续跑。

测试环境为 Python 3.12、Torch 2.7.1+cu128、Warp 1.17.0。使用独立 Newton checkout，修订固定为 `25298aea1691e060c86f95e8d2b773c898c006d3`；不要切换用户正在使用的其他 Newton 工作目录。

```bash
export WP_VBD_NEWTON_ROOT=/path/to/pinned/newton
```

未指定环境变量时，机器人统一入口使用 `.local/newton_pinned_25298`。W1 默认资产 `~/下载/assets/w1-030/robot.urdf`，Piper 默认资产 `~/下载/assets/scene/松灵 Piper 机械臂-urdf/piper/piper.urdf`；其他机器可通过 `--asset` 指定。机器人资产与训练数据不包含在源码发布中。

## 六个机器人 demo

```bash
# 选一个场景，完整 GPU 推理后渲染视频
python scripts/run_robot_validation.py --scene cloth --render
# 边推理边开窗（显示速度取决于推理速度）
python scripts/run_robot_validation.py --scene w1 --window
# 六例视频集中输出；目录须为新目录
python scripts/run_robot_demo_suite.py --output outputs/my_robot_suite
```

| `--scene` | 内容 | 名义子步/帧 | 已测整体 FPS |
|---|---|---:|---:|
| `franka` | 夹软块、抬起、保持 | 3 | 0.74 |
| `w1` | 左手持柔性袋，右手装块 | 3 | 0.46 |
| `cloth` | 从上方夹起自由布角 | 3 | 0.92 |
| `piper` | 抓软块放到自由刚体块上 | 3 | 0.74 |
| `w1_dual` | 无垫块侧夹刚体圆柱和软球 | 3 | 0.67 |
| `franka_rigid_cloth` | 抓刚体、搬运、松爪放到布上 | 3 | 0.70 |

六例均完成时间线和相应任务/保存帧检查，零拒绝重试。W1 装袋有少量接受步自适应细分，实际网络调用 877 次；其余严格每输出帧三次网络调用。计时包括网络、物理、碰撞、控制、同步和记录，排除离线视频渲染；不是单独神经前向吞吐，也不是隔离负载的性能基准。

[六例说明和复现](docs/robot_demo_suite.md) · [抓布微调与消融](docs/robot_cloth_finetuning.md) · [Piper](docs/piper_soft_place.md) · [发布验证记录](model_releases/shared_vbd_v24_cloth40/validation_summary.json)

最初两轮同权重执行优化保留了 `reference` 默认。策略 `--execution-policy fused_fast` 融合候选能量、运动及约束投影，并在物理调用内允许普通 GPU 并行聚合；它保留 FP64 物理精度与原来的检查，但不保证逐位复现。局部交错测试、完整回放及使用方法见 [第二轮推理优化](docs/inference_execution_v26.md)，第一轮结果见 [推理执行优化](docs/inference_execution_optimization.md)。完整任务与热运行短窗的计时范围不同，不能直接换算。

第三轮候选 `fused_device` 继续融合稀疏响应、小矩阵及运动检查，并复用版本化拓扑索引。三个受控短窗对上一版耗时降低约 16%～24%；六例首次完整回放五例通过任务检查，抓布的外观检查在完整重复中有波动，因此尚未提升为默认。完整推理仍约 1 FPS，设计、边界与结果见 [第三轮推理优化](docs/inference_execution_v27.md)。

可选 `fused_graph` 使用有界 CUDA Graph，原理与早期结果见 [有界 CUDA Graph 推理](docs/inference_execution_v29.md)。后续复核修正了旧 Franka 热运行脚本的控制器频率，因此旧 2.25 FPS 只适用于旧测试频率；正式完整录制不受影响。

最新一轮在图路径合并物理检查与拓扑元数据传输，省去未使用的关系槽位，并接入单步纯运动学缓存。现已默认采用 `auto`：CUDA 的 Franka 软块、单角抓布和 W1 装袋自动选择 `fused_graph`；CPU 和其余三个场景保留 `reference`，显式策略参数优先。同条件七样本短窗对前一版图配置耗时降低约 2.1% / 4.2% / 3.2%（Franka / 抓布 / W1）；Franka 保持段约 2.27 FPS，包含预热的完整计时约 1.19 FPS，仍未实时。实现、测时边界与完整任务结果见 [同步与运动学优化](docs/inference_execution_v30.md) 和 [结果目录](outputs/inference_speed_v30/README.md)。

材料补偿学习已实现并完成两阶段实验：v31 冻结主干训练材料读出头，v32 联合更新主干/材料读出并将解析前缀从两轮降为一轮。64 次更新的开发集目标降低约 18.6%；同状态交错测量帧率约提高 24%，不等于所有完整任务或原生 VBD 比较中的加速。源代码审计、调用计数、模块监督和预算对照见 [可学习物理模块评估](docs/learned_physics_replacement_v31.md)、[v31](docs/shared_material_correction_v31.md) 和 [v32](docs/shared_material_prefix_v32.md)。该研究路线未更换默认推理权重，未完成全部 42 个官方 demo 的重新混训和统一质量回归。

## 研究边界

原生 VBD 仅用于离线教师和独立参考，模型推理不调用 VBD、Featherstone 或原生求解器回退。材料本构、接触、关节、解析预测、约束检查等物理计算仍然存在。当前不宣称纯单次前向替代整个求解器，也没有证明相对原生 VBD 的通用精度或速度优势。

本次布料微调来自 6 条完整训练轨迹、1 条独立材料开发轨迹及既有类型回放，40 次更新。固定开发窗口 loss 下降 15.4%；没有全量重验全部 42 个官方 VBD demo。发布检查点含权重、配置和训练摘要，不含优化器/RNG，不能直接宣称精确续训。

## 代码入口

- `src/wp_vbd/`：共享网络、解析材料/接触/关节算子、接受历史及 Newton 适配。
- `scripts/run_robot_validation.py`、`run_robot_demo_suite.py`：单场景和六例批量入口。
- `scripts/collect_vbd_augmentation.py`、`collect_robot_cloth_substeps.py`：离线 VBD 教师采集。
- `scripts/train_analytic_corrector.py`、`train_robot_cloth_corrector.py`：共享主干训练与定向微调。
- `scripts/verify_*.py`：有物理/梯度意义的契约检查；临时日志不入库。
- `configs/robot_validation_cloth_ft.json`：六例的权重摘要、动作和数值策略。

历史 step156 快照说明保留在 `model_releases/shared_vbd_step156/`，其权重可从原有冻结标签恢复。当前默认入口只使用 v24 新权重。
