# WorldParticle · Shared VBD Research

当前分支保存跨类型共享动力学模型的研究代码。它在同一套权重下处理软体、布料、刚体、机器人关节关系和 cable/rod，采用共享神经逆响应与有界物理算子协同步进。完整 VBD 仅用于离线教师和独立参考，不参与模型推理、补迭代或失败回退。

原 WorldParticle 的实现保留在 `models/`、`trajectory_model.py` 等文件中，原项目说明见 [原始实现说明](docs/ORIGINAL_WORLDPARTICLE.md)。跨类型共享求解属于本分支新增研究能力。

## 当前架构

- 共享特征宽度 **648**，**12** 头注意力，**6** 层编码、**4** 层解码，FFN 宽度 **1296**。
- 最多 **128** 个 super-token，同时保留完整分辨率的局部关系通路。
- 活跃神经参数 **46,882,811**；每个物理子步执行 **4** 轮共享更新。
- 粒子使用平移自由度；刚体与 rod 段使用质心位置和姿态；关节、材料和接触以关系编码。材料、摩擦、质量/惯量和驱动参数进入物理算子及共享条件特征。

详细结构、物理边界和验证范围见 [当前架构](docs/CURRENT_ARCHITECTURE.md)，实现配置见 [shared_vbd_648_current.json](configs/shared_vbd_648_current.json)。原始研究目标保留在 [v4.1 方案](docs/plans/WorldParticle_VBD_Replacement_Plan_v4.1.md)。

## 代码与运行入口

| 路径 | 用途 |
|---|---|
| `src/wp_vbd/` | 共享网络、材料/关节/接触算子、历史状态和 Newton 适配器 |
| `scripts/run_native_asset_live.py` | 输入资产 JSON、物理参数和权重，打开 Newton 窗口 |
| `scripts/run_official_live.py` | 官方 demo 的共享模型推理入口 |
| `scripts/run_franka_soft_grasp.py` | 自建单求解器 Franka + 软块场景 |
| `scripts/collect_*.py`、`prepare_*.py` | 离线 VBD 采集、数据转换和资格检查 |
| `scripts/train_normal_shared_648.py` | 多类型混合训练与固定窗口 loss 监控 |
| `scripts/verify_*.py`、`tests/reference_v310/` | 相关物理、梯度及实现对照检查 |
| `assets/live/`、`configs/` | 小型静态资产、范围清单和模型配置 |

本分支只包含源码、配置和必要说明，**不包含权重、训练数据、运行日志、视频或实验报告**。因此，下列命令需要本地准备相应权重和资产，不能仅凭克隆源码复现已有检查点。

```bash
# 默认从接近抓取的位置开始，h=1/600；GPU 网络与物理更新
python scripts/run_franka_soft_grasp.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --window --output /path/to/new_run

# CPU 运行
python scripts/run_franka_soft_grasp.py --device cpu \
  --checkpoint /path/to/checkpoint.pt --window --output /path/to/new_cpu_run

# 完整命令时间线入口；完整接近动作的长程质量仍待验证
python scripts/run_franka_soft_grasp.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --start-time 0 --frames 240 \
  --output /path/to/full_timeline

# 资产 JSON + 共享权重的原生窗口
python scripts/run_native_asset_live.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --asset assets/live/native_five_families.json
```

训练入口要求已经通过资格检查的数据清单和初始化权重：

```bash
python scripts/prepare_franka_mixed_course.py \
  --base-manifest /path/to/qualified_base_manifest.json \
  --parent /path/to/parent_checkpoint.pt \
  --dataset /path/to/qualified_franka_data.npz --output /path/to/new_course
python scripts/run_franka_mixed_course.py --plan /path/to/new_course/launch_plan.json
```

## 环境与外部源码

当前实测环境为 Python 3.12、PyTorch 2.7.1+cu128、Warp 1.17.0、NumPy 1.26.4、trimesh 5.1.1。Newton 固定在 `25298aea1691e060c86f95e8d2b773c898c006d3`，不在本仓库中。

现有研究脚本默认使用 `/home/oem/code/engine/newton`，部分资产工具也引用该 Newton 环境；迁移机器时需要调整这些路径。`environment.yaml` 对应原 WorldParticle，不是新共享求解器的完整环境锁。资产读取补充依赖可安装：

```bash
python -m pip install -r requirements-vbd-assets.txt
```

## 当前结果和限制

- 当前纯 VBD 范围为 **42 个官方 demo**，其余 **9 个外部求解器耦合场景**暂时排除；清单见 `configs/pure_vbd_demo_scope.json`。
- 2478 检查点在自建低分辨率软块场景中，从独立、无初始接触的接近抓取位置开始，用 GPU 推理抬升约 **9.9 cm** 并保持 **0.5 s**。这不是完整官方 Franka demo 的验收。
- 同配置下，`h=1/600` 能抓起，`h=1/60` 未抓起。1.95 s 物理过程原耗时约 978 s，**尚不具备实时速度**。
- 新课程保留 69 组旧数据并加入 1 组 Franka 数据，已完成 **156 次**更新，70 组数据均获得实际参数更新；新增 Franka 获得 18 次更新。记录中 155 次具有正梯度并改变了跟踪参数标量，1 次为零梯度。用户要求暂停时，最终开发集复测尚未完成，未产生合格的新导出权重。
- Franka 固定窗口 loss：`0.03910 → 0.03944 → 0.03988`，目前**没有证据表明这轮扩训改善或收敛**。短窗口检查不代替完整轨迹、材料泛化或视觉验收。
- 最新静态拓扑缓存已通过 CPU 正向、反向和失效条件对照；CUDA Graph、BF16 及精简数值诊断是实验性推理选项，GPU 等价性、速度和完整轨迹尚待验证，默认保留 FP32。
- 关节—材料运动粗空间的当前 Franka 路径要求每个交互分量只有一个材料对象；不能据此宣称已支持机器人同时抓取任意多个软体、布料和 cable。

不要把本分支视为已完成所有 demo 替代、完全无穿透或实时性能认证的发行版。
