# WorldParticle · Shared VBD Research

当前分支保存跨类型共享动力学模型的研究代码。它在同一套权重下处理软体、布料、刚体、机器人关节关系和 cable/rod，采用共享神经逆响应与有界物理算子协同步进。完整 VBD 仅用于离线教师和独立参考，不参与模型推理、补迭代或失败回退。

原 WorldParticle 的实现保留在 `models/`、`trajectory_model.py` 等文件中，原项目说明见 [原始实现说明](docs/ORIGINAL_WORLDPARTICLE.md)。跨类型共享求解属于本分支新增研究能力。

当前能力与 Newton VBD 的差异、已经成立的价值及未验证的优势，见 [框架评估](docs/framework_assessment.md)。目前尚未建立相对 VBD 的速度或精度优势。

## 当前架构

- 共享特征宽度 **648**，**12** 头注意力，**6** 层编码、**4** 层解码，FFN 宽度 **1296**。
- 每个交互分量以 **128** 个 super-token 为合并目标；保留各对象时可超过这一软预算，同时保留完整分辨率的局部关系通路。
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
| `scripts/run_w1_dual_grasp.py`、`render_w1_dual_grasp.py` | W1 刚体/软体抓取，含无垫块侧夹与三视角渲染 |
| `scripts/run_w1_bag.py`、`render_w1_bag.py` | W1 斜夹柔性袋沿、右手装块、离线三视角渲染 |
| `scripts/verify_w1_bag_recording.py`、`verify_w1_bag_clearance.py` | 实际袋壳内外点、接触采样和原始夹爪可视网格复核 |
| `scripts/run_franka_rigid_cloth.py` | Franka 抓取自由刚体并放到布面，支持 GPU / CPU 和窗口 |
| `scripts/render_franka_model_video.py`、`verify_franka_video.py` | 双视角离线渲染、完整解码与动作阶段检查 |
| `scripts/collect_*.py`、`prepare_*.py` | 离线 VBD 采集、数据转换和资格检查 |
| `scripts/train_normal_shared_648.py` | 多类型混合训练与固定窗口 loss 监控 |
| `scripts/verify_*.py`、`tests/reference_v310/` | 相关物理、梯度及实现对照检查 |
| `assets/live/`、`configs/` | 小型静态资产、范围清单和模型配置 |

本分支包含源码、配置和必要说明，并在独立模型分支提供[一版 step 156 完整模型快照](model_releases/shared_vbd_step156/README.md)，含权重、优化器与 RNG。**不包含训练数据、机器人外部资产、运行日志或视频**。克隆代码时使用 `--single-branch`，需要权重时运行 `python scripts/fetch_model_snapshot.py` 与 `python scripts/verify_model_snapshot.py`；以下命令的权重路径可替换为 `model_releases/shared_vbd_step156/model.pt`，资产仍需另行准备。

```bash
# 默认从接近抓取的位置开始，h=1/600；GPU 网络与物理更新
python scripts/run_franka_soft_grasp.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --window --output /path/to/new_run

# CPU 运行
python scripts/run_franka_soft_grasp.py --device cpu \
  --checkpoint /path/to/checkpoint.pt --window --output /path/to/new_cpu_run

# 自建软块的完整接近、夹紧、抬升时间线
python scripts/run_franka_soft_grasp.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --start-time 0 --frames 240 \
  --output /path/to/full_timeline

# 资产 JSON + 共享权重的原生窗口
python scripts/run_native_asset_live.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --asset assets/live/native_five_families.json
```

新建的刚体放布场景使用同一套共享网络；机械臂通过有限关节驱动跟踪 IK 目标，自由刚体靠接触夹持，布料边缘夹持在支架上。完整动作含接近、抓取、搬运、松手、撤离与静置，共 6.4 秒物理时间。推理目前慢于实时。

```bash
# 指向独立的 Newton 固定版本源码，不需要切换已有 Newton 工作目录
export WP_VBD_NEWTON_ROOT=/path/to/newton
python scripts/run_franka_rigid_cloth.py --device cuda:0 \
  --checkpoint /path/to/checkpoint.pt --output outputs/rigid_cloth

# 可在推理时加 --window；也可对保存状态离线生成双视角视频
python scripts/render_franka_model_video.py \
  --recording outputs/rigid_cloth --output outputs/rigid_cloth.mp4
python scripts/review_franka_recording.py \
  --recording outputs/rigid_cloth --report outputs/rigid_cloth_geometry.json
python scripts/verify_franka_video.py \
  --video outputs/rigid_cloth.mp4 --report outputs/rigid_cloth_video.json
```

W1 双臂轻量场景保留外部机器人的视觉网格和惯量，固定躯干并简化碰撞。每臂 7 个转动关节和夹爪通过关节目标驱动；两个物体保持自由动态。详见 [W1 场景说明](docs/w1_dual_grasp.md)。需要自行准备 `w1-030/robot.urdf` 及其网格，机器人资产不随代码上传。

```bash
python scripts/run_w1_dual_grasp.py --asset /path/to/w1-030 \
  --checkpoint /path/to/checkpoint.pt --output outputs/w1_dual --window
python scripts/render_w1_dual_grasp.py \
  --recording outputs/w1_dual --output outputs/w1_dual.mp4
```

step 156 权重已完成该场景的 1.4 秒闭爪、抬升、保持过程，两物体实际抬升均约 3.6 cm，保存帧未出现四面体翻转。每个交互可在原有 18 列保护空间内请求最多 8 个关节树方向；两个自由物体同组时收缩到最多 6 列，总预算保持 18。默认其他场景仍使用 6 列，网络参数不增加。当前推理仍慢于实时。

运行输出不得重复使用已有结果目录。长任务可通过 `run_franka_vbd_shared.py --resume-recording` 从自身已接受的完整模型帧继续；续算检查权重、Newton 版本、资产、时间步和物理参数，并记录来源，不使用教师状态替换推理状态。

训练入口要求已经通过资格检查的数据清单和初始化权重：

```bash
python scripts/prepare_franka_mixed_course.py \
  --base-manifest /path/to/qualified_base_manifest.json \
  --parent /path/to/parent_checkpoint.pt \
  --dataset /path/to/qualified_franka_data.npz --output /path/to/new_course
python scripts/run_franka_mixed_course.py --plan /path/to/new_course/launch_plan.json
```

W1 纸袋任务使用同一 step 156 权重，完成 4.8 秒夹袋、抓块、装入、撤手及恢复。新版 `--bag-grip oblique` 从上方斜夹袋沿，保留真实袋壁变形；验收覆盖全部 289 个记录状态。袋子为 237 粒子、448 三角形的简化开口纸壳，无袋子自接触，不等同于原 MJVBDV2 场景。材料、数值支撑、弹性形变方向、分段续算和几何检查命令见 [W1 说明](docs/w1_dual_grasp.md)。当前单个长任务仍需离线计算；影片播放帧率不代表推理 FPS。

## 环境与外部源码

当前实测环境为 Python 3.12、PyTorch 2.7.1+cu128、Warp 1.17.0、NumPy 1.26.4、trimesh 5.1.1。Newton 固定在 `25298aea1691e060c86f95e8d2b773c898c006d3`，不在本仓库中。

当前 Franka、训练和官方入口通过 `WP_VBD_NEWTON_ROOT` 选择 Newton 源码，并记录真实版本；未设置时默认 `/home/oem/code/engine/newton`。推理、续算、几何复核与渲染应使用同一固定版本。部分早期资产工具仍引用本机路径；迁移机器时需要调整。`environment.yaml` 对应原 WorldParticle，不是新共享求解器的完整环境锁。资产读取补充依赖可安装：

```bash
python -m pip install -r requirements-vbd-assets.txt
```

## 当前结果和限制

- 当前纯 VBD 范围为 **42 个官方 demo**，其余 **9 个外部求解器耦合场景**暂时排除；清单见 `configs/pure_vbd_demo_scope.json`。
- 2478 检查点在自建低分辨率软块场景中，从独立、无初始接触的接近抓取位置开始，用 GPU 推理抬升约 **9.9 cm** 并保持 **0.5 s**。这不是完整官方 Franka demo 的验收。
- 同配置下，`h=1/600` 能抓起，`h=1/60` 未抓起。1.95 s 物理过程原耗时约 978 s，**尚不具备实时速度**。
- 新课程保留 69 组旧数据并加入 1 组 Franka 数据，已完成 **156 次**更新，70 组数据均获得实际参数更新；新增 Franka 获得 18 次更新。记录中 155 次具有正梯度并改变了跟踪参数标量，1 次为零梯度。用户要求暂停时，最终开发集复测尚未完成，未产生合格的新导出权重。
- Franka 固定窗口 loss：`0.03910 → 0.03944 → 0.03988`，目前**没有证据表明这轮扩训改善或收敛**。短窗口检查不代替完整轨迹、材料泛化或视觉验收。
- 静态拓扑缓存已通过 CPU / GPU 对照；表面接触分组全部接受后结束无效的剩余线搜索，并保留未接受分组的完整候选预算。小窗口测得约 1.39 倍提速，不能外推为全部场景的速度。CUDA Graph 未在该小场景获得提速，BF16 的差异更大，默认仍为 FP32 / eager。
- step 156 的共享权重已跑通自建软块的完整 4 秒动作，抬升约 10.1 cm；这项任务结果不代表新的训练收敛认证。
- 同一权重已在新建 Franka + 自由刚体 + 81 粒子布料场景运行完整 6.4 秒动作：抬升约 9.1 cm、横移约 25.9 cm，松手后布料额外下垂约 5.5 mm，刚体末态稳定。385 个保存状态的离散检查中，刚体接触最大重叠约 5.16 mm（取物台上的夹取初段），刚体—布料最大重叠约 0.044 mm。该结果由自身模型帧续算完成；没有调用 VBD / Featherstone 推进或读取教师轨迹。
- 原有 12 列粗空间保留；新场景显式启用最多 18 列的独立对象模式，容纳至多两个自由对象的各 6 列与至多 6 列关节树方向，不增加神经参数。当前不是任意数量对象的机器人耦合认证。

不要把本分支视为已完成所有 demo 替代、完全无穿透或实时性能认证的发行版。
