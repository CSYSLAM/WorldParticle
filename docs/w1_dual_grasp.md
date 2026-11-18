# W1 双臂抓取

使用外部资产 `~/下载/assets/w1-030/robot.urdf`，左手夹黄色刚体，右手夹蓝色软体。机器人从物体两侧开始，闭爪、抬升、保持。控制器只发送关节目标；双臂、手指和两个物体的运动均由共享网络推理路径推进。

在已有 Torch、Warp 和固定版本 Newton 环境中，运行 `python -m pip install -r requirements-vbd-assets.txt` 补齐资产、IK 与视频依赖。权重和机器人网格需另行准备。

```bash
cd /home/oem/code/worldparticle
export WP_VBD_NEWTON_ROOT="$PWD/.local/newton_pinned_25298"
.venv/bin/python scripts/run_w1_dual_grasp.py \
  --asset ~/下载/assets/w1-030 \
  --output .local/w1_dual_test --window
```

输出目录必须是新目录；去掉 `--window` 可以离线录制。同一仓库一次运行一个 CUDA 任务。默认使用 `checkpoints/mixed_course/last.pt`，保持现有共享模型权重。

录制后渲染全景、左手、右手三个视角：

```bash
.venv/bin/python scripts/render_w1_dual_grasp.py \
  --recording .local/w1_dual_test \
  --output .local/w1_dual_test/w1.mp4
```

换成左侧刚体圆柱、右侧四面体软体球，并提高摩擦系数：

```bash
.venv/bin/python scripts/run_w1_dual_grasp.py \
  --objects cylinder_ball --friction 2.5 --frames 144 \
  --output .local/w1_round_test
.venv/bin/python scripts/render_w1_dual_grasp.py \
  --recording .local/w1_round_test --output .local/w1_round_test/w1_round.mp4
```

圆柱半径、高度的一半及软球半径均为 2.5 厘米。球使用 43 个节点和 80 个正体积四面体。上述命令完成 2.4 秒抓取、抬升和持续夹持；抬起后保持原有闭合目标，摩擦系数全程保持不变。球和圆柱的位姿、速度始终由模型推理路径推进。另有可选 `--release` 供主动松爪试验使用，不用于持续夹持验收。

抬高桌面并水平前伸、向后取回：

```bash
.venv/bin/python scripts/run_w1_dual_grasp.py \
  --style front --objects cylinder_ball --friction 2.5 \
  --output .local/w1_front_test
.venv/bin/python scripts/render_w1_dual_grasp.py \
  --recording .local/w1_front_test --output .local/w1_front_test/w1_front.mp4
```

`front` 桌面高 1.08 米，托座高 1.12 米；夹爪水平朝前，并滚转为上下对夹。托座中央留槽，初始下爪已接近物体下方；机器人从近抓取位置前伸 0.5 厘米、闭合、抬升 9 厘米、向机器人方向回收 10 厘米、保持，默认时长 2.8 秒。初始每指开度 28 毫米，捕获目标 24 毫米；0.8–1.2 秒将目标缓调至 26.5 毫米，降低持续挤压的预紧力并保持支撑。回收段使用多个连续 IK 目标。标注的位移为控制目标，实际物体位移以 `quality.json` 为准。该姿态不复用旧俯抓轨迹；同条件的自有前抓记录可通过 `--resume-grasp` 和 `--resume-frame` 恢复，并核对控制前缀及首个续算帧。验收同时要求抬离、保持和实际向后回收超过 4 厘米。

水平姿态使用关节结构刚度 1e6、机械臂驱动刚度 60000、手指驱动刚度 4000，并把软接触数值切向支撑倍数设为 64，降低关节下垂与有限迭代中的接触滑移。摩擦系数仍为 2.5，物理 Coulomb 上限不变；默认俯抓场景的这些数值保持原值。

水平夹取的软球采用 μ=5000 Pa、λ=500000 Pa：保留较低的剪切刚度，并提高体积刚度，避免夹持时局部体积塌缩。可用 `--soft-lambda` 显式修改；原俯抓默认仍为 λ=5000 Pa。原生 Float32 状态读回时只对精度界内的联动约束误差进行两次有界投影，且不修改输入数组、粒子或速度。前向更新仍为每子步 4 轮，接触回溯最多检查 13 个固定步长。水平场景对保护空间与网络补空间分别进行物理曲率步长检查，再检查合成步长，避免补空间的高曲率把整体修正过度缩小；不改变网络权重，不调用原生动力学求解器。对双精度中幅度低于 64 倍机器精度、却因舍入显示负下降功的方向，显式归零；有实际幅度的上坡方向、负曲率和非有限值仍被拒绝。

侧向夹取、物体直接放在桌面上：

```bash
.venv/bin/python scripts/run_w1_dual_grasp.py \
  --style side --objects cylinder_ball --friction 2.5 \
  --output .local/w1_side_test
.venv/bin/python scripts/render_w1_dual_grasp.py \
  --recording .local/w1_side_test --output .local/w1_side_test/w1_side.mp4
```

`side` 将夹爪绕前伸轴旋转 90°，闭合方向改为水平左右对夹，完全移除托座。桌面仍高 1.08 米，物体靠近桌面朝机器人一侧的边缘；手掌留在桌外，手指水平前伸 4 厘米后夹取。手指目标从每指 28 毫米闭合至 22 毫米并保持。随后抬升 9 厘米、回收 10 厘米，再停手保持 0.4 秒，总时长 3 秒。

`verify_w1_table_approach.py` 用原始机器人可视网格检查目标轨迹，并核对夹爪闭合轴确为水平。运行器进一步检查实际录制的每一帧：将原始网格三角形裁切到桌面的 XY 范围，计算最小离桌高度，包含跨桌边但顶点不在桌内的三角形。`table_visual_clearance.json` 保存各帧结果；大于 1 毫米的可视网格穿桌会使最终验收失败。该检查不修改动力学状态。

侧夹对刚体与软体均使用 64 倍数值切向支撑，减小有限迭代的静摩擦蠕滑；物理摩擦系数、法向接触材料与 Coulomb 上限保持原值。可通过 `--rigid-tangent-support-scale` 修改刚体支撑倍数，其他姿态默认仍为 1。自有记录续算时，若数值支撑配置变化，先按原配置核对一个续算帧，再切换新数值配置，过程写入 `grasp_continuation.numerical_friction_support_transition`。旧 `front` 模式仍可复现上一版上下对夹和托座。

原始机器人文件不修改、不随代码分发。场景保留原始视觉网格和惯量，固定底盘、躯干和头部，手臂碰撞用局部包围盒，手指用下部夹持区域包围盒，关闭机器人自碰撞。默认软体为 27 个节点、40 个四面体的低分辨率方块。它用于简单抓取验证，不能代表完整机器人碰撞精度。

默认两件物体均为边长 5 厘米、密度 200 kg/m³ 的方块。软体的 Lamé 参数均为 5000 Pa，默认接触摩擦系数为 1.5。近抓取初始开度为每指 28 毫米，闭合目标为每指 20 毫米；实际开度取决于有限刚度驱动、接触和软体形变。手臂和手指驱动刚度分别为 20000、2000。抬升目标为 8 厘米，实际物体抬升由录制结果测量。

该例在已有 18 列保护空间内为单个交互请求最多 8 个关节树方向，覆盖每臂 7 个转动关节及 1 个独立夹爪坐标。如果同一交互包含两个自由物体，关节树方向会收缩到最多 6 列，保持总预算。其他场景仍默认 6 列，不改变共享网络参数。

默认物理时长 1.4 秒，60 帧/秒录制，每帧 10 个子步。网络和物理张量计算使用 CUDA；视频采用 CPU 离线渲染，以 0.25 倍速回放。当前推理速度仍低于实时。

`run.json` 记录权重、Newton 版本、实际网络调用、原生求解器调用计数及进度；`trajectory.npz` 保存状态；`quality.json` 从真实录制状态计算两物体抬升和保持阶段最小离台高度。运行结束和抓取成功分别记录，失败过程不会被当成成功结果。正常运行禁止构造或调用原生 VBD、Featherstone 求解器。

2026-11-16 验证结果：现有 step 156 权重完成 1.4 秒过程，实际调用共享主干 3360 次。左侧刚体抬升 3.59 厘米，右侧软体中心抬升 3.58 厘米；保持阶段最低离台高度分别为 3.18、2.15 厘米。录制帧中最小四面体体积比为 0.721，未发生翻转。当前仍有抬升目标跟踪误差。

同日圆柱和软球持续夹持验证：保持同一权重，摩擦系数全程为 2.5，完成 2.4 秒抓取、抬升与保持，未松爪、未滑落。圆柱和球末态中心抬升分别为 3.74、5.16 厘米；保持段最小离台高度分别为 3.72、4.07 厘米。保存帧中最小四面体体积比为 0.719。原生 VBD、Featherstone 调用均为零。

该验证复用自身同参数推理的前 1.4 秒抓取段，恢复空的夹持点历史与关节绕角分支，再继续推理 1 秒、调用主干 2400 次。恢复后首帧的位置与原记录完全一致，速度最大差异小于 5e-10；未读取教师轨迹。`--resume-grasp` 仅支持无固定粒子的自有抓取记录，检查资产、权重、Newton、设备、时间步、摩擦和控制目标，并要求首个续算帧匹配；不匹配会终止。

2026-11-17 水平夹取验证：同一 step 156 权重完成 2.8 秒动作，圆柱和软球分别实际抬升 8.90、8.66 厘米，向后回收 9.79、10.71 厘米。最后 0.4 秒停手保持期间，物体最低点离托座高度至少 9.01、7.76 厘米，未掉落；保存帧最小四面体体积比为 0.786。该记录保留自有推理前 2.4 秒，恢复历史后继续 0.4 秒，新增 960 次主干调用；首个续算帧位置误差低于 3e-13、速度误差低于 8e-9。原生 VBD、Featherstone 调用均为零。结果位于 `.local/w1_front_v316/hold_verified`，视频为 `.local/w1_front_v316/w1_front_verified_156.mp4`。

2026-11-17 无垫块侧夹验证：同一 step 156 权重完成 3 秒过程，圆柱和软球分别实际抬升 7.96、8.25 厘米，回收 10.13、9.24 厘米，末段保持期间均未掉落。实际录制各帧的原始机器人可视网格离桌间隙最小 12.31 毫米，未检测到穿桌。保存帧中最小四面体体积比为 0.884。复用自有推理前 0.8 秒闭爪记录，恢复后首帧位置、速度完全一致；随后将数值刚体切向支撑从 1 切换到 64 并新增 5280 次主干调用，物理摩擦系数仍为 2.5。完整有效轨迹共覆盖 7200 次主干调用，原生 VBD/Featherstone 调用均为零。记录位于 `.local/w1_side_v317/supported`，视频为 `.local/w1_side_v317/w1_side_pinch_156.mp4`。

简化纸袋装块任务：

```bash
.venv/bin/python scripts/run_w1_bag.py --output .local/w1_bag_test
.venv/bin/python scripts/render_w1_bag.py \
  --recording .local/w1_bag_test --output .local/w1_bag_test/w1_bag.mp4
```

场景参考 `/home/oem/code/engine/newton/newton/examples/mjvbdv2/example_mjvbd_v2_w1_bag_packing.py` 及其 `assets/w1_paper_bag/build_blender.py` 的尺寸和折面公式，不导入或运行耦合 demo。袋子初始直立，采用 13×32×25 厘米、237 节点、448 三角形的五面开口纸壳，省略折叠提手。袋子没有固定粒子、附着或脚本位姿。机器人双臂、夹爪、纸袋和方块均由同一份共享权重的推理路径推进，控制器只发关节目标。当前简化任务不启用袋子自接触，因此不等同于原场景的翻袋、自折叠和双零食装袋验证。

左手从上方接近并夹住袋口侧边；右手水平侧夹桌面上 5 厘米方块，抬到袋口上方、移入袋口中心、打开夹爪并撤离，让方块自然下落。默认 4.8 秒、60 Hz 录制、每帧 10 子步，每子步仍为 4 轮共享网络响应。纸张密度 0.18 kg/m²，膜刚度 5e4、折边和袋口分区弯曲刚度参考原场景；接触摩擦系数为 2.5。桌面高 1 米，无垫块。

运行记录包含左右夹爪接触载荷、真实粒子/刚体位姿速度、关节控制、桌面网格间隙与权重来源。`quality.json` 从最终纸袋拟合坐标系检查方块八角是否完整入袋，并检查方块曾被抬升、右手已撤离、左手仍在袋口及袋体未明显塌缩。最终验收与几何预览独立。可用 `--geometry-only` 只检查目标轨迹；不会生成模型推理结果。`--resume` 与 `--resume-frame` 只允许恢复自身推理记录，并要求首个续算帧吻合。

2026-11-17 装袋验证通过：同一 step 156 权重从初始状态完成 4.8 秒动作，调用共享主干 11520 次。方块被右手抬高 32.38 厘米，打开右爪后自由落入袋底；右手撤离后与方块相距 35.64 厘米，方块末段速度低于 0.28 毫米/秒。左手在抓稳后的每个记录帧中均有双侧袋沿接触。独立验收将实际变形袋口临时封盖，仅用于内外点判定，在 2 毫米接触容差下确认最后 37 帧的方块各角均在实际袋壳内部；仿真中袋口始终敞开。实际可视网格离桌最小 12.43 毫米，原生求解器调用、教师轨迹读取和优化器更新均为零。该版本初始为直立的低分辨率纸袋，省略提手和自接触，未运行参考 MJVBDV2 demo。

记录在 `.local/w1_bag_v318/rim_grip`，独立验收在 `.local/w1_bag_v318/task_verification.json`，视频为 `.local/w1_bag_v318/w1_bag_packing_156.mp4`。此轮整条推理及记录流程耗时约 2056 秒，平均约 0.140 仿真帧/秒，不含离线视频渲染。可用 `scripts/verify_w1_bag_recording.py --recording ... --report ...` 重新验收。

柔性袋版本使用 `--bag-profile flexible`：

```bash
.venv/bin/python scripts/run_w1_bag.py \
  --bag-profile flexible --frames 50 --output .local/w1_flexible_bag_prefix
.venv/bin/python scripts/run_w1_bag.py \
  --bag-profile flexible --soft-tangent-support-scale 512 --elastic-shape-response \
  --resume .local/w1_flexible_bag_prefix --resume-frame 48 \
  --output .local/w1_flexible_bag_test
.venv/bin/python scripts/render_w1_bag.py \
  --recording .local/w1_flexible_bag_test \
  --output .local/w1_flexible_bag_test/w1_flexible_bag.mp4
```

该版本膜刚度为 300、袋壁弯曲刚度为 0.003、折痕弯曲刚度为 0.0006、袋口弯曲刚度为 0.06，袋口相对袋壁仍有加强。左手夹稳后轻压袋沿 2.5 厘米并侧移 1 厘米，装入后再回提 1.5 厘米，控制器始终只发送关节目标。袋子的节点不做动画或渲染位移放大。`stiff` 保留上一版材料及动作，可用于复现对照。

形变指标逐帧拟合并扣除袋子整体平移和转动，再计算非刚性 RMS、最大节点偏移和第 90 百分位偏移。柔性版本除装袋、夹持、离桌等原验收外，还要求非刚性 RMS 超过 3 毫米、最大节点偏移超过 12 毫米，且放入方块后的非刚性 RMS 超过 3 毫米。视频叠加原始记录计算出的实际形变数值。续算同时检查材料系数，避免串用不同物性的状态前缀。

为改善有限 4 轮更新对袋壁形变的响应，柔性袋例支持 `--elastic-shape-response`。它为每个真实弹性对象构造 6 个仿射应变和 18 个二次位移候选，按质量去除平移与转动，在当前物理曲率中求取低频响应，再压缩为一个形变方向加入现有保护空间。该方向和原刚体、关节方向仍共用同一个神经粗空间响应头；总列数仍不超过 18，不新增网络参数，不替换材料力，不调用原生求解器，也不绕过物理验收。数值检查覆盖刚体质量正交、独立二次型响应、旋转平移协变、零残差固定点，并验证未启用时原有刚体/关节行为。

上述两步命令复现当前验收方式：先用自身推理生成接近前缀，再核对首个续算帧；核对后将软接触数值切向支撑由 64 调整到 512、启用弹性形变方向。材料、物理摩擦系数和神经权重始终不变，配置切换记录在 `grasp_continuation`。软接触支撑独立测试确认法向载荷、库仑上限、能量梯度和正定数值曲率保持相应约束。未启用增强的柔性短测作为同物性、同控制的对照保留。

2026-11-17 柔性袋验收通过：保留 step 156 权重，峰值非刚性 RMS 为 5.85 毫米、最大节点偏移为 16.61 毫米；装入后峰值 RMS 为 5.59 毫米，左手回提后降至 3.80 毫米，显示袋壁起皱和弹性恢复。计算均先扣除整体平移与转动，渲染不放大位移。同物性、同控制的短段对照在 1.6 秒时，原更新 RMS 为 1.64 毫米，加入通用弹性方向后为 5.82 毫米。

完整 4.8 秒记录中，左手抓稳后双侧接触持续存在，右手将方块抬高 34.98 厘米并释放，撤离后方块仍完整留在实际袋壳内部。末段 37 帧通过实际袋壳内外点检查（2 毫米接触容差）。最小三角形面积比为 0.910，可视机器人网格最小离桌间隙为 12.43 毫米，原生求解器调用和优化器更新均为零。本次复用自身同物性前 0.8 秒记录，核对首个续算帧后继续推理 4 秒、调用主干 9600 次，完整轨迹覆盖 11520 次主干调用。

有效记录为 `.local/w1_bag_flexible_v319/shape_modes`，视频为 `.local/w1_bag_flexible_v319/w1_flexible_bag_156.mp4`，物理形变曲线为 `.local/w1_bag_flexible_v319/deformation_comparison.png`。视频含全景、操作近景和俯视，20.77 秒慢放；仍为离线推理。原有无增强场景的物体运动回归检查通过，弹性方向的二次型、刚体正交与协变检查也通过。

袋沿夹持可选 `--bag-grip oblique`：左爪朝下倾斜 60°，夹持目标提高 6 毫米，使掌部和指根留在袋口上方，右手装袋轨迹保持原设计。仍只发送关节目标，不固定袋节点。旧 `horizontal` 姿态保留用于重放和对照。新姿态必须重新生成自身推理前缀，续算器拒绝跨姿态复用记录：

```bash
.venv/bin/python scripts/run_w1_bag.py \
  --bag-profile flexible --bag-grip oblique --frames 50 \
  --output .local/w1_oblique_prefix
.venv/bin/python scripts/run_w1_bag.py \
  --bag-profile flexible --bag-grip oblique \
  --soft-tangent-support-scale 512 --elastic-shape-response \
  --resume .local/w1_oblique_prefix --resume-frame 48 \
  --output .local/w1_oblique_test
.venv/bin/python scripts/verify_w1_bag_clearance.py \
  --recording .local/w1_oblique_test \
  --report .local/w1_oblique_test/gripper_clearance.json
.venv/bin/python scripts/render_w1_bag.py \
  --recording .local/w1_oblique_test \
  --output .local/w1_oblique_test/w1_oblique.mp4
```

新增的离线几何审查检查原始左腕、掌部和双指可视网格，包含袋壁三角形内部采样，默认采样间距 2 毫米、采样穿入容差 3 毫米。闭合可视网格使用有符号距离；原始 CAD 中的非闭合掌部和腕部网格不补洞或假定实体，而要求其与所有袋面三角形的包围盒分离，否则审查不通过。该检查为有限采样精度下的结果，不证明连续时间零穿透，不参与接触计算，不修改录制状态。新版视频第三视角朝向袋沿夹持位置。

2026-11-17 倾斜夹袋模型记录通过验收：完整 4.8 秒、289 个记录状态。原始左腕、掌部和双指可视网格逐帧检查通过；在 2 毫米袋面采样间距下，最大采样穿入深度为 0，非闭合掌部/腕部网格与袋面三角形包围盒保持分离。该采样结果不代表连续表面绝对零穿透。旧水平夹持记录同方法检测到约 5.98 毫米采样穿入。

新姿态下峰值非刚性 RMS 为 5.95 毫米、最大节点偏移为 16.48 毫米；最终 RMS 为 3.49 毫米。方块抬高 35.21 厘米，右手撤离后与方块相距 37.99 厘米，末段 37 帧的方块角点均在实际袋壳内（2 毫米容差）。最小可视机器人网格离桌间隙为 12.46 毫米。原生求解器调用、教师读取及优化器更新均为零，仍使用 step 156 权重。

记录位于 `.local/w1_bag_oblique_v320/final`，验收分别为 `task_verification.json` 与 `gripper_clearance.json`。长任务曾被运行进程中断，从自身完整保存帧续算，所有衔接均核对首帧，最大位置差异在浮点精度范围内。运行器现定期保存接触采样，续算继承已有采样；独立验收明确记录样本数量、时间范围与最大间隔。本轮 1–3 秒部分接触证据来自每 0.1 秒的实际日志采样，后段逐帧保存，不把稀疏接触数据当作逐帧载荷证据。几何交叠检查和运动轨迹均覆盖全部 289 个状态。视频保留袋沿近景，释放后另一个近景切换到俯视袋内。
