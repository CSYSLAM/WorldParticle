# Piper 夹软块并放到刚体块上

使用本地 `~/下载/assets/scene/松灵 Piper 机械臂-urdf/piper/piper.urdf`。保留原始可视网格、碰撞体、惯性、6 个机械臂转动关节和夹指 `joint8=-joint7` 的 mimic 关系。原始资产不复制到版本库。

当前默认使用在 v24 step100 基础上微调 40 次的共享权重 `model_releases/shared_vbd_v24_cloth40/model.pt`，没有为 Piper 单独训练。网络和主要物理张量在 CUDA 上运行，原生 VBD/Featherstone 构造与推进在推理入口中明确禁用。

新权重已通过完整 312 帧、936 次网络调用、零重试的复核，最大机器人–软体表面重叠约 1.25 mm，全程体积比 0.605–1.103，最终软块稳定堆叠在自由刚体块上。新视频位于 `outputs/current_weight_demos_20261120/videos/04_piper_soft_place.mp4`，审计为同目录上一级的 `piper_audit.json`。统一入口：

```bash
.venv/bin/python scripts/run_robot_validation.py --scene piper --render
```

加 `--window` 可边推理边显示。六例汇总见 [统一场景说明](robot_demo_suite.md)。下文保留旧 step100 的测量作为历史对照。

## 场景和动作

- 蓝色软方块：边长 50 mm，4×4×4 四面体网格单元，密度 400 kg/m³，Lamé 参数 μ=λ=5000，粒子半径 3 mm。
- 橙色刚体支撑块：边长 60 mm，密度 1800 kg/m³，**自由动态刚体**，靠桌面接触支撑，不固定在世界。
- 物理摩擦系数 2；切向响应数值缩放 8，后者不是额外摩擦系数。
- 桌面高 150 mm，Piper 固定基座；采用前方斜向夹取，避开垂直向下姿态的关节限位。
- 0–0.9 s 接近并降低；0.9–1.4 s 闭爪；1.4–2.1 s 抬起；2.1–2.9 s 平移；2.9–3.6 s 降至支撑块；3.6–4.0 s 松爪；4.0–4.5 s 撤离；最后静置到 5.2 s。

IK 只产生关节驱动目标。机器人由有限刚度驱动和模型响应推进，软块没有粒子固定、夹爪附着、手部跟随或逐帧位姿覆盖。模型使用两轮解析预测与一次共享网络补偿、共享末端速度读出、解析材料剩余误差修正；名义每帧 3 个子步，困难阶段允许有界自适应恢复并计数。

Piper 的 URDF 惯性坐标变换在原生 Float32 导入后留下微小非对称项。场景入口只对相对非对称小于 1e−6 的张量取对称部分，并验证正定；这份资产约为 1e−8 量级。没有改变质量、URDF 源文件或放宽求解器对称性检查。

## 入口

从项目根目录执行：

```bash
WP_VBD_NEWTON_ROOT=.local/newton_pinned_25298 .venv/bin/python \
  scripts/run_piper_soft_place.py --device cuda:0 --output outputs/piper_soft_place

WP_VBD_NEWTON_ROOT=.local/newton_pinned_25298 .venv/bin/python \
  scripts/render_piper_soft_place.py --recording outputs/piper_soft_place \
  --output outputs/piper_soft_place.mp4

WP_VBD_NEWTON_ROOT=.local/newton_pinned_25298 .venv/bin/python \
  scripts/audit_piper_soft_place.py --recording outputs/piper_soft_place \
  --report outputs/piper_soft_place_audit.json
```

`--asset` 可传原始 URDF 或所在目录，CPU 推理将设备改为 `cpu`。输出目录必须是新目录。`--geometry-only` 仅检查规划和网格间隙，不是动力学结果，也不能生成模型推理视频。

视频为整体视角和抓取/放置近景，0.5 倍速离线回放。实际推理速度由 `run.json` 的输出帧数和墙钟时间计算，与视频播放帧率区分。

## 验证范围

审计覆盖保存状态的整体抬升、软块在刚体块坐标系中的位置和支撑接触、撤爪后稳定性、材料体积和去刚体运动的真实形变，以及原生碰撞查询的重叠量。保存完整轨迹、权重 SHA256、运行源码快照和最终接受历史。离散帧检查不构成连续无碰撞证明。

本轮产物统一位于 `.local/piper_soft_place/`；最终数据与审计以该目录的 `README.md` 为索引。

## 完整 GPU 运行结果

冻结 step100 在独立初始状态下完成 5.2 秒、312 帧，936 个物理子步、936 次神经前向、2808 轮物理响应，拒绝重试和恢复均为零。原生 VBD/Featherstone 调用、教师读取和新增训练更新均为零。

313 个保存状态的任务与原生几何审计通过：

| 指标 | 结果 |
| --- | ---: |
| 整个软块最低表面离桌的最高净间隙 | 137.79 mm |
| 最终软块中心相对刚体块中心的水平偏差 | 1.15 mm |
| 最终软块底面相对块顶间隙 | −0.0028 mm，近于接触零间隙 |
| 撤爪后夹爪工具点到软块中心距离 | 81.28 mm |
| 最终材料顶点最大速度 | 0.454 mm/s |
| 去除刚体运动后的形变 RMS 峰值 | 2.64 mm |
| 四面体体积比范围 | 0.594–1.092，无翻转 |
| 记录帧的机器人–软块表面重叠峰值 | 0.713 mm |
| 刚体块–软块表面重叠峰值 | 0.391 mm |

录制墙钟约 461.47 秒，实际整体输出速度约 **0.676 FPS**；不包含后续离线渲染，不是网络前向的单独吞吐率，也不是隔离其他负载的性能基准。

最终录像 `.local/piper_soft_place/piper_soft_place.mp4`，录制目录 `model/`，审计 `audit.json`。首次运行在最后静置阶段收到终止信号，其保存结果没有作为完整验收；独立后台重跑已经正常完成。本轮整理只保留最终有效产物与简短执行记录，移除一次性预览和失败/中断的重复轨迹。
