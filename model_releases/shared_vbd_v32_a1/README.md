# Shared VBD v32 / A1+N1 / step64

这是六个最新机器人视频使用的同一份完整推理检查点，保留原始字节和 SHA256，默认稳定入口仍使用 v24。

- 共享主干 648 维、12 头、6 层编码 / 4 层解码；材料读出头复用当前共享 latent。
- 活跃神经参数 **47,137,509**；结构标识沿用兼容的 `shared_material_corrector_v31`，本轮训练版本为 v32。
- 每物理子步一轮解析前缀 + 一次共享网络调用，固定场景名义每帧三子步；事件细分按实际接受步计数。
- 64 次联合微调，采用 step64；开发集目标 0.760396 → 0.618654。损失仍在改善，本轮按预算停止。
- 本轮实际使用 72 条训练轨迹，未重新混训全部 300 条增广或 42 个官方 demo。
- 六项固定任务完成各自检查；Piper 最小局部四面体体积比为 0.04291，不能据任务 PASS 宣称材质质量全部通过。
- 权重 **189,205,123 字节**，SHA256 `b73dbd02f92adcf7ee9b82587fa937c967b1d8f165ecb1974d10cc206c4b5c60`。

## 恢复与校验

六个普通 Git 分块，每个不超过 32 MiB；不使用 LFS 指针。

```bash
python scripts/fetch_model_snapshot.py --snapshot model_releases/shared_vbd_v32_a1
python scripts/verify_model_snapshot.py --snapshot model_releases/shared_vbd_v32_a1
```

本地分块优先，逐块和组合文件均验证 SHA256；HTTP 回退使用冻结标签 `shared-vbd-v32-a1-20261121`。组装出的 `model.pt` 被 Git 忽略。

## 运行

准备与主仓库 README 相同的 Torch/Warp 环境及 Newton 修订 `25298aea1691e060c86f95e8d2b773c898c006d3`。外部机器人资产和教师数据不包含在本快照中。

```bash
export WP_VBD_NEWTON_ROOT=/path/to/pinned/newton
python scripts/run_robot_validation.py --scene cloth --recipe configs/robot_validation_v32.json --window
python scripts/run_robot_demo_suite.py --recipe configs/robot_validation_v32.json \
  --output outputs/my_v32_six_demos --parallel-render
```

六例为 franka、cloth、w1、piper、w1_dual、franka_rigid_cloth。动作和材料参数固定在 `configs/robot_validation_v32.json`；视频、日志和原始轨迹不进入 Git。场景与材料质量边界见 [六例验证](../../docs/v32_six_scene_validation.md)。

## 训练边界

检查点含完整模型、配置和训练摘要，**不含优化器/RNG 状态**。配套 `selected.pt.training_state.pt` 已在原工作区保留，未作为数百 MB 的 Git 历史对象重复上传。原始训练数据也不在仓库中；本发布支持推理恢复和以新优化器初始化微调，不宣称提供精确训练续跑或完整训练复现。训练入口目前尚无经过验证的恢复 CLI。

材料补偿头仅用于 surface/joint v38；其他回放契约通过显式配置保留原解析材料响应。接触、关节、H/残差装配、粗空间投影与接受检查仍然存在。该研究快照未证明实时、未知资产泛化、自适应大步长或对全部官方 demo 的性能优势。
