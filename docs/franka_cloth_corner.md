# Franka 从上方夹起布角

当前默认是 `overhead`：夹爪工具轴朝下，在布角处垂直合爪后抬起。此前侧夹结果保留为历史对照，当前视频使用新的垂直夹取方案。

参考 `/home/oem/code/repos/newton/newton/examples/cloth/example_cloth_franka.py` 的布料交互，但不执行该厘米制 VBD/Featherstone 耦合 demo。当前使用在 step100 基础上定向微调 40 次的共享网络，默认权重为 `model_releases/shared_vbd_v24_cloth40/model.pt`。原生 VBD/Featherstone 构造和推进均禁用。

## 初始布料与动作

布料展开尺寸为 18 cm × 18 cm、81 顶点、128 三角形。为了让垂直平行夹爪获得可夹入口，一侧预设 **22.5 mm 宽的自由折边**，只夹折边的一角。折边是输入布料的初始/静止几何，所有节点仍可动，没有夹爪附着、固定点、吸附或随手部覆盖粒子位置。

平铺薄布的完整下探试跑会把角推开，或在开始抬升时滑出指尖，因此未作为成功结果。本版从更近的垂直预夹取位置开始：0–0.2 s 轻微下降并合爪，0.3–2.0 s 竖直抬起约 30 cm 并略向桌外移，保持到 2.8 s。它验证垂直夹持和提起，不宣称已经解决任意平铺薄布的自动拾取。

膜刚度 300、弯曲刚度 0.0015、面密度 0.375 kg/m²、接触半径 1.5 mm、摩擦系数 2。折边布料总质量约 12.15 g，材料不通过网络另学。显示棋盘格随真实三角形运动，不改变物理。简单用例未启用自接触。

## 复现

从项目根目录执行，输出须为新目录：

```bash
WP_VBD_NEWTON_ROOT=.local/newton_pinned_25298 .venv/bin/python \
  scripts/run_franka_cloth_corner.py --device cuda:0 --fold-profile edge \
  --checkpoint model_releases/shared_vbd_v24_cloth40/model.pt --output outputs/franka_cloth_topdown

WP_VBD_NEWTON_ROOT=.local/newton_pinned_25298 .venv/bin/python \
  scripts/render_franka_cloth_corner.py --recording outputs/franka_cloth_topdown \
  --output outputs/franka_cloth_topdown.mp4

WP_VBD_NEWTON_ROOT=.local/newton_pinned_25298 .venv/bin/python \
  scripts/audit_franka_cloth_corner.py --recording outputs/franka_cloth_topdown \
  --report outputs/franka_cloth_topdown_audit.json
```

默认夹取姿态已经切换为 `overhead`；`--grasp-style side` 仅保留旧用例复现兼容性。CPU 将设备改为 `cpu`，`--geometry-only` 只是规划预览。

微调后的有效产物位于 `.local/cloth_finetune/`，原整边折起场景录制在 `model_edge/`，双视角视频为 `franka_cloth_finetuned.mp4`，审计结果为 `model_edge_audit.json`。默认使用已验证的 `edge` 资产；`corner` 是额外研究几何，不能将原整边案例的通过结论套用过去。

旧垂直夹取录像位于 `.local/franka_cloth_overhead/`，保留为失败外观对照。旧侧夹结果仍位于 `.local/franka_cloth_corner/`，不得与新录像混用。视频为 0.5 倍速离线回放，播放 FPS 与推理速度区分。

新权重在相同三子步、两轮解析预测加一次网络校正下完整运行 168 帧，实际网络调用 504 次，零重试。末段面板最低宽度比 0.644、法向一致性 0.754，抓角保持期双侧接触比例 100%。同 GPU、完全相同初始状态和电机指令的三轮纯解析对照仍卷曲；详见 [微调与对照记录](robot_cloth_finetuning.md)。

## 旧权重垂直夹取：任务通过，外观未通过

下列数值只说明旧权重抓取和网格检查通过。用户反馈的末段卷曲问题已复现；补充布面宽度及法向一致性检查后，`folded/` 的总体状态为 `FAIL_VISUAL_DRAPE`，不能作为有效的最终变形结果。三子步定向微调方案见 [robot_cloth_finetuning.md](robot_cloth_finetuning.md)。

GPU 运行完成 2.8 秒、168 帧、504 物理子步/网络前向、1512 轮物理响应，零拒绝重试或恢复。169 个保存状态审计通过：工具轴最大偏离竖直约 0.35°，布角抬升 299.83 mm，保持期整布最低表面离桌至少 122.03 mm，布角双侧接触采样比例 100%。

三角形面积比为 0.940–1.049，最大主拉伸约 1.124，机器人–布料表面重叠峰值约 0.594 mm。这次旧权重运行没有新增训练、教师读取或原生 VBD/Featherstone 调用。`folded/` 保留为失败外观对照；失败的平铺薄布试跑也不计入有效结果。
