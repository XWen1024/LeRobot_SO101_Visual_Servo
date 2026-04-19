# CLAUDE.md — Visual Tracker 项目开发指南

## 项目概述

本项目是一个 SO-ARM101 六轴机械臂视觉伺服抓取系统。主入口为 `servo_grasp_main.py`，使用 YOLO-World 本地检测，通过视觉误差控制机械臂完成居中→逼近→抓取流程。

旧系统（`main.py`）基于 VLM（doubao API）+ OpenCV 追踪器，不含机械臂控制，**不要修改旧系统文件**。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `servo_grasp_main.py` | 应用入口，初始化日志、加载配置、启动 Qt |
| `src/servo_controller.py` | 核心控制器，状态机 + 运动逻辑 |
| `src/robot_manager.py` | SO-101 硬件封装 |
| `src/tracker.py` | YOLO-World 检测线程 |
| `src/gui/servo_window.py` | 主窗口 GUI |
| `config.yaml` | 所有可调参数 |

---

## 关节约定

- lerobot 归一化单位，范围约 **-100 到 +100**
- `shoulder_lift` **增大** = 臂更直立 = 摄像头视角上移 = 目标在画面中**下移**
- `shoulder_lift` **减小** = 臂俯向桌面 = 摄像头视角下移 = 目标在画面中**上移**
- `shoulder_pan` 增大 = 向右转
- 发送关节命令时**必须同时发送所有 5 个臂关节**（不含夹爪），否则未发送的关节因无力矩指令而受重力下垂

---

## 视觉误差计算

```python
error_x = (cx - (frame_w / 2 + offset_x)) / frame_w   # 正 = 目标在右
error_y = (cy - (frame_h / 2 + offset_y)) / frame_h   # 正 = 目标在下
```

居中修正方向：
- `new_pan = cur_pan + pan_gain * error_x`（右偏则右转）
- `new_lift = cur_lift + tilt_gain * error_y`（目标偏下则抬臂，让目标上移）

---

## 逼近逻辑（当前状态，尚未完全验证）

从示教数据（`demo_1775319475.csv`）拟合的关节轨迹：
- `shoulder_lift` 从约 -88 逐步增大到约 +70（每帧 +`approach_step`）
- `elbow_flex` 用二次多项式跟随：

```python
elbow = 0.001089 * lift**2 - 1.023 * lift - 5.55
```

`area_ratio >= 0.45` 时触发抓取（示教数据显示接触前约 0.149，接触瞬间约 0.62）。

**已知问题**：逼近轨迹实际效果尚未最终确认。调试时优先 check：
1. `approach_step` 的符号与方向
2. 机械臂初始姿态是否与示教起始姿态（lift ≈ -88）一致
3. `center_offset_x/y` 是否正确补偿了摄像头光轴

---

## 示教录制模式

卸力模式下 YOLO 继续追踪，人工操作机械臂，CSV 记录全部数据。关键实现：
- `_on_tick()` 检查 `_demo_recording` 标志，若为 True 则**完全绕过** `controller.update()`
- `_start_demo()` 必须先调 `controller.stop()` 再 `disable_torque()`，否则控制器仍会每帧发送关节命令重新上力矩

---

## 两段式移动（_tick_move）

移动到预设姿态时，先完成除 `shoulder_pan` 外所有关节，再移动 `shoulder_pan`。目的：防止扫桌面电缆。

---

## 开发规范

- 所有关节命令必须带完整的 5 个臂关节（`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`），夹爪单独用 `set_gripper()` 控制
- 新增关节命令前先调用 `get_joints()` 获取当前值，只修改需要改变的关节
- 日志级别为 DEBUG，第三方库（ultralytics、lerobot、draccus、PIL）设为 WARNING
- 控制器频率由 `move_fps`（默认 15Hz）的 QTimer 驱动，不要在 Qt 主线程做阻塞操作
- lerobot 的 `send_action()` 在 Feetech 总线上是同步写入，每次调用约 1-2ms，可安全在主线程调用
- `enable_torque()` 含回退逻辑（跳过夹爪），**不要简化**

---

## 不要做的事

- 不要修改 `main.py`、`src/vlm_detector.py`、`src/gui/main_window.py`（旧系统）
- 不要在 `_do_centering` / `_do_approach` 中只发送部分关节
- 不要在 demo 录制模式下调用 `controller.update()`
- 不要用 `robot.bus.write()` 直接写寄存器绕过 `send_joints`（除非调试 torque 问题）
