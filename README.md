# Visual Tracker — SO-ARM101 视觉抓取系统

本项目包含两套独立的视觉系统，共享同一套摄像头与 GUI 基础设施：

| 入口 | 功能 |
|------|------|
| `servo_grasp_main.py` | **主系统**：YOLO-World 本地检测 + SO-ARM101 六轴机械臂视觉伺服抓取 |
| `main.py` | 旧系统：自然语言 VLM（doubao）+ OpenCV 追踪器（不含机械臂） |

---

## 硬件要求

- SO-ARM101（lerobot SO-101 Follower）六轴机械臂，USB 串口连接
- USB 摄像头（推荐 1280×720）
- Windows 10/11，Python 3.12+

---

## 快速启动

### 1. 安装依赖

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

依赖项包括：`PySide6`、`opencv-python`、`ultralytics`（YOLO-World）、`lerobot`（SO-101 驱动）

### 2. 配置

编辑 `config.yaml`：

```yaml
robot:
  port: COM24          # 机械臂串口
  id: so101_slave

servo:
  pan_gain: 25.0       # 水平误差增益
  tilt_gain: 15.0      # 垂直误差增益
  approach_step: 1.5   # 每帧逼近步长（关节单位）
  centering_threshold: 0.04   # 视觉误差阈值（归一化）
  grasp_area_threshold: 0.45  # 触发抓取的 bbox 面积占比
  gripper_open: 70     # 夹爪打开值
  gripper_close: 0     # 夹爪闭合值
  center_offset_x: 143 # 摄像头光心水平偏移（像素）
  center_offset_y: 61  # 摄像头光心垂直偏移（像素）

yolo:
  model: yolov8s-worldv2.pt
  conf_threshold: 0.15
```

### 3. 运行

```powershell
python servo_grasp_main.py
```

---

## 操作流程

1. 启动后在摄像头选择对话框中选择目标摄像头
2. 主界面左侧面板点击**连接机械臂**，连接成功后依次点击**录制待机位 / 待抓取位**保存预设姿态
3. 输入目标物体名称（如"Blue Cup"），点击**移至待抓取位**，再点击**开始识别**
4. 系统进入 YOLO-World 检测循环，检测到目标后进入 **TRACKING**（跟踪居中）状态
5. 点击**开始抓取**，状态机进入 **APPROACHING**（逼近）→ **CLOSING**（闭爪）→ **GRASPED**（完成）
6. 抓取完成后可点击**放置物品**将机械臂移至放置位并松爪

---

## 系统架构

### 状态机（GraspState）

```
DISCONNECTED
 │  机械臂连接成功
 ▼
STANDBY  ←─── 放置完成 / stop()
 │  go_pregrasp()
 ▼
MOVING（移动至预设姿态，两段式）
 │  到位
 ▼
PREGRASP
 │  start_tracking()
 ▼
DETECTING（YOLO 检测中）
 │  首次获得 bbox
 ▼
TRACKING（跟踪居中）─── bbox 丢失 ──► DETECTING
 │  start_approach()
 ▼
APPROACHING（逼近，夹爪张开）
 │  area_ratio >= grasp_area_threshold
 ▼
CLOSING（夹爪闭合）
 │  notify_grasped()
 ▼
GRASPED ──► place() ──► PLACING ──► STANDBY
```

### 文件结构

```
VisualTracker/
├── servo_grasp_main.py      # 主入口（视觉伺服抓取系统）
├── main.py                  # 旧入口（VLM 追踪系统，勿修改）
├── manual_record_replay.py  # 命令行轨迹录制 / 回放工具
├── config.yaml              # 全局配置
├── requirements.txt         # Python 依赖
├── positions.json           # GUI 保存的预设姿态（待机 / 待抓取 / 放置位）
├── yolov8s-worldv2.pt       # YOLO-World 模型权重
├── data/
│   └── demo_*.csv           # 示教录制数据（每次录制自动保存至根目录）
├── docs/
│   ├── SO-ARM101开源6轴机械臂使用文档.pdf
│   └── 视觉定位Grounding调用文档.md
└── src/
    ├── config.py            # 配置加载（YAML → dict）
    ├── robot_manager.py     # SO-101 连接、关节读写、力矩控制
    ├── servo_controller.py  # 视觉伺服控制器（状态机 + 运动逻辑）
    ├── yolo_tracker.py      # YOLO-World 目标检测 + EMA bbox 平滑（主系统用）
    ├── tracker.py           # VLM + OpenCV 追踪器（旧系统用，勿修改）
    ├── vlm_detector.py      # doubao VLM 检测（旧系统用，勿修改）
    ├── audio_manager.py     # numpy 合成音效
    ├── camera_manager.py    # 后台摄像头帧采集
    └── gui/
        ├── servo_window.py  # 视觉伺服主窗口（PySide6）
        ├── main_window.py   # VLM 旧窗口（勿修改）
        ├── camera_selector.py  # 摄像头选择对话框
        ├── draw_utils.py    # bbox / 状态徽章等绘制工具
        └── styles.py        # 深色 QSS 样式
```

---

## 核心模块说明

### robot_manager.py

封装 lerobot `SO101Follower`，提供：
- `connect(port, id)` / `disconnect()`
- `get_joints()` → `dict[str, float]`（6 个关节位置，`.pos` 后缀）
- `send_joints(dict)` → 发送关节指令（必须包含全部 5 个臂关节）
- `enable_torque()` / `disable_torque()`（含单关节回退逻辑）
- `set_gripper(value)` → 读取当前所有关节后一并发送，防止夹爪命令导致其他关节下垂

关节名称（lerobot 归一化，范围约 -100 到 +100）：
```
shoulder_pan.pos   shoulder_lift.pos   elbow_flex.pos
wrist_flex.pos     wrist_roll.pos      gripper.pos
```

### servo_controller.py

视觉伺服控制器，帧级运行（`update(frame)`）：

**TRACKING（跟踪居中）**
- 计算归一化误差：`error_x = (cx - (w/2 + offset_x)) / w`
- 调整 `shoulder_pan`（水平）和 `shoulder_lift`（垂直）
- 每帧增量限幅：`move_step_limit`
- 每帧发送全部 5 个臂关节，防止重力下垂

**APPROACHING（逼近）**
- 每帧 `shoulder_lift += approach_step`
- `elbow_flex` 按二次多项式计算（从示教数据拟合）：
  ```
  elbow = 0.001089 * lift² − 1.023 * lift − 5.55
  ```
- 同时修正 `shoulder_pan` 偏差
- 到达判断：`area_ratio >= grasp_area_threshold (0.45)`

**MOVING（移动至预设姿态）**
- 两段式：先完成 `elbow_flex / shoulder_lift / wrist_*` 到位，再移动 `shoulder_pan`
- 防止扫桌面电缆

### yolo_tracker.py

- 使用 `ultralytics` YOLO-World (`yolov8s-worldv2.pt`) 进行开放词汇本地检测
- 后台推理线程，不阻塞 Qt 主线程
- EMA 平滑（`ema_alpha: 0.6`）减少 bbox 抖动
- 丢失判断：连续 `max_lost_frames` 帧无检测则报告丢失

---

## 示教录制模式

侧边栏"示教录制"区域，用于采集人工操作数据：

1. 点击**开始录制（卸力模式）**：控制器停止，机械臂卸力，YOLO 继续跟踪
2. 手动移动机械臂完成一次抓取示范
3. 点击**停止录制**：力矩重新启用，CSV 自动保存到项目根目录（`demo_<timestamp>.csv`）

CSV 格式（每帧一行）：
```
timestamp, shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper,
bbox_x, bbox_y, bbox_w, bbox_h, area_ratio, cx_norm, cy_norm
```

历史录制数据存放于 `data/` 目录。

---

## 命令行工具

`manual_record_replay.py` — 不依赖 GUI，直接通过命令行录制和回放机械臂轨迹：

```powershell
python manual_record_replay.py
# 输入串口号（如 COM24），然后按提示选择 r/p/s/l/q
```

---

## 已知问题 / 待解决

- **逼近逻辑（APPROACHING）**：当前使用从示教数据拟合的二次多项式控制 `elbow_flex`，实际效果尚未最终验证。如果逼近轨迹不对，优先检查：
  1. `approach_step` 符号（正值 = `shoulder_lift` 增大 = 臂向上抬）
  2. 多项式系数与实际起始关节值的匹配
  3. `center_offset_x / center_offset_y` 是否反映了摄像头实际光轴偏移

- **夹爪力矩过载**：Feetech STS3215 在高负载时触发过载保护，`enable_torque()` 会绕过夹爪单独恢复臂关节

---

## 配置参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `servo.pan_gain` | 25.0 | 水平误差 → shoulder_pan 增量系数 |
| `servo.tilt_gain` | 15.0 | 垂直误差 → shoulder_lift 增量系数 |
| `servo.approach_step` | 1.5 | 每帧 shoulder_lift 步进量 |
| `servo.centering_threshold` | 0.04 | 归一化误差阈值，小于此值认为已居中 |
| `servo.grasp_area_threshold` | 0.45 | bbox 面积/画面面积 阈值，触发抓取 |
| `servo.move_step_limit` | 3.0 | 每帧关节增量上限（防止剧烈抖动） |
| `servo.move_timeout_sec` | 15.0 | 移动超时时间（秒） |
| `servo.center_offset_x/y` | 143 / 61 | 摄像头光轴偏移补偿（像素） |
| `yolo.conf_threshold` | 0.15 | YOLO 检测置信度阈值 |
| `yolo.max_lost_frames` | 15 | 目标消失帧数上限 |
| `yolo.ema_alpha` | 0.6 | EMA 平滑系数（越大越跟手） |
