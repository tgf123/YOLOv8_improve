# 🔍 FD²-YOLO 系列项目合集（YOLOv8 / YOLOv11 / YOLOv12 单/双 Backbone）

本合集收录六个基于 YOLO 系列的创新改进项目，结合空域与频域特征、模块级积木式设计、可变形检测头与多任务兼容，适用于论文投稿、模型部署与创新研究。

---

## 📦 项目目录

| 项目名称 | 类型 | 模型特性 | 状态 |
|----------|------|----------|------|
| `YOLOv8` | 改进YOLO | Ultralytics YOLOv8 基础增强 | ✅ 已完成 |
| `YOLOv11` | 自研结构 | 支持积木式配置组合、轻量化 | ✅ 已完成 |
| `YOLOv12` | 多任务扩展 | 检测 + 分割 + 姿态 + 旋转框 | ✅ 已完成 |
| `YOLOv8 双Backbone` | 模型创新 | 空-频域融合 + 多尺度 | ✅ 已完成 |
| `YOLOv11 双Backbone` | 模型创新 | Hadamard动态融合 + 可变形检测头 | ✅ 已完成 |
| `YOLOv12 双Backbone` | 模型创新 | 多任务支持 + FD²-YOLO框架 | ✅ 已完成 |

---

## 🚀 模型创新亮点（以 YOLOv11 双Backbone 为例）

### 🧠 空频双Backbone结构

- **空间Backbone：** 保留YOLO主干特征提取能力；
- **频域Backbone：** 引入小波分解（DWT），增强纹理与边缘特征；
- **融合方式：** 使用 `DIFF模块` 动态融合两域特征（Hadamard机制 + KxK大核卷积）。

### 🔧 模块级创新（可配置组合）

| 模块名称 | 描述 | 作用 |
|----------|------|------|
| `C3K2-LW` | 卷积注意力融合模块 | 替代原始C3模块，提升轻量性与表达力 |
| `DIFF模块` | 动态融合频空特征 | Hadamard机制引导信息选择 |
| `DIA-Head` | 可变形检测头 | 提升检测精度与鲁棒性 |
| `FreqConv` | 频域卷积模块 | 融合高频与低频特征响应 |
| `FD²-Block` | 多分支频域聚合模块 | 多尺度重构空频特征，适配复杂场景 |

---

## 🧱 模块积木式配置系统

无需修改代码逻辑，仅通过 `*.yaml` 文件即可组合你的自定义模型结构：

```yaml
backbone:
  - type: FreqBackbone
    args: [C3K2_LW, FD2_Block]
neck:
  - type: DIFF
    args: [hadamard=True, kernel_size=5]
head:
  - type: Detect
    args: [DIA=True]
