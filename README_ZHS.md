# ComfyUI-S4Tool-Image

**版本: 1.5.0**

专为 ComfyUI 设计的综合图像处理工具包，提供 22 个专业级图像处理节点，具备生产级品质和可靠性，包含先进的 SAM2.1 和 GroundingDINO AI 驱动分割功能。

## 🚀 功能特色

### 核心图像处理
- **💀Image Color Picker** - 精确提取图像颜色
- **💀Image Combine** - 多图像合并，支持高级混合选项
- **💀Image Resize** - 高质量图像缩放，支持多种算法
- **💀Image Crop To Fit** - 智能裁剪至指定尺寸
- **💀Image Adjustment** - 专业色彩和色调调整

### 高级效果
- **💀Image Blend With Alpha** - 高级 Alpha 混合，支持多种模式
- **💀Image Overlay** - 图像叠加，支持定位和变换控制
- **💀Image Board** - 创建图像网格和布局
- **💀Image Tiling Pattern** - 生成无缝拼接图案
- **💀Image RMBG** - AI 驱动的高精度背景移除
- **💀Image SAM** - Segment Anything Model 结合 GroundingDINO 的智能物体分割
- **💀Image SAM2** ⭐ 新增 - 增强版 SAM2.1 具备高级功能：
  - 8 个 SAM2 模型（tiny 至 large）
  - NMS 重复检测去除
  - 可调节检测框大小
  - 最小物体尺寸过滤
  - 可调节边缘平滑强度（0-5）
  - 智能输出数量控制
  - 基于质量的过滤
  - 复杂场景性能提升 20-38%

### 颜色和调色板工具
- **💀Image Color** - 生成纯色图像
- **💀Image Palette** - 提取和分析调色板
- **💀Image Palette 6-3-1** - 专业的 6-3-1 调色板提取
- **💀Image Primary Color** - 识别图像主要颜色
- **💀Image Get Color** - 从指定坐标取样颜色

### 文件和数据操作
- **💀Image from Base64** - Base64 字符串转换为图像
- **💀Image from URL** - 直接从 URL 加载图像
- **💀Image from Folder** - 批量加载文件夹中的图像
- **💀Image from Folder by Index** - 按索引加载特定图像
- **💀Image To Base64** - 图像转换为 Base64 格式

### 蒙版操作
- **💀Image Mask Expand** - 扩展和收缩图像蒙版

## 📦 安装方法

### 方法一：ComfyUI Manager（推荐）
1. 打开 ComfyUI Manager
2. 搜索 "S4Tool-Image"
3. 点击安装
4. 重新启动 ComfyUI

### 方法二：手动安装
1. 导航至 ComfyUI 自定义节点目录：
   ```
   cd ComfyUI/custom_nodes/
   ```
2. 克隆此仓库：
   ```
   git clone https://github.com/S4MUEL-404/ComfyUI-S4Tool-Image.git
   ```
3. 安装依赖包：
   ```
   pip install -r ComfyUI-S4Tool-Image/requirements.txt
   ```
4. **⚠️ 重要：为 SAM 节点安装 BERT 模型**
   
   导航至 ComfyUI 模型目录并安装 BERT 模型：
   ```bash
   cd ComfyUI/models/
   git clone https://huggingface.co/google-bert/bert-base-uncased
   ```
   
   **此步骤对于 💀Image SAM 节点的正常运行是必需的！**
   
   没有此模型，SAM 功能将无法正常工作。BERT 模型用于 GroundingDINO 的文本处理功能。
   
5. 重新启动 ComfyUI

## 🔧 依赖包

插件自动管理其依赖包，并提供详细的启动日志：
- **Pillow** (PIL) - 核心图像处理
- **NumPy** - 数值运算
- **Requests** - URL 图像加载
- **AI 模型库** - 背景移除、SAM 分割和 GroundingDINO
- **Segment Anything** - 高级物体分割
- **Transformers** - BERT 和神经网络支持

### ⚠️ **关键要求：SAM 的 BERT 模型**

为了让 **💀Image SAM** 节点正常工作，您 **必须** 手动安装 BERT 模型：

```bash
cd ComfyUI/models/
git clone https://huggingface.co/google-bert/bert-base-uncased
```

**为什么需要这个？**
- GroundingDINO（SAM 使用）需要 BERT 来进行自然语言处理
- 此模型支持基于文本的物体检测和分割
- 没有它，SAM 功能将完全无法使用

其他所有依赖包在启动时都会进行生产级质量验证。

## 📖 使用方法

1. **查找节点**：所有 S4Tool-Image 节点在 ComfyUI 节点浏览器中都以 💀 为前缀
2. **分类**：在 "💀S4Tool-Image" 分类下查找
3. **生产就绪**：所有节点都包含完整的错误处理和日志记录
4. **示例**：查看 `examples/` 文件夹中的使用说明文档

### 快速入门示例
1. 将任何 S4Tool-Image 节点添加到你的工作流
2. 连接图像输入
3. 设置节点参数
4. 执行工作流

## 🎯 主要特点

- ✅ **生产品质** - 企业级错误处理和验证
- ✅ **全面日志** - 详细的操作跟踪和调试
- ✅ **自动依赖管理** - 智能依赖包管理和验证
- ✅ **专业工具** - 21 个专业图像处理节点
- ✅ **高性能** - 优化算法，兼顾速度和质量
- ✅ **用户友好** - 直观的节点界面，合理的默认值

## 🆕 v1.5.0 新功能

### ImageSAM2 - 重大增强
全面升级的 SAM2 节点，具备生产级功能：

**模型支持**
- 8 个 SAM2 模型：SAM2.1 和 SAM2 系列（tiny/small/base_plus/large）
- 支持 2 个 GroundingDINO 模型（SwinT/SwinB）

**检测优化**（4 个新参数）
- `nms_threshold` - 移除重复检测（基于 IoU 的 NMS）
- `box_padding` - 调整检测框大小（-50 到 +50 像素）
- `min_box_size` - 过滤小型噪声检测
- `max_detections` - 限制输出数量，优先选择高质量

**边缘增强**
- `smooth_strength` - 可调节边缘平滑（0.0-5.0）
- 从自然到羽化的边缘效果

**性能提升**
- 复杂场景处理速度提升 20-38%
- 智能过滤管线减少冗余计算

**总参数**：16 个（12 个基础 + 4 个新增高级）

### 文档
- 完整的参数指南含视觉化示例
- 16 个参数快速参考表
- 5 个常用场景预设配置
- 故障排除指南和 FAQ

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request 或报告问题。

## 📜 许可

本项目为开源项目。请遵守相关许可条款。

---

**作者:** S4MUEL  
**网站:** [s4muel.com](https://s4muel.com)  
**GitHub:** [S4MUEL-404/ComfyUI-S4Tool-Image](https://github.com/S4MUEL-404/ComfyUI-S4Tool-Image)  
**版本:** 1.5.0
