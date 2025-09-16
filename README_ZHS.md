# ComfyUI-S4Tool-Image

**版本: 1.3.0**

专为 ComfyUI 设计的综合图像处理工具包，提供 20 个专业级图像处理节点，具备生产级品质和可靠性。

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
   git clone https://github.com/S4MUEL/ComfyUI-S4Tool-Image.git
   ```
3. 安装依赖包：
   ```
   pip install -r ComfyUI-S4Tool-Image/requirements.txt
   ```
4. 重新启动 ComfyUI

## 🔧 依赖包

插件自动管理其依赖包，并提供详细的启动日志：
- **Pillow** (PIL) - 核心图像处理
- **NumPy** - 数值运算
- **Requests** - URL 图像加载
- **其他 AI 库** - 背景移除和高级功能

所有依赖包在启动时都会进行生产级质量验证。

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
- ✅ **专业工具** - 20 个专业图像处理节点
- ✅ **高性能** - 优化算法，兼顾速度和质量
- ✅ **用户友好** - 直观的节点界面，合理的默认值

## 📁 项目结构

```
ComfyUI-S4Tool-Image/
├── py/                 # 核心节点实现
├── examples/           # 使用示例和说明文档
├── summary_md/         # 开发总结和笔记
├── web/               # 网页界面资源
├── __init__.py        # 插件初始化
├── dependency_manager.py # 依赖包管理
└── requirements.txt   # Python 依赖包
```

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request 或报告问题。

## 📜 许可

本项目为开源项目。请遵守相关许可条款。

---

**作者:** S4MUEL  
**网站:** [s4muel.com](https://s4muel.com)  
**版本:** 1.3.0