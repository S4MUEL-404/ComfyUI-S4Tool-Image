# ComfyUI-S4Tool-Image

**Author**: S4MUEL404  
**Website**: [https://s4muel.com](https://s4muel.com)  
**Version**: 1.1.0

---

## Introduction

ComfyUI-S4Tool-Image is a powerful custom node package for ComfyUI, providing a variety of image processing, color analysis, and image composition tools. It is designed to enhance your image workflow with modular, easy-to-use nodes.

### Dependencies

- Python packages required for best experience:
  - `torch` (matching your ComfyUI build)
  - `numpy`
  - `pillow`
  - `opencv-python`
  - `scikit-image`

- One‑shot install (recommend running inside your ComfyUI environment):

```bash
pip install --upgrade numpy pillow opencv-python scikit-image
```

Note: `torch` should be installed according to your CUDA/CPU setup. If you already run ComfyUI, you likely have it installed. If not, see `https://pytorch.org/get-started/locally/` for the correct command.

### Node Overview

- **ImageCombine**  
  Combine 2-4 images into one, with customizable layout (horizontal/vertical), spacing, and background color.

- **ImageResize**  
  Resize images with multiple interpolation methods (nearest/bilinear/bicubic/area/lanczos), scaling strategies (stretch/keep proportion/fill-crop/pad), and conditional rules.

- **ImageAdjustment**  
  Adjust brightness, contrast, and saturation while preserving alpha channel.

- **ImageBlendWithAlpha**  
  Merge an image with an alpha mask, outputting an RGBA image. Supports mask inversion and auto-resizing.

- **ImageMaskExpand**  
  Expand a mask by a specified number of pixels using morphological operations.

- **ImageCropToFit**  
  Crop an image to remove all fully transparent borders, keeping the transparent background.

- **ImageColor**  
  Generate a solid color or gradient image. Supports HEX color, gradient angle, and size customization.

- **ImageBoard**  
  Create a transparent canvas of the given width and height.

- **ImageOverlay**  
  Overlay one image onto another with support for position, rotation, scaling, opacity, and blend modes.

- **ImagePalette**  
  Extract the five main colors from an image using various algorithms. Outputs a palette preview and HEX codes.

- **ImagePalette631**  
  Extract the three main colors from an image, similar to ImagePalette but with three colors.

- **ImagePrimaryColor**  
  Extract the primary color of an image. Outputs a preview and HEX code.

- **ImageFromBase64**  
  Decode a base64 string into an image and mask.

- **ImageToBase64**  
  Encode an image as a base64 string.

- **ImageTilingPattern**  
  Tile an image as a pattern on a transparent background, with support for spacing, offset, rotation, and output size.

- **ImageGetColor**  
  Read the hex color value of a pixel at (x, y) from an image.

- **SetImageBatch**  
  Create and optionally store a named image batch from one or two images and masks for later retrieval.

- **CombineImageBatch**  
  Combine two image batches into a single batch and optionally store with a new name.

- **GetImageBatch**  
  Retrieve an image and its mask from a stored batch by name using 1-based indexing.

---

## 节点包介绍

ComfyUI-S4Tool-Image 是一个为 ComfyUI 设计的强大自定义节点包，提供丰富的图像处理、颜色分析和图像合成工具。它让您的图像工作流更加高效、模块化、易用。

### 依赖说明

- 推荐在 ComfyUI 的 Python 环境中安装以下依赖，以获得最佳体验：
  - `torch`（请根据你的 CUDA/CPU 环境安装合适版本）
  - `numpy`
  - `pillow`
  - `opencv-python`
  - `scikit-image`

- 一键安装命令（在 ComfyUI 环境中执行）：

```bash
pip install --upgrade numpy pillow opencv-python scikit-image
```

提示：`torch` 一般随 ComfyUI 已安装；如未安装，请参考 PyTorch 官网安装指引：`https://pytorch.org/get-started/locally/`。

### 节点简介

- **ImageCombine**  
  将2-4张图片合成为一张，支持横向/纵向排列、间距和背景色自定义。

- **ImageResize**  
  多种插值方式（nearest/bilinear/bicubic/area/lanczos）、缩放策略（拉伸/等比/填充裁剪/留边）与条件控制的图片缩放。

- **ImageAdjustment**  
  调整亮度、对比度、饱和度，保留透明通道。

- **ImageBlendWithAlpha**  
  将图片与Alpha蒙版合并，输出RGBA图片。支持蒙版反转和自动尺寸匹配。

- **ImageMaskExpand**  
  通过形态学操作扩展蒙版指定像素。

- **ImageCropToFit**  
  裁剪图片，去除所有全透明边框，保留透明背景。

- **ImageColor**  
  生成纯色或渐变色图片，支持HEX颜色、渐变角度和尺寸自定义。

- **ImageBoard**  
  生成指定尺寸的全透明画布。

- **ImageOverlay**  
  将一张图片叠加到另一张图片上，支持位置、旋转、缩放、透明度和多种混合模式。

- **ImagePalette**  
  提取图片的五种主色，支持多种算法，输出调色板预览和HEX色值。

- **ImagePalette631**  
  提取图片的三种主色，功能类似ImagePalette但输出三种颜色。

- **ImagePrimaryColor**  
  提取图片主色，输出预览和HEX色值。

- **ImageFromBase64**  
  将Base64字符串解码为图片和蒙版。

- **ImageToBase64**  
  将图片编码为Base64字符串。

- **ImageTilingPattern**  
  将图片平铺为图案，支持间距、偏移、旋转和输出尺寸设置。

- **ImageGetColor**  
  读取图像中 (x, y) 像素的十六进制颜色值。

- **SetImageBatch**  
  由一到两对图片与蒙版创建图像批次，并可选以名称全局存储，供后续节点读取。

- **CombineImageBatch**  
  合并两个图像批次为一个，并可选以新名称存储。

- **GetImageBatch**  
  通过批次名称按1基索引获取批次中的图像与蒙版。