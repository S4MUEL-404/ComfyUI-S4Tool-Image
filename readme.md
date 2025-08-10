# ComfyUI-S4Tool-Image

**Author**: S4MUEL404  
**Website**: [https://s4muel.com](https://s4muel.com)  
**Version**: 1.0.0

---

## Introduction

ComfyUI-S4Tool-Image is a powerful custom node package for ComfyUI, providing a variety of image processing, color analysis, and image composition tools. It is designed to enhance your image workflow with modular, easy-to-use nodes.

### Node Overview

- **ImageCombine**  
  Combine 2-4 images into one, with customizable layout (horizontal/vertical), spacing, and background color.

- **ImageBlendWithAlpha**  
  Merge an image with an alpha mask, outputting an RGBA image. Supports mask inversion and auto-resizing.

- **ImageMaskExpand**  
  Expand a mask by a specified number of pixels using morphological operations.

- **ImageCropToFit**  
  Crop an image to remove all fully transparent borders, keeping the transparent background.

- **ImageColor**  
  Generate a solid color or gradient image. Supports HEX color, gradient angle, and size customization.

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

---

## 节点包介绍

ComfyUI-S4Tool-Image 是一个为 ComfyUI 设计的强大自定义节点包，提供丰富的图像处理、颜色分析和图像合成工具。它让您的图像工作流更加高效、模块化、易用。

### 节点简介

- **ImageCombine**  
  将2-4张图片合成为一张，支持横向/纵向排列、间距和背景色自定义。

- **ImageBlendWithAlpha**  
  将图片与Alpha蒙版合并，输出RGBA图片。支持蒙版反转和自动尺寸匹配。

- **ImageMaskExpand**  
  通过形态学操作扩展蒙版指定像素。

- **ImageCropToFit**  
  裁剪图片，去除所有全透明边框，保留透明背景。

- **ImageColor**  
  生成纯色或渐变色图片，支持HEX颜色、渐变角度和尺寸自定义。

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

![MaskExpand_BlendWithAlpha](https://github.com/user-attachments/assets/b5710efa-397f-47bf-a44a-d78958ed464c)
![ImageToBase64](https://github.com/user-attachments/assets/75be4f09-e463-4354-979e-95179486e784)
![ImageTilingPattern](https://github.com/user-attachments/assets/cfe38211-307e-4f7e-a777-d904c0bbd824)
![ImagePrimaryColor](https://github.com/user-attachments/assets/f4c451a0-89de-4378-9766-6c4c96286bc4)
![ImagePalette631](https://github.com/user-attachments/assets/69797f53-0830-474e-adb0-fbabd9d47f55)
![ImagePalette](https://github.com/user-attachments/assets/4996db28-48da-4193-9201-7b71ef5dee24)
![ImageOverlay](https://github.com/user-attachments/assets/1f99758d-7a38-4a49-8e27-ee7307f6eaae)
![ImageCombine](https://github.com/user-attachments/assets/b3ab7fe4-809c-4f9f-9b85-69e93ae560b5)
![ImageColor](https://github.com/user-attachments/assets/bde022c9-d8e6-4971-a843-61e63ebfde2e)
![Base64_CropToFit](https://github.com/user-attachments/assets/b273546c-3f32-4a56-b25f-0ee4bfff585f)
