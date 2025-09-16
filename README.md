# ComfyUI-S4Tool-Image

**Version: 1.3.0**

A comprehensive image processing toolkit for ComfyUI, providing 20 professional-grade image manipulation nodes with production-ready quality and reliability.

## 🚀 Features

### Core Image Processing
- **💀Image Color Picker** - Extract colors from images with precision
- **💀Image Combine** - Merge multiple images with advanced blending options
- **💀Image Resize** - High-quality image resizing with multiple algorithms
- **💀Image Crop To Fit** - Smart cropping to fit specific dimensions
- **💀Image Adjustment** - Professional color and tone adjustments

### Advanced Effects
- **💀Image Blend With Alpha** - Advanced alpha blending with multiple modes
- **💀Image Overlay** - Layer images with positioning and transformation controls
- **💀Image Board** - Create image grids and layouts
- **💀Image Tiling Pattern** - Generate seamless tiling patterns
- **💀Image RMBG** - Background removal with AI-powered precision

### Color & Palette Tools
- **💀Image Color** - Generate solid color images
- **💀Image Palette** - Extract and analyze color palettes
- **💀Image Palette 6-3-1** - Specialized 6-3-1 color palette extraction
- **💀Image Primary Color** - Identify dominant colors in images
- **💀Image Get Color** - Sample colors from specific image coordinates

### File & Data Operations
- **💀Image from Base64** - Convert Base64 strings to images
- **💀Image from URL** - Load images directly from URLs
- **💀Image from Folder** - Batch load images from directories
- **💀Image from Folder by Index** - Load specific images by index
- **💀Image To Base64** - Convert images to Base64 format

### Mask Operations
- **💀Image Mask Expand** - Expand and contract image masks

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "S4Tool-Image" 
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
1. Navigate to your ComfyUI custom_nodes directory:
   ```
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```
   git clone https://github.com/S4MUEL/ComfyUI-S4Tool-Image.git
   ```
3. Install dependencies:
   ```
   pip install -r ComfyUI-S4Tool-Image/requirements.txt
   ```
4. Restart ComfyUI

## 🔧 Dependencies

The plugin automatically manages its dependencies and provides detailed startup logging:
- **Pillow** (PIL) - Core image processing
- **NumPy** - Numerical operations
- **Requests** - URL image loading
- **Additional AI libraries** - Background removal and advanced features

All dependencies are automatically checked at startup with production-quality validation.

## 📖 Usage

1. **Find Nodes**: All S4Tool-Image nodes are prefixed with 💀 in the ComfyUI node browser
2. **Categories**: Look under "💀S4Tool-Image" category
3. **Production Ready**: All nodes include comprehensive error handling and logging
4. **Examples**: Check the `examples/` folder for usage documentation

### Quick Start Example
1. Add any S4Tool-Image node to your workflow
2. Connect your image input
3. Configure node parameters
4. Execute workflow

## 🎯 Key Features

- ✅ **Production Quality** - Enterprise-grade error handling and validation
- ✅ **Comprehensive Logging** - Detailed operation tracking and debugging
- ✅ **Automatic Dependencies** - Smart dependency management and validation
- ✅ **Professional Tools** - 20 specialized image processing nodes
- ✅ **High Performance** - Optimized algorithms for speed and quality
- ✅ **User Friendly** - Intuitive node interfaces with helpful defaults

## 📁 Project Structure

```
ComfyUI-S4Tool-Image/
├── py/                 # Core node implementations
├── examples/           # Usage examples and documentation
├── summary_md/         # Development summaries and notes
├── web/               # Web interface assets
├── __init__.py        # Plugin initialization
├── dependency_manager.py # Dependency management
└── requirements.txt   # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or report issues.

## 📜 License

This project is open source. Please respect the licensing terms.

---

**Author:** S4MUEL  
**Website:** [s4muel.com](https://s4muel.com)  
**Version:** 1.3.0