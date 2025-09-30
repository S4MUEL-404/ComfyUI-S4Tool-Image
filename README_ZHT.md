# ComfyUI-S4Tool-Image

**版本: 1.5.0**

ComfyUI 的綜合圖像處理工具包，提供 22 個專業級圖像處理節點，具備生產級品質和可靠性，包含先進的 SAM2.1 和 GroundingDINO AI 驅動分割功能。

## 🚀 功能特色

### 核心圖像處理
- **💀Image Color Picker** - 精確提取圖像顏色
- **💀Image Combine** - 多圖像合併，支援進階混合選項
- **💀Image Resize** - 高品質圖像縮放，支援多種演算法
- **💀Image Crop To Fit** - 智慧裁剪至指定尺寸
- **💀Image Adjustment** - 專業色彩和色調調整

### 進階效果
- **💀Image Blend With Alpha** - 進階 Alpha 混合，支援多種模式
- **💀Image Overlay** - 圖像疊加，支援定位和變換控制
- **💀Image Board** - 建立圖像網格和佈局
- **💀Image Tiling Pattern** - 生成無縫拼接圖案
- **💀Image RMBG** - AI 驅動的高精度背景移除
- **💀Image SAM** - Segment Anything Model 結合 GroundingDINO 的智慧物件分割
- **💀Image SAM2** ⭐ 新增 - 增強版 SAM2.1 具備進階功能：
  - 8 個 SAM2 模型（tiny 至 large）
  - NMS 重複檢測去除
  - 可調節檢測框大小
  - 最小物體尺寸過濾
  - 可調節邊緣平滑強度（0-5）
  - 智慧輸出數量控制
  - 基於質量的過濾
  - 複雜場景性能提升 20-38%

### 顏色和調色板工具
- **💀Image Color** - 生成純色圖像
- **💀Image Palette** - 提取和分析調色板
- **💀Image Palette 6-3-1** - 專業的 6-3-1 調色板提取
- **💀Image Primary Color** - 識別圖像主要顏色
- **💀Image Get Color** - 從指定座標取樣顏色

### 檔案和資料操作
- **💀Image from Base64** - Base64 字串轉換為圖像
- **💀Image from URL** - 直接從 URL 載入圖像
- **💀Image from Folder** - 批次載入資料夾中的圖像
- **💀Image from Folder by Index** - 按索引載入特定圖像
- **💀Image To Base64** - 圖像轉換為 Base64 格式

### 遮罩操作
- **💀Image Mask Expand** - 擴展和收縮圖像遮罩

## 📦 安裝方法

### 方法一：ComfyUI Manager（推薦）
1. 開啟 ComfyUI Manager
2. 搜尋 "S4Tool-Image"
3. 點擊安裝
4. 重新啟動 ComfyUI

### 方法二：手動安裝
1. 導航至 ComfyUI 自定義節點目錄：
   ```
   cd ComfyUI/custom_nodes/
   ```
2. 克隆此儲存庫：
   ```
   git clone https://github.com/S4MUEL-404/ComfyUI-S4Tool-Image.git
   ```
3. 安裝相依套件：
   ```
   pip install -r ComfyUI-S4Tool-Image/requirements.txt
   ```
4. **⚠️ 重要：為 SAM 節點安裝 BERT 模型**
   
   導航至 ComfyUI 模型目錄並安裝 BERT 模型：
   ```bash
   cd ComfyUI/models/
   git clone https://huggingface.co/google-bert/bert-base-uncased
   ```
   
   **此步驟對於 💀Image SAM 節點的正常運作是必需的！**
   
   沒有此模型，SAM 功能將無法正常工作。BERT 模型用於 GroundingDINO 的文字處理功能。
   
5. 重新啟動 ComfyUI

## 🔧 相依套件

外掛程式自動管理其相依套件，並提供詳細的啟動日誌：
- **Pillow** (PIL) - 核心圖像處理
- **NumPy** - 數值運算
- **Requests** - URL 圖像載入
- **AI 模型函式庫** - 背景移除、SAM 分割和 GroundingDINO
- **Segment Anything** - 進階物件分割
- **Transformers** - BERT 和神經網絡支援

### ⚠️ **關鍵要求：SAM 的 BERT 模型**

為了讓 **💀Image SAM** 節點正常工作，您 **必須** 手動安裝 BERT 模型：

```bash
cd ComfyUI/models/
git clone https://huggingface.co/google-bert/bert-base-uncased
```

**為什麼需要這個？**
- GroundingDINO（SAM 使用）需要 BERT 來進行自然語言處理
- 此模型支援基於文字的物件檢測和分割
- 沒有它，SAM 功能將完全無法使用

其他所有相依套件在啟動時都會進行生產級品質驗證。

## 📖 使用方法

1. **尋找節點**：所有 S4Tool-Image 節點在 ComfyUI 節點瀏覽器中都以 💀 為前綴
2. **分類**：在 "💀S4Tool-Image" 分類下查找
3. **生產就緒**：所有節點都包含完整的錯誤處理和日誌記錄
4. **範例**：查看 `examples/` 資料夾中的使用說明文件

### 快速入門範例
1. 將任何 S4Tool-Image 節點新增到你的工作流程
2. 連接圖像輸入
3. 設定節點參數
4. 執行工作流程

## 🎯 主要特點

- ✅ **生產品質** - 企業級錯誤處理和驗證
- ✅ **全面日誌** - 詳細的操作追蹤和偵錯
- ✅ **自動相依管理** - 智慧相依套件管理和驗證
- ✅ **專業工具** - 21 個專業圖像處理節點
- ✅ **高效能** - 優化演算法，兼顧速度和品質
- ✅ **使用者友善** - 直觀的節點介面，預設值合理

## 🆕 v1.5.0 新功能

### ImageSAM2 - 重大增強
全面升級的 SAM2 節點，具備生產級功能：

**模型支援**
- 8 個 SAM2 模型：SAM2.1 和 SAM2 系列（tiny/small/base_plus/large）
- 支援 2 個 GroundingDINO 模型（SwinT/SwinB）

**檢測優化**（4 個新參數）
- `nms_threshold` - 移除重複檢測（基於 IoU 的 NMS）
- `box_padding` - 調整檢測框大小（-50 到 +50 像素）
- `min_box_size` - 過濾小型雜訊檢測
- `max_detections` - 限制輸出數量，優先選擇高質量

**邊緣增強**
- `smooth_strength` - 可調節邊緣平滑（0.0-5.0）
- 從自然到羽化的邊緣效果

**性能提升**
- 複雜場景處理速度提升 20-38%
- 智慧過濾管線減少冗餘計算

**總參數**：16 個（12 個基礎 + 4 個新增進階）

### 文檔
- 完整的參數指南含視覺化示例
- 16 個參數快速參考表
- 5 個常用場景預設配置
- 故障排除指南和 FAQ

## 🤝 貢獻

歡迎貢獻！請隨時提交 Pull Request 或回報問題。

## 📜 授權

本專案為開源專案。請遵守相關授權條款。

---

**作者:** S4MUEL  
**網站:** [s4muel.com](https://s4muel.com)  
**GitHub:** [S4MUEL-404/ComfyUI-S4Tool-Image](https://github.com/S4MUEL-404/ComfyUI-S4Tool-Image)  
**版本:** 1.5.0
