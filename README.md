# ComfyUI 相机水印插件 (ComfyUI Camera Watermark)

![插件效果图](https://github.com/karas17/ComfyUI-Camera-Watermark/blob/main/images/01.png) 

一个为 ComfyUI 设计的多功能、高度可定制的相机水印和图像处理节点。无论您是想模拟经典的徕卡风格、添加EXIF信息，还是为图片增加专业的留白边框，这个插件都能轻松实现。

A versatile and highly customizable node for ComfyUI to add camera-style watermarks and frames to your images. Whether you want to emulate the classic Leica look, add EXIF data, or create professional-looking framed images, this plugin has you covered.

---

## ✨ 主要功能 (Features)

* **多种布局样式 (Multiple Layouts):** 内置10种精心设计的预设布局，包括底部留白、右侧留白、顶部和底部留白以及多种画内叠加样式。
* **EXIF数据集成 (EXIF Data Integration):** 能够自动读取并显示图片的EXIF信息（如相机型号、光圈、快门、ISO等）。
* **动态自适应缩放 (Dynamic Scaling):** 无论输入图片尺寸大小，水印和文字都能自动缩放，保持视觉上的协调与美观。
* **智能Logo系统 (Smart Logo System):**
    * 自动裁切Logo素材周围的透明边框。
    * Logo尺寸能够智能地与文字高度或留白区域大小关联，确保比例完美。
* **样式预设与自定义 (Style Presets & Customization):**
    * 为不同布局内置了最佳实践预设，一键应用。
    * 提供“恢复预设”按钮，方便您在自定义后快速还原。
* **随机样式功能 (Random Style Feature):** 可开启随机模式，在您指定的样式范围内进行随机选择，为创作带来更多可能性。
* **高度可定制 (Highly Customizable):** 从字体、颜色、大小到间距和边距，几乎所有视觉元素都可供您精细调整。
* **自定义资源 (Custom Assets):** 支持方便地添加您自己的字体和Logo素材。

---

## 📦 安装 (Installation)

1.  导航到您的 ComfyUI 安装目录下的 `custom_nodes` 文件夹。
    (e.g., `D:\ComfyUI\custom_nodes\`)
2.  使用 `git clone` 克隆本仓库：
    ```bash
    git clone [https://github.com/karas17/ComfyUI-Camera-Watermark.git](https://github.com/karas17/ComfyUI-Camera-Watermark.git) 
    ```
3.  或者，直接下载本仓库的ZIP压缩包，并解压到 `custom_nodes` 文件夹内。
4.  重启 ComfyUI。

---

## 🛠️ 使用方法 (Usage)

安装后，您会在节点菜单中找到两个新节点：

### 1. 加载/附加 EXIF (ImageLoaderWithEXIF)
这个节点是流程的起点。它会加载一张图片，并尝试读取其EXIF数据。即使您的图片没有EXIF信息，它也能正常工作。
![插件效果图](https://github.com/karas17/ComfyUI-Camera-Watermark/blob/main/images/02.jpg) 
### 2. 相机水印 (CameraWatermarkNode)
这是核心节点，负责生成水印和边框。
![插件效果图](https://github.com/karas17/ComfyUI-Camera-Watermark/blob/main/images/03.jpg) 
* **layout (布局):** 选择您喜欢的水印样式。
* **restore_preset (恢复预设):** 点击此按钮，会将当前选中样式的参数恢复为默认预设值。
* **randomize_style (随机样式):** 开启后，节点会根据 `seed` 在 `random_style_selection` 范围内随机选择一个样式。
* **random_style_selection (随机样式范围):** 定义随机选择的范围。格式可以是 `1, 3, 5` 或 `1-5, 8`。
* **seed (种子):** 更改此数值会触发一次新的随机选择。
* **其他参数:** 您可以自由调整所有关于文字、Logo、颜色和尺寸的设置。

---

## 🎨 自定义 (Customization)

### 添加字体 (Adding Fonts)
将您的 `.ttf` 或 `.otf` 字体文件放入插件目录下的 `fonts` 文件夹内，然后重启ComfyUI即可在节点的字体下拉菜单中找到它们。

### 添加Logo (Adding Logos)
将您的 `.png` 格式的Logo文件（推荐使用透明背景）放入插件目录下的 `logos` 文件夹内，然后重启ComfyUI即可在节点的Logo下拉菜单中找到它们。

---

## 授权 (License)

本项目采用 [MIT License](LICENSE) 授权。
