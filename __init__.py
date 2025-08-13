import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import re
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# Author: KARAS
# Version: 3.9.0 - Implemented dynamic scaling for small images
#
# --- 依赖导入 ---
try:
    import piexif
except ImportError:
    piexif = None

# --- 插件资源路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
fonts_dir = os.path.join(current_dir, "fonts")
logos_dir = os.path.join(current_dir, "logos")
os.makedirs(fonts_dir, exist_ok=True)
os.makedirs(logos_dir, exist_ok=True)

# --- 资源扫描与缓存 ---
def get_resource_files(directory: str, extensions: Tuple[str, ...]) -> List[str]:
    """Scans a directory for files with given extensions."""
    try:
        return sorted([f for f in os.listdir(directory) if f.lower().endswith(extensions)])
    except FileNotFoundError:
        return []

font_files = get_resource_files(fonts_dir, ('.ttf', '.ttc', '.otf')) or ["default"]
bold_font_files = [f for f in font_files if 'bold' in f.lower() or 'bd' in f.lower()] or font_files
logo_files = ["None"] + sorted([os.path.splitext(f)[0] for f in get_resource_files(logos_dir, ('.png',))])

# --- 预设数据 ---
PHONE_MODELS = [
    "HUAWEI Pura 70 Ultra", "HUAWEI Pura 70 Pro", "HUAWEI Mate 60 Pro",
    "iPhone 15 Pro Max", "iPhone 15 Pro",
    "Samsung Galaxy S24 Ultra", "Xiaomi 14 Ultra",
    "OPPO Find X7 Ultra", "vivo X100 Ultra", "Google Pixel 8 Pro", "自定义型号"
]

LAYOUT_OPTIONS = [
    "1. 底部留白 - 左右分布", 
    "2. 底部留白 - 居中", 
    "3. 右侧留白 - 上下竖排", 
    "4. 顶部 & 底部留白", 
    "5. 画内叠加 - 左下角", 
    "6. 画内叠加 - 右下角", 
    "7. 画内叠加 - 左上角", 
    "8. 画内叠加 - 右上角", 
    "9. 画内叠加 - 居中", 
    "10. 独立信息条"
]

LAYOUT_PRESETS = {
    LAYOUT_OPTIONS[0]: { 
         "logo": "leica",
         "logo_size_ratio": 1.0,
         "bar_size": 200,
         "bg_color": "#FFFFFF",
         "title_color": "#000000",
         "subtitle_color": "#888888",
    },
    LAYOUT_OPTIONS[1]: { 
        "block_b_text_1": "",
        "block_a_text_1": "", 
        "block_a_text_2": "", 
        "logo": "hasselblad",
        "logo_size_ratio": 1,
        "bg_color": "#FFFFFF",
        "title_color": "#000000",
        "subtitle_color": "#888888",
    },
    LAYOUT_OPTIONS[2]: { 
        "bar_size": 300, 
        "block_a_text_2": "",
        "logo_size_ratio": 1.0,
    },
    LAYOUT_OPTIONS[3]: { 
        "bar_size": 150,
        "block_a_text_2": "",
        "logo_size_ratio": 1.0, 
    },
    LAYOUT_OPTIONS[4]: { 
        "logo": "zeiss",
        "title_color": "#FFFFFF",  
        "subtitle_color": "#CCCCCC",
        "logo_size_ratio": 1.0,
    },
    LAYOUT_OPTIONS[5]: {
        "logo": "zeiss",
        "title_color": "#FFFFFF", 
        "subtitle_color": "#CCCCCC",
        "logo_size_ratio": 1.0,
    },
    LAYOUT_OPTIONS[6]: {
        "logo": "zeiss",
        "title_color": "#FFFFFF", 
        "subtitle_color": "#CCCCCC",
        "logo_size_ratio": 1.0,
    },
    LAYOUT_OPTIONS[7]: {
        "logo": "zeiss",
        "title_color": "#FFFFFF",  
        "subtitle_color": "#CCCCCC",
        "logo_size_ratio": 1.0,
    },
    LAYOUT_OPTIONS[8]: {
        "logo": "zeiss",
        "title_color": "#FFFFFF", 
        "subtitle_color": "#CCCCCC",
        "logo_size_ratio": 1.0,
    },
}


# -----------------------------------------------------------------
# 节点 1: 加载图片/附加EXIF
# -----------------------------------------------------------------
class ImageLoaderWithEXIF:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "input")
        files = get_resource_files(input_dir, ('.jpg', '.jpeg', '.png', '.webp'))
        return {
            "required": { "image_file": (sorted(files), {"image_upload": True}) },
            "optional": { "image_passthrough": ("IMAGE",) }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "exif_json")
    FUNCTION = "load_or_passthrough"
    CATEGORY = "Watermark"

    def load_or_passthrough(self, image_file: str, image_passthrough: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, str]:
        if piexif is None:
            raise ImportError("需要 'piexif' 库。请运行: pip install piexif")

        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "input", image_file)
        output_exif_json = "{}"
        try:
            with Image.open(image_path) as img_for_exif:
                if 'exif' in img_for_exif.info and img_for_exif.info['exif']:
                    exif_dict = piexif.load(img_for_exif.info['exif'])
                    exif_data = self.parse_exif(exif_dict)
                    output_exif_json = json.dumps(exif_data)
        except Exception as e:
            print(f"无法从 '{image_file}' 读取EXIF: {e}")

        if image_passthrough is not None:
            return (image_passthrough, output_exif_json)
        else:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                if "exif" in img.info:
                    try:
                        exif_dict = piexif.load(img.info["exif"])
                        orientation = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
                        if orientation > 1:
                            img_rgb = self.apply_exif_orientation(img_rgb, orientation)
                    except Exception as e:
                        print(f"处理EXIF方向时出错: {e}")
                
                image_tensor = np.array(img_rgb).astype(np.float32) / 255.0
                return (torch.from_numpy(image_tensor)[None,], output_exif_json)

    def apply_exif_orientation(self, image: Image.Image, orientation: int) -> Image.Image:
        """Applies EXIF orientation to the image."""
        if orientation == 2: return image.transpose(Image.FLIP_LEFT_RIGHT)
        if orientation == 3: return image.rotate(180)
        if orientation == 4: return image.transpose(Image.FLIP_TOP_BOTTOM)
        if orientation == 5: return image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        if orientation == 6: return image.rotate(-90, expand=True)
        if orientation == 7: return image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        if orientation == 8: return image.rotate(90, expand=True)
        return image

    def parse_exif(self, exif_dict: Dict) -> Dict[str, Any]:
        """Parses raw EXIF data into a structured dictionary."""
        exif_data = {}
        zeroth_ifd = exif_dict.get('0th', {})
        exif_ifd = exif_dict.get('Exif', {})

        if piexif.ImageIFD.Make in zeroth_ifd:
            exif_data['Make'] = zeroth_ifd[piexif.ImageIFD.Make].decode('utf-8', errors='ignore').strip('\x00').strip()
        if piexif.ImageIFD.Model in zeroth_ifd:
            exif_data['Model'] = zeroth_ifd[piexif.ImageIFD.Model].decode('utf-8', errors='ignore').strip('\x00').strip()
        if piexif.ExifIFD.LensModel in exif_ifd:
            exif_data['LensModel'] = exif_ifd[piexif.ExifIFD.LensModel].decode('utf-8', errors='ignore').strip('\x00').strip()
        if piexif.ExifIFD.DateTimeOriginal in exif_ifd:
            exif_data['Timestamp'] = exif_ifd[piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore').strip('\x00').strip()
        if piexif.ExifIFD.FNumber in exif_ifd:
            f_num = exif_ifd[piexif.ExifIFD.FNumber]
            exif_data['Aperture'] = f"f/{(f_num[0]/f_num[1]):.1f}" if f_num[1] > 0 else "f/0.0"
        if piexif.ExifIFD.ExposureTime in exif_ifd:
            exp = exif_ifd[piexif.ExifIFD.ExposureTime]
            if exp[0] > 0 and exp[1] > 0:
                shutter = exp[0] / exp[1]
                exif_data['ShutterSpeed'] = f"1/{int(1/shutter)}s" if shutter < 1 else f"{shutter}s"
        if piexif.ExifIFD.ISOSpeedRatings in exif_ifd:
            exif_data['ISO'] = f"ISO{exif_ifd[piexif.ExifIFD.ISOSpeedRatings]}"
        if piexif.ExifIFD.FocalLength in exif_ifd:
            focal = exif_ifd[piexif.ExifIFD.FocalLength]
            if focal[1] > 0:
                exif_data['FocalLength'] = f"{int(focal[0]/focal[1])}mm"
        return exif_data

# -----------------------------------------------------------------
# 节点 2: 相机水印生成器
# -----------------------------------------------------------------
@dataclass
class TextContent:
    block_a: List[str]
    block_b: List[str]

@dataclass
class TextBlock:
    lines: List[Tuple[str, ImageFont.FreeTypeFont, str]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    line_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

class CameraWatermarkNode:
    _font_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (LAYOUT_OPTIONS,),
                "restore_preset": ("BOOLEAN", {"default": False, "label_on": "应用当前样式预设", "label_off": "应用当前样式预设"}),
                "randomize_style": ("BOOLEAN", {"default": False}),
                "random_style_selection": ("STRING", {"multiline": False, "default": "1-10"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_exif": ("BOOLEAN", {"default": True}),
                "block_a_text_1": ("STRING", {"multiline": False, "default": "iPhone 15 Pro Max"}),
                "block_a_text_2": ("STRING", {"multiline": False, "default": "Main Camera"}),
                "block_a_text_3": ("STRING", {"multiline": False, "default": "© KARAS"}),
                "block_b_text_1": ("STRING", {"multiline": False, "default": "SONY"}),
                "block_b_text_2": ("STRING", {"multiline": False, "default": "24mm | f/1.8 | 1/125s | ISO 50"}),
                "block_b_text_3": ("STRING", {"multiline": False, "default": ""}),
                "logo": (logo_files, {"default": "leica" if "leica" in [lf.lower() for lf in logo_files] else logo_files[0]}),
                "logo_size_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "bar_size": ("INT", {"default": 200, "min": 0, "max": 2048, "step": 10}),
                "bar_width": ("INT", {"default": 1894, "min": 64, "max": 8192, "step": 64}),
                "bar_height": ("INT", {"default": 165, "min": 64, "max": 8192, "step": 64}),
                "padding": ("INT", {"default": 50, "min": 0, "max": 400, "step": 1}),
                "spacing": ("INT", {"default": 15, "min": 0, "max": 200, "step": 1}),
                "font_name": (font_files, {}),
                "font_bold_name": (bold_font_files, {}),
                "font_size_large": ("INT", {"default": 48, "min": 8, "max": 200, "step": 1}),
                "font_size_medium": ("INT", {"default": 28, "min": 8, "max": 200, "step": 1}),
                "font_size_small": ("INT", {"default": 24, "min": 8, "max": 200, "step": 1}),
                "bg_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "title_color": ("STRING", {"multiline": False, "default": "#000000"}),
                "subtitle_color": ("STRING", {"multiline": False, "default": "#888888"}),
            },
            "optional": { "exif_data": ("STRING", {"forceInput": True}), }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Watermark"

    def tensor_to_pil(self, tensor: torch.Tensor) -> Optional[Image.Image]:
        if tensor is None: return None
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
        
    def get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        cache_key = f"{font_name}_{size}"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        
        if not font_name or font_name == "default":
            try: font = ImageFont.load_default(size=size)
            except AttributeError: font = ImageFont.load_default()
        else:
            font_path = os.path.join(fonts_dir, font_name)
            try: font = ImageFont.truetype(font_path, size=size)
            except Exception as e:
                print(f"警告: 无法加载字体 '{font_name}'。错误: {e}")
                try: font = ImageFont.load_default(size=size)
                except AttributeError: font = ImageFont.load_default()
        
        self._font_cache[cache_key] = font
        return font

    def get_text_bbox(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
        if hasattr(font, 'getbbox'):
            return font.getbbox(text)
        else:
            w, h = font.getsize(text)
            return (0, 0, w, h)
    
    def _crop_transparent_border(self, pil_image: Image.Image) -> Image.Image:
        """Crops the transparent border from a PIL image."""
        if pil_image.mode != "RGBA":
            return pil_image 

        bbox = pil_image.getbbox()
        if bbox:
            return pil_image.crop(bbox)
        return pil_image

    def _parse_range_string(self, range_string: str) -> List[int]:
        numbers = set()
        parts = range_string.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    if start <= end:
                        numbers.update(range(start, end + 1))
                except ValueError:
                    print(f"警告: 无效的范围 '{part}' in random style selection.")
            else:
                try:
                    numbers.add(int(part))
                except ValueError:
                    print(f"警告: 无效的数字 '{part}' in random style selection.")
        return sorted(list(numbers))

    def _prepare_content(self, **kwargs) -> Tuple[TextContent, str]:
        exif_data_str = kwargs.get('exif_data')
        has_valid_exif = kwargs['use_exif'] and exif_data_str and exif_data_str.strip() != "{}"
        
        b_a1, b_a2, b_a3 = kwargs['block_a_text_1'], kwargs['block_a_text_2'], kwargs['block_a_text_3']
        b_b1, b_b2, b_b3 = kwargs['block_b_text_1'], kwargs['block_b_text_2'], kwargs['block_b_text_3']
        logo_match_source = b_a1.lower()

        if has_valid_exif:
            try:
                data = json.loads(exif_data_str)
                param_parts = [data.get("FocalLength"), data.get("Aperture"), data.get("ShutterSpeed"), data.get("ISO")]
                b_a1 = data.get("Model", "")
                b_a2 = data.get("LensModel", "")
                b_a3 = kwargs.get('author', '© KARAS')
                b_b1 = data.get("Make", "")
                b_b2 = " | ".join(filter(None, param_parts))
                b_b3 = data.get("Timestamp", "")
                logo_match_source = data.get("Make", "").lower()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"无法解析EXIF: {e}")

        content = TextContent(block_a=[b_a1, b_a2, b_a3], block_b=[b_b1, b_b2, b_b3])
        
        final_logo = kwargs['logo']
        if final_logo == "None" and logo_match_source:
             for logo_name in logo_files:
                if logo_name != "None" and logo_name.lower() in logo_match_source:
                    final_logo = logo_name
                    break
        
        return content, final_logo

    def _draw_block(self, draw, block: TextBlock, start_x: int, start_y: int, align: str = 'left', spacing: int = 10):
        y = start_y
        for i, (text, font, color) in enumerate(block.lines):
            bbox = block.line_bboxes[i]
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            x = start_x
            if align == 'center':
                x = start_x - text_width / 2
            elif align == 'right':
                x = start_x - text_width

            draw.text((x - bbox[0], y - bbox[1]), text, font=font, fill=color)
            y += text_height + spacing

    def _draw_vertical_text_block(self, draw_layer, block: TextBlock, center_x: int, start_y: int, spacing: int):
        if not block.lines:
            return 0
        
        temp_surface = Image.new("RGBA", (block.width, block.height), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_surface)

        y_cursor = 0
        min_x_offset = min(box[0] for box in block.line_bboxes)

        for i, (text, font, color) in enumerate(block.lines):
            bbox = block.line_bboxes[i]
            text_height = bbox[3] - bbox[1]
            temp_draw.text((-min_x_offset, y_cursor - bbox[1]), text, font=font, fill=color)
            y_cursor += text_height + spacing

        rotated_surface = temp_surface.rotate(90, expand=True, resample=Image.Resampling.LANCZOS)

        paste_x = center_x - rotated_surface.width / 2
        paste_y = start_y

        draw_layer.paste(rotated_surface, (int(paste_x), int(paste_y)), rotated_surface)
        
        return rotated_surface.height

    def generate_image(self, **kwargs):
        
        params = kwargs.copy()
        layout = params['layout']
        seed = params['seed']
        restore_preset = params.pop('restore_preset', False)

        if params.get('randomize_style', False):
            random.seed(seed)
            selection_str = params.get('random_style_selection', '1-10')
            allowed_indices = self._parse_range_string(selection_str)
            allowed_indices_0based = [i - 1 for i in allowed_indices if 1 <= i <= len(LAYOUT_OPTIONS)]
            if allowed_indices_0based:
                chosen_index = random.choice(allowed_indices_0based)
                layout = LAYOUT_OPTIONS[chosen_index]
                print(f"随机样式已启用 (种子: {seed})，选择: {layout}")
        
        if restore_preset:
            if layout in LAYOUT_PRESETS:
                print(f"恢复预设: {layout}")
                params.update(LAYOUT_PRESETS[layout])
        
        content, final_logo_name = self._prepare_content(**params)
        
        input_pil = self.tensor_to_pil(params['image'])
        img_w, img_h = input_pil.size
        
        # --- Dynamic Scaling Logic ---
        base_width = 1200.0 # Standard width for scaling
        scale_factor = min(1.0, img_w / base_width)

        bar_size = int(params['bar_size'] * scale_factor)
        padding = int(params['padding'] * scale_factor)
        spacing = int(params['spacing'] * scale_factor)
        font_size_large = int(params['font_size_large'] * scale_factor)
        font_size_medium = int(params['font_size_medium'] * scale_factor)
        font_size_small = int(params['font_size_small'] * scale_factor)
        
        bg_color = params['bg_color']

        if layout.startswith("1.") or layout.startswith("2.") or layout.startswith("10."):
            canvas = Image.new("RGBA", (img_w, img_h + bar_size), bg_color)
            canvas.paste(input_pil.convert("RGBA"), (0, 0))
            draw_area = (0, img_h, img_w, bar_size)
        elif layout.startswith("3."):
            canvas = Image.new("RGBA", (img_w + bar_size, img_h), bg_color)
            canvas.paste(input_pil.convert("RGBA"), (0, 0))
            draw_area = (img_w, 0, bar_size, img_h)
        elif layout.startswith("4."):
            canvas = Image.new("RGBA", (img_w, img_h + bar_size * 2), bg_color)
            canvas.paste(input_pil.convert("RGBA"), (0, bar_size))
            draw_area_top = (0, 0, img_w, bar_size)
            draw_area_bottom = (0, img_h + bar_size, img_w, bar_size)
            draw_area = draw_area_top
        elif layout.startswith("5.") or layout.startswith("6.") or layout.startswith("7.") or layout.startswith("8.") or layout.startswith("9."):
            canvas = input_pil.convert("RGBA")
            draw_area = (0, 0, img_w, img_h)
        else: 
            bar_width = int(params['bar_width'] * scale_factor)
            bar_height = int(params['bar_height'] * scale_factor)
            canvas = Image.new("RGBA", (bar_width, bar_height), bg_color)
            draw_area = (0, 0, canvas.width, canvas.height)
        
        draw_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(draw_layer)
        
        fonts = {
            'large_bold': self.get_font(params['font_bold_name'], font_size_large),
            'medium': self.get_font(params['font_name'], font_size_medium),
            'small': self.get_font(params['font_name'], font_size_small)
        }
        
        block_a_fonts = [fonts['large_bold'], fonts['medium'], fonts['small']]
        block_b_fonts = [fonts['large_bold'], fonts['medium'], fonts['small']]
        block_a_colors = [params['title_color'], params['subtitle_color'], params['subtitle_color']]
        block_b_colors = [params['title_color'], params['subtitle_color'], params['subtitle_color']]

        block_a = TextBlock()
        for i, text in enumerate(content.block_a):
            if text:
                block_a.lines.append((text, block_a_fonts[i], block_a_colors[i]))
        if block_a.lines:
            block_a.line_bboxes = [self.get_text_bbox(line[0], line[1]) for line in block_a.lines]
            min_x = min(box[0] for box in block_a.line_bboxes)
            max_x = max(box[2] for box in block_a.line_bboxes)
            block_a.width = max_x - min_x
            line_heights = [box[3] - box[1] for box in block_a.line_bboxes]
            block_a.height = int(sum(line_heights) + max(0, len(line_heights) - 1) * spacing)

        block_b = TextBlock()
        for i, text in enumerate(content.block_b):
            if text:
                block_b.lines.append((text, block_b_fonts[i], block_b_colors[i]))
        if block_b.lines:
            block_b.line_bboxes = [self.get_text_bbox(line[0], line[1]) for line in block_b.lines]
            min_x = min(box[0] for box in block_b.line_bboxes)
            max_x = max(box[2] for box in block_b.line_bboxes)
            block_b.width = max_x - min_x
            line_heights = [box[3] - box[1] for box in block_b.line_bboxes]
            block_b.height = int(sum(line_heights) + max(0, len(line_heights) - 1) * spacing)

        pil_logo = None
        if final_logo_name != "None":
            try:
                logo_filename = next((f for f in os.listdir(logos_dir) if os.path.splitext(f)[0].lower() == final_logo_name.lower()), None)
                if logo_filename: 
                    pil_logo = Image.open(os.path.join(logos_dir, logo_filename)).convert("RGBA")
                    pil_logo = self._crop_transparent_border(pil_logo)
            except Exception as e: 
                print(f"无法加载内置Logo '{final_logo_name}': {e}")
        
        if pil_logo:
            bar_based_target_height = 0
            if not (layout.startswith("5.") or layout.startswith("6.") or layout.startswith("7.") or layout.startswith("8.") or layout.startswith("9.")):
                max_size_ref = draw_area[3] if not layout.startswith("3.") else draw_area[2]
                bar_based_target_height = int(max_size_ref * 0.5)

            text_based_target_height = 0
            if layout in [LAYOUT_OPTIONS[0], LAYOUT_OPTIONS[9]]:
                text_based_target_height = block_b.height
            elif layout in [LAYOUT_OPTIONS[1], LAYOUT_OPTIONS[3]] or layout.startswith("5.") or layout.startswith("6.") or layout.startswith("7.") or layout.startswith("8.") or layout.startswith("9."):
                text_based_target_height = block_a.height if block_a.height > 0 else block_b.height
            
            base_height = max(text_based_target_height, bar_based_target_height)
            final_target_height = int(base_height * params['logo_size_ratio'])

            if final_target_height > 0:
                original_width, original_height = pil_logo.size
                if original_height > 0:
                    ratio = final_target_height / original_height
                    new_width = int(original_width * ratio)
                    pil_logo = pil_logo.resize((new_width, final_target_height), Image.Resampling.LANCZOS)

        if layout == LAYOUT_OPTIONS[0] or layout == LAYOUT_OPTIONS[9]:
            x, y, w, h = draw_area
            y_a = y + (h - block_a.height) / 2
            self._draw_block(draw, block_a, x + padding, y_a, spacing=spacing)
            
            y_b = y + (h - block_b.height) / 2
            if pil_logo:
                right_group_width = block_b.width + spacing + pil_logo.width
                start_x_right = x + w - padding - right_group_width
                logo_x = start_x_right
                logo_y = y + (h - pil_logo.height) / 2
                draw_layer.paste(pil_logo, (int(logo_x), int(logo_y)), pil_logo)
                block_b_x = logo_x + pil_logo.width + spacing
                self._draw_block(draw, block_b, block_b_x, y_b, align='left', spacing=spacing)
            else:
                self._draw_block(draw, block_b, x + w - padding, y_b, align='right', spacing=spacing)

        elif layout == LAYOUT_OPTIONS[1]:
            x, y, w, h = draw_area
            elements = []
            if pil_logo: elements.append({'type': 'logo', 'data': pil_logo, 'height': pil_logo.height})
            if block_a.lines: elements.append({'type': 'block', 'data': block_a, 'height': block_a.height})
            if block_b.lines: elements.append({'type': 'block', 'data': block_b, 'height': block_b.height})
            
            total_height = sum(el['height'] for el in elements) + max(0, len(elements) - 1) * spacing
            y_start = y + (h - total_height) / 2
            
            for el in elements:
                if el['type'] == 'logo':
                    logo = el['data']
                    logo_x = x + (w - logo.width) / 2
                    draw_layer.paste(logo, (int(logo_x), int(y_start)), logo)
                else:
                    self._draw_block(draw, el['data'], x + w/2, y_start, align='center', spacing=spacing)
                y_start += el['height'] + spacing
        
        elif layout == LAYOUT_OPTIONS[2]:
            x, y, w, h = draw_area
            center_x = x + w / 2

            if block_b.lines:
                y_b = y + padding
                self._draw_vertical_text_block(draw_layer, block_b, center_x, y_b, spacing)

            if block_a.lines:
                rotated_height_a = block_a.width
                y_a = y + h - padding - rotated_height_a
                self._draw_vertical_text_block(draw_layer, block_a, center_x, y_a, spacing)

            if pil_logo:
                logo_x = center_x - pil_logo.width / 2
                logo_y = y + (h - pil_logo.height) / 2
                draw_layer.paste(pil_logo, (int(logo_x), int(logo_y)), pil_logo)

        elif layout == LAYOUT_OPTIONS[3]:
            x_top, y_top, w_top, h_top = draw_area_top
            x_bot, y_bot, w_bot, h_bot = draw_area_bottom
            
            if pil_logo:
                total_top_width = pil_logo.width + spacing + block_a.width
                start_x_top = x_top + (w_top - total_top_width) / 2
                logo_x = start_x_top
                logo_y = y_top + (h_top - pil_logo.height) / 2
                draw_layer.paste(pil_logo, (int(logo_x), int(logo_y)), pil_logo)
                block_a_x = start_x_top + pil_logo.width + spacing
                self._draw_block(draw, block_a, block_a_x, y_top + (h_top - block_a.height) / 2, align='left', spacing=spacing)
            else:
                self._draw_block(draw, block_a, x_top + w_top / 2, y_top + (h_top - block_a.height) / 2, align='center', spacing=spacing)
            
            self._draw_block(draw, block_b, x_bot + w_bot / 2, y_bot + (h_bot - block_b.height) / 2, align='center', spacing=spacing)
        
        elif layout.startswith("5.") or layout.startswith("6.") or layout.startswith("7.") or layout.startswith("8.") or layout.startswith("9."):
            x, y, w, h = draw_area
            elements = []
            if block_a.lines: elements.append({'type': 'block', 'data': block_a, 'height': block_a.height, 'width': block_a.width})
            if block_b.lines: elements.append({'type': 'block', 'data': block_b, 'height': block_b.height, 'width': block_b.width})
            if pil_logo: elements.append({'type': 'logo', 'data': pil_logo, 'height': pil_logo.height, 'width': pil_logo.width})
            
            if elements:
                total_height = sum(el['height'] for el in elements) + max(0, len(elements) - 1) * spacing
                max_width = max(el['width'] for el in elements) if elements else 0

                if "左下角" in layout: y_start, x_start, align = h - padding - total_height, padding, 'left'
                elif "右下角" in layout: y_start, x_start, align = h - padding - total_height, w - padding, 'right'
                elif "左上角" in layout: y_start, x_start, align = padding, padding, 'left'
                elif "右上角" in layout: y_start, x_start, align = padding, w - padding, 'right'
                else: y_start, x_start, align = (h - total_height) / 2, w / 2, 'center'

                current_y = y_start
                for el in elements:
                    el_width = el['width']
                    if el['type'] == 'logo':
                        logo = el['data']
                        if align == 'left': logo_x = x_start
                        elif align == 'right': logo_x = x_start - el_width
                        else: logo_x = x_start - el_width / 2
                        draw_layer.paste(logo, (int(logo_x), int(current_y)), logo)
                    else:
                        self._draw_block(draw, el['data'], x_start, current_y, align=align, spacing=spacing)
                    current_y += el['height'] + spacing

        canvas.alpha_composite(draw_layer)
        final_image = canvas.convert("RGB")
        
        return (self.pil_to_tensor(final_image),)

# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "CameraWatermarkNode": CameraWatermarkNode,
    "ImageLoaderWithEXIF": ImageLoaderWithEXIF,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraWatermarkNode": "相机水印",
    "ImageLoaderWithEXIF": "加载/附加 EXIF",
}
