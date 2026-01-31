"""
LED 图片显示程序
生成 AI 图片并显示到 LED 显示屏
"""
import time
import os
import signal
import sys
import argparse
from datetime import datetime

import config

# ========== 加载 .env 文件 ==========
def load_env_file():
    """加载 .env 文件中的环境变量"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env_file()

# PID文件路径
PID_FILE = "/tmp/led_image_display.pid"


def check_previous_instance():
    """检查是否有之前的实例在运行"""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"发现之前的进程 (PID: {old_pid}) 正在运行，正在关闭...")
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(0.5)
                print("已关闭之前的进程")
            except (OSError, ProcessLookupError):
                pass
        except (ValueError, IOError):
            pass

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


def cleanup():
    """程序退出时清理"""
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def cleanup_leds():
    """关闭所有LED"""
    try:
        for pixels in pixel_objects:
            pixels.fill(0)
            pixels.show()
    except Exception:
        pass


check_previous_instance()
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_leds(), cleanup(), sys.exit(0)))
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_leds(), cleanup(), sys.exit(0)))

# LED 控制相关导入
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write
import adafruit_pixelbuf
from PIL import Image, ImageEnhance, ImageFilter

# 图片生成相关导入
import base64
import requests
import os
from io import BytesIO
from datetime import datetime


# ========== LED 配置 ==========
BOARD_ROWS = 5
BOARD_COLS = 1
LED_ROWS_PER_BOARD = 8
LED_COLS_PER_BOARD = 32
SCREEN_COLS = BOARD_COLS * LED_COLS_PER_BOARD  # 32
SCREEN_ROWS = BOARD_ROWS * LED_ROWS_PER_BOARD  # 40
PIXELS_PER_BOARD = LED_ROWS_PER_BOARD * LED_COLS_PER_BOARD  # 256
PINS = config.PINS
BRIGHTNESS = 0.1  # 亮度 10%


class Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    def __init__(self, pin, size, **kwargs):
        self._pin = pin
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin, buf)


def hsv_to_rgb(h, s, v):
    """HSV转RGB"""
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = int(v * (1 - s) * 255)
    q = int(v * (1 - f * s) * 255)
    t = int(v * (1 - (1 - f) * s) * 255)
    v_int = int(v * 255)

    if i % 6 == 0:
        return (v_int, t, p)
    elif i % 6 == 1:
        return (q, v_int, p)
    elif i % 6 == 2:
        return (p, v_int, t)
    elif i % 6 == 3:
        return (p, q, v_int)
    elif i % 6 == 4:
        return (t, p, v_int)
    else:
        return (v_int, p, q)


def get_pixel_index(col, row):
    """将屏幕坐标转换为灯板索引和灯板内索引（蛇形扫描）"""
    board_index = row // LED_ROWS_PER_BOARD
    local_row = row % LED_ROWS_PER_BOARD
    local_col = col

    if local_col % 2 == 0:
        pixel_index = local_col * LED_ROWS_PER_BOARD + local_row
    else:
        pixel_index = local_col * LED_ROWS_PER_BOARD + (LED_ROWS_PER_BOARD - 1 - local_row)

    return board_index, pixel_index


def set_pixel_no_clear(col, row, color):
    """设置单个像素"""
    board_idx, pixel_idx = get_pixel_index(col, row)
    pixel_objects[board_idx][pixel_idx] = color


def clear_screen():
    """清除屏幕"""
    for pixels in pixel_objects:
        pixels.fill(0)
        pixels.show()


def draw_border(hue_offset=0):
    """绘制彩色渐变边框"""
    # 上边框
    for col in range(SCREEN_COLS):
        hue = (col / SCREEN_COLS + hue_offset) % 1.0
        color = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
        set_pixel_no_clear(col, 0, color)

    # 下边框
    for col in range(SCREEN_COLS):
        hue = (col / SCREEN_COLS + hue_offset) % 1.0
        color = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
        set_pixel_no_clear(col, SCREEN_ROWS - 1, color)

    # 左边框
    for row in range(SCREEN_ROWS):
        hue = (row / SCREEN_ROWS + hue_offset) % 1.0
        color = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
        set_pixel_no_clear(0, row, color)

    # 右边框
    for row in range(SCREEN_ROWS):
        hue = (row / SCREEN_ROWS + hue_offset) % 1.0
        color = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
        set_pixel_no_clear(SCREEN_COLS - 1, row, color)


# 初始化所有LED组
pixel_objects = []
for pin in PINS:
    pixels = Pi5Pixelbuf(pin, PIXELS_PER_BOARD, auto_write=False, byteorder="GRB", brightness=BRIGHTNESS)
    pixel_objects.append(pixels)


# ========== 图片生成 ==========
API_URL = "https://api.minimaxi.com/v1/image_generation"

# LED 优化提示词后缀 - 要求黑色背景和大内容
LED_PROMPT_SUFFIX = (
    ", pixel art style, "
    "8-bit style, "
    "bitmap style, "
    "solid black background, "
    "white or bright content on black background, "
    "high contrast, "
    "simplified design, "
    "content should be LARGE and FILL THE FRAME, "
    "maximized subject size, "
    "suitable for LED display"
)


def generate_image(prompt: str, api_key: str = None) -> Image.Image:
    """调用 MiniMax API 生成图片"""
    if not api_key:
        raise ValueError("请设置 MiniMax API Key")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "image-01",
        "prompt": prompt,
        "aspect_ratio": "1:1",
        "response_format": "base64"
    }

    print(f"正在生成图片: {prompt}")
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    if "data" in result and "image_base64" in result["data"]:
        image_base64_list = result["data"]["image_base64"]
    else:
        raise ValueError(f"API 返回格式异常: {result}")

    if isinstance(image_base64_list, list):
        image_base64 = image_base64_list[0]
    else:
        image_base64 = image_base64_list

    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    elapsed = time.time() - start_time
    print(f"图片生成成功！原始尺寸: {image.size}，耗时: {elapsed:.2f}秒")

    return image


def generate_led_image(prompt: str, api_key: str = None, threshold: int = 100, contrast: float = 2.5) -> Image.Image:
    """
    生成适合 LED 显示的图片（带后处理）

    Args:
        prompt: 基础提示词
        api_key: API Key
        threshold: 二值化阈值
        contrast: 对比度增强

    Returns:
        处理后的 PIL Image (32x40)
    """
    # 添加 LED 优化后缀
    led_prompt = prompt + LED_PROMPT_SUFFIX

    # 生成图片
    image = generate_image(led_prompt, api_key=api_key)

    # 后处理
    image = enhance_for_led(image, threshold=threshold, contrast=contrast)

    return image


def enhance_for_led(image: Image.Image, threshold: int = 128, contrast: float = 2.0) -> Image.Image:
    """
    增强图片以适应 LED 显示
    - 灰度转换
    - 对比度增强
    - 反相：亮的变黑（背景），暗的变白（内容）
    - 去噪
    - 膨胀填充
    """
    # 转换为灰度图
    if image.mode != 'L':
        img = image.convert('L')
    else:
        img = image.copy()

    # 对比度增强
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    # 二值化：亮的像素（白色内容）变白，暗的像素（黑色背景）变黑
    img = img.point(lambda p: 255 if p > threshold else 0, mode='L')

    # 膨胀 - 填充小空洞
    img = img.filter(ImageFilter.MaxFilter(3))

    # 再次二值化
    img = img.point(lambda p: 255 if p > 127 else 0, mode='L')

    # 调整大小到 LED 尺寸
    img = img.resize((SCREEN_COLS, SCREEN_ROWS), Image.Resampling.NEAREST)

    return img


# ========== 图片显示 ==========
def display_image(img: Image.Image, show_border: bool = True, border_animate: bool = True):
    """
    将图片显示到 LED 屏幕
    - 背景：黑色（不亮）
    - 内容：彩虹彩色渐变
    img: PIL Image 对象（支持 L 模式）
    show_border: 是否显示边框
    border_animate: 是否动画效果
    """
    # 转换为灰度模式
    if img.mode != 'L':
        img = img.convert('L')

    # 调整图像大小以适应屏幕
    img = img.resize((SCREEN_COLS, SCREEN_ROWS), Image.Resampling.NEAREST)

    # 清除屏幕
    for pixels in pixel_objects:
        pixels.fill(0)

    # 遍历屏幕每个像素
    for row in range(SCREEN_ROWS):
        for col in range(SCREEN_COLS):
            pixel = img.getpixel((col, row))

            # 如果像素是白色（内容区域），点亮 LED
            if pixel > 127:
                # 转换到 LED 坐标
                board_idx, pixel_idx = get_pixel_index(col, row)
                # 使用彩虹颜色（根据位置生成不同颜色）
                hue = ((col + row) / (SCREEN_COLS + SCREEN_ROWS)) % 1.0
                r, g, b = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
                pixel_objects[board_idx][pixel_idx] = (r, g, b)

    # 显示边框
    if show_border:
        if border_animate:
            # 动态边框
            for i in range(60):  # 3秒动画
                draw_border(hue_offset=i / 60)
                for pixels in pixel_objects:
                    pixels.show()
                time.sleep(0.05)
        else:
            draw_border()
            for pixels in pixel_objects:
                pixels.show()
    else:
        for pixels in pixel_objects:
            pixels.show()


def display_image_with_scroll(img: Image.Image, direction: str = "left", speed: float = 0.05):
    """
    带滚动效果的图片显示
    direction: "left" 或 "up"
    speed: 滚动速度（秒）
    """
    img = img.resize((SCREEN_COLS, SCREEN_ROWS), Image.Resampling.NEAREST)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size

    if direction == "left":
        # 水平滚动
        for offset in range(width):
            for pixels in pixel_objects:
                pixels.fill(0)

            for row in range(SCREEN_ROWS):
                for col in range(SCREEN_COLS):
                    src_col = col + offset
                    if src_col < width:
                        pixel = img.getpixel((src_col, row))
                        if pixel > 127:
                            # 使用彩虹颜色
                            hue = ((col + row) / (SCREEN_COLS + SCREEN_ROWS)) % 1.0
                            r, g, b = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
                            board_idx, pixel_idx = get_pixel_index(col, row)
                            pixel_objects[board_idx][pixel_idx] = (r, g, b)

            for pixels in pixel_objects:
                pixels.show()
            time.sleep(speed)

    # 最后保持显示
    time.sleep(2)


def animated_display(img: Image.Image, duration: float = 5.0):
    """
    带边框动画的图片显示
    - 背景：黑色
    - 内容：彩虹彩色渐变
    边框颜色会持续流动
    """
    # 转换为灰度模式
    if img.mode != 'L':
        img = img.convert('L')

    img = img.resize((SCREEN_COLS, SCREEN_ROWS), Image.Resampling.NEAREST)

    start_time = time.time()
    frame = 0

    while time.time() - start_time < duration:
        # 清除屏幕
        for pixels in pixel_objects:
            pixels.fill(0)

        # 显示图片（彩虹彩色）
        for row in range(SCREEN_ROWS):
            for col in range(SCREEN_COLS):
                pixel = img.getpixel((col, row))
                if pixel > 127:
                    # 使用彩虹颜色
                    hue = ((col + row) / (SCREEN_COLS + SCREEN_ROWS)) % 1.0
                    r, g, b = hsv_to_rgb(hue, 1.0, BRIGHTNESS)
                    board_idx, pixel_idx = get_pixel_index(col, row)
                    pixel_objects[board_idx][pixel_idx] = (r, g, b)

        # 动态边框
        draw_border(hue_offset=frame / 30)

        for pixels in pixel_objects:
            pixels.show()

        time.sleep(0.033)  # 30 FPS
        frame += 1


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(description="LED 图片显示工具")
    parser.add_argument("--prompt", "-p", type=str, help="图片描述提示词")
    parser.add_argument("--text", "-t", type=str, help="生成英文文字图片")
    parser.add_argument("--image", "-i", type=str, help="使用本地图片文件")
    parser.add_argument("--api-key", type=str, help="MiniMax API Key")
    parser.add_argument("--duration", type=float, default=5.0, help="显示时长（秒）")
    parser.add_argument("--no-border", action="store_true", help="不显示边框")
    parser.add_argument("--scroll", action="store_true", help="滚动显示")
    parser.add_argument("--brightness", type=float, default=0.4, help="亮度 0.1-1.0")
    parser.add_argument("--list", action="store_true", help="列出 images 目录中的图片")

    args = parser.parse_args()

    global BRIGHTNESS
    BRIGHTNESS = args.brightness

    api_key = args.api_key or os.environ.get("MINIMAX_API_KEY")

    # 列出图片
    if args.list:
        images_dir = "images"
        if os.path.exists(images_dir):
            files = [f for f in os.listdir(images_dir) if f.endswith('.png') and not f.endswith('_led.png')]
            print(f"images 目录中的图片:")
            for f in sorted(files):
                print(f"  - {f}")
        else:
            print("images 目录不存在")
        return

    # 检查操作
    if not any([args.prompt, args.text, args.image]):
        print("请指定操作:")
        print("  python led_image_display.py -p 'cute cat'          # 生成并显示")
        print("  python led_image_display.py -t 'LOVE'              # 生成文字并显示")
        print("  python led_image_display.py -i heart.png           # 显示本地图片")
        print("  python led_image_display.py --list                 # 列出可用图片")
        return

    try:
        # 确保 images 目录存在
        os.makedirs("images", exist_ok=True)

        # 生成或加载图片
        if args.prompt:
            # 使用 LED 优化生成
            image = generate_led_image(args.prompt, api_key, threshold=100, contrast=2.5)
            filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif args.text:
            # 生成英文文字（LED 优化）- 大字体填满屏幕
            prompt = f"'{args.text}' in VERY LARGE BOLD white letters that FILL THE ENTIRE FRAME, solid black background, simple sans-serif font, pixel art style, 8-bit style, text should occupy 90% of the image"
            image = generate_led_image(prompt, api_key, threshold=100, contrast=2.5)
            filename = f"text_{args.text.upper()}"
        elif args.image:
            # 加载本地图片并进行后处理
            if os.path.exists(args.image):
                image = Image.open(args.image)
                print(f"已加载图片: {args.image}")
                filename = os.path.splitext(os.path.basename(args.image))[0]
            elif os.path.exists(f"images/{args.image}"):
                image = Image.open(f"images/{args.image}")
                print(f"已加载图片: images/{args.image}")
                filename = os.path.splitext(os.path.basename(args.image))[0]
            else:
                print(f"图片不存在: {args.image}")
                return
            # 后处理
            image = enhance_for_led(image, threshold=100, contrast=2.5)
        else:
            print("请指定操作")
            return

        # 保存图片到 images 目录
        original_path = f"images/{filename}.png"
        image.save(original_path)
        print(f"原始图片已保存: {original_path}")

        # 显示图片
        print("正在显示到 LED 屏幕...")

        if args.scroll:
            display_image_with_scroll(image, speed=0.03)
        else:
            display_image(image, show_border=not args.no_border, border_animate=True)
            animated_display(image, duration=args.duration)

        print("显示完成")

    except ValueError as e:
        print(f"错误: {e}")
    except requests.RequestException as e:
        print(f"API 调用失败: {e}")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cleanup_leds()
        cleanup()


if __name__ == "__main__":
    main()
