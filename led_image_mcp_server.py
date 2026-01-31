"""
MCP服务器 - AI图片生成与显示服务
使用 mcp (官方SDK) 实现SSE传输
结合 MiniMax API 图片生成 和 LED 显示功能
"""
import asyncio
import sys
import os
import signal
import time
import threading
import asyncio
import json
import base64
import requests
from queue import Queue
from typing import Optional
from datetime import datetime
from io import BytesIO

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write
from PIL import Image, ImageEnhance, ImageFilter
import adafruit_pixelbuf

import config

# 强制行缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
PID_FILE = "/tmp/led_image_mcp_server.pid"


def check_previous_instance():
    """检查是否有之前的实例在运行"""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"关闭之前的进程 (PID: {old_pid})...")
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(0.5)
            except (OSError, ProcessLookupError):
                pass
        except (ValueError, IOError):
            pass

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


def cleanup():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def cleanup_leds():
    try:
        for pixels in pixel_objects:
            pixels.fill(0)
            pixels.show()
    except Exception:
        pass


check_previous_instance()
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_leds(), cleanup(), sys.exit(0)))
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_leds(), cleanup(), sys.exit(0)))


# ========== 配置区域 ==========
BOARD_ROWS = 5
BOARD_COLS = 1
LED_ROWS_PER_BOARD = 8
LED_COLS_PER_BOARD = 32

PINS = config.PINS

# LED显示亮度 (0.0-1.0)
BRIGHTNESS = 0.1

# MiniMax API 配置
API_URL = "https://api.minimaxi.com/v1/image_generation"
# API Key (可以通过环境变量 MINIMAX_API_KEY 设置)
DEFAULT_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

# LED 优化提示词后缀
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

SCREEN_COLS = BOARD_COLS * LED_COLS_PER_BOARD  # 32
SCREEN_ROWS = BOARD_ROWS * LED_ROWS_PER_BOARD  # 40
PIXELS_PER_BOARD = LED_ROWS_PER_BOARD * LED_COLS_PER_BOARD  # 256


# ========== LED 初始化 ==========
class Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    def __init__(self, pin, size, **kwargs):
        self._pin = pin
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin, buf)


pixel_objects = []
for pin in PINS:
    pixels = Pi5Pixelbuf(pin, PIXELS_PER_BOARD, auto_write=False, byteorder="GRB", brightness=BRIGHTNESS)
    pixel_objects.append(pixels)


# ========== LED 显示辅助函数 ==========
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


# ========== 图片生成函数 ==========
def generate_image(prompt: str, api_key: str = None) -> Image.Image:
    """调用 MiniMax API 生成图片"""
    key = api_key or DEFAULT_API_KEY
    if not key:
        raise ValueError("请设置 MiniMax API Key (通过环境变量 MINIMAX_API_KEY)")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "image-01",
        "prompt": prompt,
        "aspect_ratio": "1:1",
        "response_format": "base64"
    }

    print(f"[图片生成] 正在生成图片: {prompt[:50]}...")
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
    print(f"[图片生成] 成功！原始尺寸: {image.size}，耗时: {elapsed:.2f}秒")

    return image


def enhance_for_led(image: Image.Image, threshold: int = 100, contrast: float = 2.5) -> Image.Image:
    """
    增强图片以适应 LED 显示
    - 灰度转换
    - 对比度增强
    - 二值化
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


def generate_led_image(prompt: str, api_key: str = None, threshold: int = 100, contrast: float = 2.5) -> Image.Image:
    """生成适合 LED 显示的图片（带后处理）"""
    # 添加 LED 优化后缀
    led_prompt = prompt + LED_PROMPT_SUFFIX

    # 生成图片
    image = generate_image(led_prompt, api_key=api_key)

    # 后处理
    image = enhance_for_led(image, threshold=threshold, contrast=contrast)

    return image


# ========== 图片显示函数 ==========
def display_image(img: Image.Image, duration: float = 5.0, show_border: bool = True, animate_border: bool = True):
    """
    将图片显示到 LED 屏幕
    - 背景：黑色（不亮）
    - 内容：彩虹彩色渐变
    - 边框：动态彩虹效果
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

    # 显示内容
    for pixels in pixel_objects:
        pixels.show()

    # 边框动画
    if show_border and animate_border:
        border_start_time = time.time()
        frame = 0
        while time.time() - border_start_time < duration:
            draw_border(hue_offset=frame / 30)
            for pixels in pixel_objects:
                pixels.show()
            time.sleep(0.033)  # 30 FPS
            frame += 1


def save_image(img: Image.Image, prompt: str) -> str:
    """保存图片到 images 目录"""
    os.makedirs("images", exist_ok=True)

    # 清理提示词用于文件名
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:20])
    filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.png"

    # 保存原始LED尺寸图片
    save_path = f"images/{filename}"
    img.save(save_path)
    print(f"[保存] 图片已保存: {save_path}")

    return filename


# ========== LED 工作线程 + 任务队列 ==========
led_task_queue = Queue()  # 图片显示任务队列
image_gen_queue = Queue()  # 图片生成任务队列
led_worker_thread = None
worker_running = False
current_task_type = None  # 'text' 或 'image'
image_gen_in_progress = False  # 是否有图片正在生成


def led_worker():
    """LED工作线程：从队列中取出任务并执行"""
    global worker_running, current_task_type, image_gen_in_progress
    worker_running = True

    while worker_running:
        try:
            # 首先处理图片生成任务（如果有的话）
            if not image_gen_in_progress:
                try:
                    gen_task = image_gen_queue.get_nowait()
                    task_type, params = gen_task

                    if task_type == 'generate':
                        # 生成图片任务：params = (prompt, api_key, duration)
                        prompt, api_key, duration = params
                        print(f"[LED工作线程] 开始生成图片: {prompt[:30]}...", flush=True)
                        image_gen_in_progress = True

                        try:
                            # 生成图片
                            image = generate_led_image(
                                prompt,
                                api_key=api_key,
                                threshold=100,
                                contrast=2.5
                            )
                            # 保存图片
                            filename = save_image(image, prompt)
                            print(f"[LED工作线程] 图片生成完成，开始显示", flush=True)

                            # 将显示任务加入主队列
                            while not led_task_queue.empty():
                                try:
                                    led_task_queue.get_nowait()
                                    led_task_queue.task_done()
                                except:
                                    break
                            led_task_queue.put(('image', (None, image, duration)))
                            print(f"[LED工作线程] 显示任务已加入队列", flush=True)
                        except Exception as e:
                            print(f"[LED工作线程] 图片生成失败: {e}", flush=True)
                        finally:
                            image_gen_in_progress = False
                            image_gen_queue.task_done()

                    continue  # 继续检查下一个生成任务
                except:
                    pass

            # 处理显示任务
            try:
                task = led_task_queue.get(timeout=0.5)
            except:
                continue

            if task is None:
                break

            task_type, params = task
            current_task_type = task_type

            try:
                if task_type == 'image':
                    _, image, duration = params
                    display_image(image, duration=duration, show_border=True, animate_border=True)
                elif task_type == 'text':
                    # 文字显示功能由 led_mcp_server.py 提供，这里保留接口
                    pass
            except Exception as e:
                print(f"[LED工作线程] 执行任务错误: {e}")
            finally:
                current_task_type = None  # 任务完成后清空
                led_task_queue.task_done()

        except Exception:
            pass


def start_led_worker():
    """启动LED工作线程"""
    global led_worker_thread, worker_running

    if led_worker_thread is not None and led_worker_thread.is_alive():
        return

    worker_running = False
    led_worker_thread = threading.Thread(target=led_worker, daemon=True)
    led_worker_thread.start()
    print("[LED] 工作线程已启动", flush=True)


def stop_led_worker():
    """停止LED工作线程"""
    global worker_running

    if led_worker_thread is None or not led_worker_thread.is_alive():
        return

    worker_running = False
    led_task_queue.put(None)
    led_worker_thread.join(timeout=2.0)
    print("[LED] 工作线程已停止", flush=True)


def queue_image_task(image: Image.Image, duration: float = 5.0):
    """将图片显示任务加入队列"""
    while not led_task_queue.empty():
        try:
            led_task_queue.get_nowait()
            led_task_queue.task_done()
        except Exception:
            break

    led_task_queue.put(('image', (None, image, duration)))
    print(f"[MCP] 图片显示任务已加入队列", flush=True)


def queue_image_gen_task(prompt: str, api_key: str = None, duration: float = 5.0):
    """将图片生成任务加入队列（异步处理，MCP调用立即返回）"""
    # 清空队列中旧的生成任务
    while not image_gen_queue.empty():
        try:
            image_gen_queue.get_nowait()
            image_gen_queue.task_done()
        except Exception:
            break

    image_gen_queue.put(('generate', (prompt, api_key, duration)))
    print(f"[MCP] 图片生成任务已加入队列: {prompt[:30]}...", flush=True)


# ========== 创建MCP服务器 ==========
app = Server("led-image-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用工具"""
    return [
        Tool(
            name="generate_and_display_image",
            description="根据提示词生成AI图片并显示到LED屏幕。会自动添加LED优化提示词，确保黑色背景和白色内容。图片内容会显示为彩虹彩色渐变效果。",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图片描述提示词（英文效果更好），例如 'cute cat', 'beautiful flower', 'romantic heart'",
                        "examples": ["cute cat", "beautiful flower", "romantic heart"]
                    },
                    "api_key": {
                        "type": "string",
                        "description": "MiniMax API Key（可选，默认使用环境变量 MINIMAX_API_KEY）"
                    },
                    "duration": {
                        "type": "integer",
                        "description": "显示持续时间（秒），默认5秒",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 60
                    },
                    "threshold": {
                        "type": "integer",
                        "description": "二值化阈值（0-255），默认100。较低的值会让更多像素变亮",
                        "default": 100,
                        "minimum": 0,
                        "maximum": 255
                    },
                    "contrast": {
                        "type": "number",
                        "description": "对比度增强倍数，默认2.5。较高的值会让内容更清晰",
                        "default": 2.5,
                        "minimum": 1.0,
                        "maximum": 5.0
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="generate_text_image",
            description="生成英文文字图片并显示到LED屏幕。文字会自动优化为适合LED显示的风格。",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要显示的英文文字（建议2-4个字符），例如 'LOVE', 'HAPPY', 'SUN'",
                        "examples": ["LOVE", "HAPPY", "SUN"]
                    },
                    "api_key": {
                        "type": "string",
                        "description": "MiniMax API Key（可选，默认使用环境变量 MINIMAX_API_KEY）"
                    },
                    "duration": {
                        "type": "integer",
                        "description": "显示持续时间（秒），默认5秒",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 60
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_status",
            description="获取LED图片服务器状态（快速查询，不阻塞）",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="check_generation_status",
            description="检查图片生成和显示状态。如果未完成，会阻塞等待3秒后返回当前状态。可用于轮询等待图片生成完成。",
            inputSchema={
                "type": "object",
                "properties": {
                    "wait_timeout": {
                        "type": "integer",
                        "description": "最大等待时间（秒），默认30秒。超过此时间无论是否完成都会返回",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 120
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """调用工具"""
    print(f"[MCP] 调用工具: {name}, 参数: {arguments}", flush=True)

    if name == "generate_and_display_image":
        prompt = arguments.get("prompt", "")
        api_key = arguments.get("api_key") or None
        duration = arguments.get("duration", 5)

        if not prompt:
            result = {"success": False, "error": "请输入提示词"}
        else:
            # 异步处理：将生成任务加入队列
            queue_image_gen_task(prompt, api_key=api_key, duration=duration)

            # 延迟2秒后再返回，让任务有时间执行
            await asyncio.sleep(2)

            result = {
                "success": True,
                "message": f"已收到图片生成请求，正在后台处理: {prompt[:30]}...",
                "prompt": prompt,
                "duration": duration,
                "status": "queued"
            }

    elif name == "generate_text_image":
        text = arguments.get("text", "").upper()
        api_key = arguments.get("api_key") or None
        duration = arguments.get("duration", 5)

        if not text:
            result = {"success": False, "error": "请输入文字内容"}
        else:
            # 生成文字图片提示词 - 强调大字体填满屏幕
            prompt = f"'{text}' in VERY LARGE BOLD white letters that FILL THE ENTIRE FRAME, solid black background, simple sans-serif font, pixel art style, 8-bit style, text should occupy 90% of the image"

            # 异步处理：将生成任务加入队列
            queue_image_gen_task(prompt, api_key=api_key, duration=duration)

            # 延迟2秒后再返回，让任务有时间执行
            await asyncio.sleep(2)

            result = {
                "success": True,
                "message": f"已收到文字图片生成请求，正在后台处理: {text}...",
                "text": text,
                "duration": duration,
                "status": "queued"
            }

    elif name == "get_status":
        queue_size = led_task_queue.qsize() if led_task_queue else 0
        gen_queue_size = image_gen_queue.qsize() if image_gen_queue else 0
        result = {
            "status": "ok",
            "message": "LED图片服务器运行中",
            "display_queue_size": queue_size,
            "generation_queue_size": gen_queue_size,
            "generation_in_progress": image_gen_in_progress,
            "worker_running": worker_running,
            "brightness": BRIGHTNESS,
            "screen_size": f"{SCREEN_COLS}x{SCREEN_ROWS}",
            "current_task": current_task_type
        }

    elif name == "check_generation_status":
        """检查图片生成状态，如果未完成则等待1秒后返回"""
        wait_timeout = arguments.get("wait_timeout", 30)
        start_time = time.time()
        check_count = 0

        while True:
            check_count += 1
            gen_queue_size = image_gen_queue.qsize() if image_gen_queue else 0

            # 判断是否完成：生成队列为空且没有正在进行的生成任务
            # 图片生成完成就认为完成，不用等显示完成
            is_complete = (
                not image_gen_in_progress and
                gen_queue_size == 0
            )

            if is_complete:
                result = {
                    "status": "completed",
                    "message": "图片生成已完成",
                    "check_count": check_count
                }
                break

            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed >= wait_timeout:
                result = {
                    "status": "timeout",
                    "message": "等待超时，任务仍在进行中",
                    "generation_in_progress": image_gen_in_progress,
                    "display_queue_size": queue_size,
                    "generation_queue_size": gen_queue_size,
                    "elapsed_seconds": int(elapsed),
                    "check_count": check_count
                }
                break

            # 如果是第一次检查且未完成，等待1秒后返回友好提示
            if check_count == 1:
                print(f"[状态检查] 任务未完成，等待1秒后重试...", flush=True)
                await asyncio.sleep(1)
                result = {
                    "status": "generating",
                    "message": "图片生成中，请稍等...",
                    "generation_in_progress": image_gen_in_progress,
                    "check_count": check_count
                }
                break

        print(f"[状态检查] 结果: {result['status']}, 检查次数: {check_count}", flush=True)

    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# ========== 运行服务器 ==========
async def main():
    """主函数"""
    from mcp.server.sse import SseServerTransport
    from starlette.responses import Response

    # 启动LED工作线程
    start_led_worker()

    # SSE传输
    sse_transport = SseServerTransport("/messages")

    async def asgi_handler(scope, receive, send):
        """ASGI处理函数"""
        if scope["type"] == "lifespan":
            return
        elif scope["type"] == "http":
            path = scope.get("path", "")
            print(f"[ASGI] path={path}", flush=True)

            if path == "/sse" or path.startswith("/sse?"):
                # SSE连接
                async with sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
                    await app.run(read_stream, write_stream, app.create_initialization_options())
            elif path == "/messages" or path.startswith("/messages?"):
                # 消息端点
                await sse_transport.handle_post_message(scope, receive, send)
            else:
                # 未知路径
                await Response("Not Found", status_code=404)(scope, receive, send)

    import uvicorn
    config = uvicorn.Config(asgi_handler, host="0.0.0.0", port=8081, log_level="info")
    server = uvicorn.Server(config)

    print("=" * 50, flush=True)
    print("MCP LED Image Server running on http://0.0.0.0:8081", flush=True)
    print("=" * 50, flush=True)
    print("Endpoints:", flush=True)
    print("  GET  /sse      - MCP SSE stream", flush=True)
    print("  POST /messages - MCP message endpoint", flush=True)
    print("", flush=True)
    print("Available tools:", flush=True)
    print("  generate_and_display_image - 生成AI图片并显示", flush=True)
    print("  generate_text_image        - 生成文字图片并显示", flush=True)
    print("  get_status                 - 获取服务器状态", flush=True)
    print("", flush=True)
    print("Config:", flush=True)
    print(f"  Brightness: {BRIGHTNESS*100:.0f}%", flush=True)
    print(f"  Screen size: {SCREEN_COLS}x{SCREEN_ROWS}", flush=True)
    print(f"  API Key: {'已设置' if DEFAULT_API_KEY else '未设置 (需设置环境变量)'}", flush=True)
    sys.stdout.flush()

    await server.serve()


if __name__ == "__main__":
    try:
        print("Starting LED Image MCP Server...", flush=True)
        sys.stdout.flush()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        stop_led_worker()
        cleanup_leds()
        cleanup()
