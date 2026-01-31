# AI LED 艺术墙

> 2025 DigiKey AI应用创意挑战赛参赛作品

## 项目介绍

**AI LED 艺术墙** 是一个将人工智能与 LED 显示技术结合的创意项目。用户可以通过自然语言与 AI 对话，AI 会调用 MCP 工具生成图片并实时显示在 LED 屏幕上。

## 核心功能

- AI 对话：与小智 AI 智能体自然对话
- 图片生成：调用 MiniMax API 实时生成 AI 图片
- LED 显示：1280 颗 WS2812B LED 组成的 32×40 显示矩阵
- 彩虹效果：动态彩虹渐变边框和内容渲染

## 硬件配置

| 组件 | 说明 |
|-----|------|
| 树莓派 5 | 主控制器，运行 MCP 服务 |
| 行空板 K10 | 运行 xiaozhi-esp32，拾音和对话 |
| WS2812B LED 灯板 | 5 块，32×40 = 1280 颗 LED |
| 5V 3A 电源 | LED 供电 |

## 运行方式

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install mcp
pip install adafruit-circuitpython-led-animation
pip install adafruit-blinka
pip install adafruit-circuitpython-raspberry-pi5-neopixel-write
pip install Pillow
pip install uvicorn

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 MiniMax API Key

# 启动服务
python led_image_mcp_server.py
```

## 项目文档

- [参赛项目文档](docs/参赛项目文档.md)
- [图片生成智能体定义](docs/图片生成智能体定义.md)

## 相关链接

- 项目讨论：[https://www.eefocus.com/forum/thread-234917-1-1.html](https://www.eefocus.com/forum/thread-234917-1-1.html)
- GitHub：[https://github.com/HonestQiao/LED-Art-Wall](https://github.com/HonestQiao/LED-Art-Wall)

## 作者

HonestQiao（乔帮主/乔大妈）- 资深嵌入式开发工程师
