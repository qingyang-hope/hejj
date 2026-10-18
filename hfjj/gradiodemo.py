"""
通义千问Turbo聊天机器人 - Gradio界面实现
概述：
本模块创建了一个基于Gradio的Web界面，用于与通义千问Turbo大语言模型进行交互。
通过阿里云DashScope API（兼容OpenAI接口的服务）调用通义千问模型，
提供了一个用户友好的聊天界面，支持对话历史记录管理和上下文理解。

主要功能：
1. 创建Gradio聊天界面，支持用户与AI模型的交互
2. 管理聊天历史记录，维护对话上下文
3. 处理不同格式的历史消息，确保兼容性
4. 提供API密钥验证和错误处理机制
5. 支持自定义示例问题

依赖项：
- gradio: 用于创建交互式Web界面
- openai: 用于调用兼容OpenAI格式的API
- os: 用于读取环境变量

环境变量要求：
- DASHScope_API_KEY: 阿里云DashScope平台的API密钥，用于身份验证和模型访问

使用方法：
1. 设置环境变量: set DASHScope_API_KEY=您的API密钥
2. 运行脚本: python GradioDemo.py
3. 在浏览器中访问: http://127.0.0.1:7860

使用示例：
# 在Windows上设置环境变量
# set DASHScope_API_KEY=your_actual_api_key
# 在Linux/Mac上设置环境变量
# export DASHScope_API_KEY=your_actual_api_key
# 运行应用程序
# python GradioDemo.py

注意事项：
1. API密钥安全: DASHScope_API_KEY是敏感信息，请妥善保管，不要提交到版本控制系统
2. 网络连接: 使用时需要稳定的互联网连接
3. 配额限制: 通义千问API可能有调用次数和速率限制，请参考官方文档
4. 错误处理: 如果遇到连接错误，请检查网络连接和API密钥设置
5. 消息格式: 系统会自动处理不同格式的历史消息，但建议使用标准格式

常见问题：
- 无法连接API: 检查网络连接和API密钥
- 响应时间长: 可能是模型负载高，请耐心等待
- 内容生成错误: 可能是请求格式问题，尝试简化输入内容
"""

# 导入必要的库
import gradio as gr          # Gradio库，用于创建Web界面
import os                   # 用于访问环境变量
from openai import OpenAI   # OpenAI兼容API的客户端库，用于调用通义千问

# 从系统环境变量中获取DashScope API密钥
# 确保在运行此脚本前已设置DASHScope_API_KEY环境变量
api_key = os.getenv("DASHScope_API_KEY")


def call_qwen(message, history):
    """
    调用通义千问max模型的函数
    参数:
    message (str): 用户当前输入的消息内容
    history (list): 聊天历史记录，支持两种格式：
        - 格式1: [(用户消息, 助手回复), ...] - 元组列表形式
        - 格式2: [{"role": "user", "content": "消息内容"}, ...] - 字典列表形式
    返回:
    str: 模型生成的回复内容，如果发生错误则返回格式化的错误信息

    功能说明:
    1. 验证API密钥是否存在，确保服务可用性
    2. 创建OpenAI客户端，配置DashScope的兼容模式API端点
    3. 构建包含历史对话和当前消息的完整消息列表，维护对话上下文
    4. 处理不同格式的历史消息，确保兼容性
    5. 调用通义千问模型qwen-max生成回复
    6. 捕获并处理可能的异常，返回友好的错误信息

    错误处理:
    - API密钥不存在：返回错误提示，指导用户设置环境变量
    - 历史记录格式错误：尝试多种格式解析，出错时记录日志但不中断执行
    - API调用失败：返回原始错误信息，便于调试
    """
    # 检查API密钥是否存在，确保服务可用性
    # 如果未设置API密钥，立即返回错误提示
    if not api_key:
        return "错误: 未设置DASHScope_API_KEY环境变量，请设置后重试。"

    # 初始化OpenAI客户端，使用DashScope的兼容模式API端点
    # 这个初始化步骤配置了API密钥和基础URL，使得我们可以通过标准OpenAI接口调用阿里云的服务
    client = OpenAI(
        api_key=api_key,                # 使用环境变量中的API密钥进行身份验证
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # DashScope兼容模式端点
    )

    # 构建消息列表，用于维持对话上下文
    # 这个列表将被传递给模型，包含完整的对话历史和当前消息
    messages = []

    # 如果存在历史对话记录，将其添加到消息列表中
    if history:
        # 遍历历史记录，正确处理Gradio ChatInterface的消息格式
        # 这里添加了异常处理，确保即使历史记录格式不正确也不会导致程序崩溃
        try:
            # 尝试处理字典格式的history（较新版本Gradio的格式）
            for msg in history:
                # 检查是否为字典格式且包含必要的'role'和'content'字段
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append(msg)
                # 检查是否为元组或列表格式（旧版本Gradio的格式）
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    # 将元组格式转换为API需要的字典格式
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            # 如果处理历史记录时出错，打印错误信息但继续执行
            # 这确保了即使历史记录处理失败，用户也能继续与模型交互
            print(f"处理历史记录时出错: {e}")

    # 添加当前用户的最新消息到消息列表中
    # 这确保了模型能够接收到用户的最新请求
    messages.append({"role": "user", "content": message})

    try:
        # 使用qwen-max模型，这是通义千问系列中的高性能版本
        response = client.chat.completions.create(
            model="qwen-max",       # 模型名称
            messages=messages,      # 传递完整的对话历史和当前消息
            stream=False            # 设置为非流式响应（一次性返回完整结果）
        )
        # 提取并返回模型生成的回复内容
        return response.choices[0].message.content
    except Exception as e:
        # 捕获并处理所有可能的异常，返回友好的错误信息
        return "Error: " + str(e)


# 使用ChatInterface组件，这是Gradio提供的专门用于创建聊天界面的组件
demo = gr.ChatInterface(
    fn=call_qwen,                       # 指定处理聊天消息的回调函数，将调用通义千问API
    title="通义千问-max",                # 界面标题
    description="基于通义千问max的聊天机器人",  # 界面描述
    # 示例问题列表，供用户快速体验
    examples=[
        ["你好"],
        ["你叫什么名字？"],
        ["给我讲一个笑话呗"]
    ]
)

# 主程序入口点
# 当直接运行此脚本时，启动Gradio Web服务器
if __name__ == "__main__":
    # 启动Gradio服务，默认监听本地7860端口
    demo.launch(theme=gr.themes.Soft())
