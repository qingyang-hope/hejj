
#案例1 一个接收文本输入并返回该文本倒序输出的应用。 
# import gradio as gr

# def reverse_text(text):
#     return text[::-1]

# demo = gr.Interface(fn=reverse_text, inputs="text", outputs="text")

# demo.launch()

# # 更新界面布局 
# import gradio as gr

# # 这是修改后的函数
# def reverse_and_count(text):
#     reversed_text = text[::-1]
#     length = len(text)
#     return reversed_text, length

# demo = gr.Interface(
#     fn=reverse_and_count,
#     inputs="text",
#     # flagging_mode="never",
#     outputs=["text", "number"],  # 第一个输出是文本，第二个输出是一个数字
#     title="文本处理工具",  # 设置页面标题
#     description="输入一段文字，查看其倒序形式及字符数。",  # 添加简短说明
#     examples=[["你好，世界"], ["Hello World"]]
# )

# demo.launch()


#案例2 使用 Image 组件输入图像，并将其转换为铅笔素描。

import gradio as gr
import numpy as np
import cv2  

def image_to_sketch(image):
    gray_image = image.convert('L')
    inverted_image = 255 - np.array(gray_image)
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blurred = 255 - blurred
    pencil_sketch = cv2.divide(np.array(gray_image), inverted_blurred, scale=256.0)
    return pencil_sketch

demo = gr.Interface(
    fn=image_to_sketch,
    inputs=[gr.Image(label="Input Image", type="pil")],
    outputs=[gr.Image(label="Pencil Sketch")],
    title="Image to Pencil Sketch",
    description="Convert an input image to a pencil sketch."
)

demo.launch()
