import json
import os
import base64
from openai import OpenAI
from openpyxl.packaging.manifest import mimetypes
import cv2
# 图片路径
local_image_path = "./food_1/food_combos/4-cheesecake_strawberry_raspberry/cheesecake_masked.jpg"
# 食物名称列表
food_names = ["cheesecake", "strawberry", "raspberry"]

# 读取图片文件内容（二进制模式）
try:
    with open(local_image_path, "rb") as image_file:
        image_data = image_file.read()
except FileNotFoundError:
    print(f"错误：找不到文件 {local_image_path}")
    exit()

# 将图片数据编码为 Base64 字符串
base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
mime_type, _ = mimetypes.guess_type(local_image_path)
image_url_data = f"data:{mime_type};base64,{base64_encoded_image}"

# 系统提示
system_prompt = """
{
  "cheesecake": ["cheesecake", "cream cheese", "graham cracker crust", "dessert"],
  "strawberry": ["strawberry", "fruit", "strawberry shortcake", "strawberry jam"],
  "raspberry": ["raspberry", "fruit", "raspberry jam", "raspberry sauce"]
}
请根据提供的图片，判断图片中的食物类别属于以下哪个列表项：cheesecake, strawberry, raspberry。
输出仅为食物类别名称，如 "cheesecake" 或 "strawberry" 或 "raspberry"。
"""

# 文本信息
text = f"图片对应的食物可能是 {', '.join(food_names)}"

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-ab02fa43004641d29cb7ea8eb19b36ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

try:
    # 调用 API
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image_url_data}}]}
        ]
    )

    if completion.choices:
        message_content = completion.choices[0].message.content
        print(f"图片中的食物类别是: {message_content}")
    else:
        print("错误：API响应中未找到任何 'choices'。")
        print("完整响应:", completion)

except Exception as e:
    print(f"调用API时出错: {e}")

    
image = cv2.imread(local_image_path)
new_image_path = "./food_1/food_combos/4-cheesecake_strawberry_raspberry/"+message_content+".jpg"
print(new_image_path)

cv2.imwrite(new_image_path, image)