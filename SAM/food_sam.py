import argparse
import os
import sys
import torch
from PIL import Image
import json
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from openpyxl.packaging.manifest import mimetypes
import base64
import glob
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, text_prompt, original_image):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    mask_img_bin = torch.zeros(mask_list.shape[-2:])
    masked_original_image = original_image.copy()
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        mask_img_bin[mask.cpu().numpy()[0] == True] = 1
        masked_original_image[mask.cpu().numpy()[0] == False] = 0

    # np.save(os.path.join(output_dir, 'mask_bin.npy'), mask_img_bin)
    cv2.imwrite(os.path.join(output_dir, f"{text_prompt}_masked.jpg"), cv2.cvtColor(masked_original_image, cv2.COLOR_RGB2BGR))
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    # plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f"{text_prompt}_mask.json"), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    base_dir = "../food_mask/"
    api_key = ""# 你的api_key
    image_extensions = ['*.JPG', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    all_image_files = []

    for sub_dir in os.listdir(base_dir):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for ext in image_extensions:
                image_pattern = os.path.join(sub_dir_path, ext)
                image_files = glob.glob(image_pattern)
                all_image_files.extend(image_files)
                for img_file in image_files:
                    print(f"local_image_path = \"{img_file}\"")
    for local_image_path in all_image_files:
        full_dir_path = os.path.dirname(local_image_path)
        name = os.path.basename(full_dir_path)
        system_prompt1 = """
        # 角色：基于常识的食物名称分割器
        ## 背景：
        你需要处理一个描述食物组合的字符串。这个字符串有时会用下划线连接一个食物名称中的多个单词（如 `carrot_stick`），有时下划线则用来分隔不同的食物（如 `apple_celery` 中的下划线）。字符串开头的数字和连字符（如 `8-`）只是一个标号，没有实际意义。
        ## 任务：
        根据对常见食物名称的理解，智能地将输入的字符串（**忽略开头的标号部分**）分割成一个代表**独立、有意义的食物种类**的列表。分割的最终结果应该尽可能地模拟真正在图片中区分不同食物种类的过程。
        ## 核心规则：
        1.  **忽略标号**: 字符串开头由数字和连字符构成的部分（例如 `8-`, `5-`）**仅为标号，与食物种类数量或分割逻辑无关，应完全忽略**。处理时，请只关注连字符之后的部分。
        2.  **基于常识分割**: 分割的核心依据是**识别常见的食物名称**。
        3.  **处理下划线**:
            * 当下划线连接的单词组合起来构成一个**常见的、单一的食物名称**时（例如 `carrot_stick`, `energy_bar`, `peanut_butter`, `apple_slices`, `potato_chips`, `turkey_sandwich`, `cheddar_cheese`, `juice_box` 等），应将它们**合并**视为一个独立的食物条目。
            * 当下划线分隔的是**明显不同的食物种类**时，或者连接的单词**不构成**一个公认的、单一的食物实体时，下划线应作为**分隔符**，区分不同的食物条目。
        4.  **目标**: 输出的列表应准确反映字符串中描述的、最符合人们日常认知的独立食物种类。
        ## 输入格式示例：
        `<标号>-<名称部分1>_<名称部分2>_<名称部分3>...`

        ## 输出格式要求：
        一个包含识别出的独立食物名称字符串的列表，例如：`["食物种类1", "食物种类2_合并词", "食物种类3", ...]`
        * **无额外内容**：除了必需的字符串外，不应包含任何介绍、解释、注释或其他文字。
        ## 示例：

        ### 示例 1 (核心问题示例):
        输入: `8-pbj_carrot_stick_apple_celery`
        输出: `"pbj", "carrot_stick", "apple", "celery"
        (解释：忽略 "8-"。基于常识，"pbj"（花生酱果酱三明治）、"apple"（苹果）、"celery"（芹菜）是独立食物。"carrot_stick"（胡萝卜条）是一个常见的单一食物名称，因此合并为一个条目，而不是分成 "carrot" 和 "stick"。)

        """
        system_prompt2 = """
        # 角色：食物名称关联器
        ## 任务：
        根据提供的食物名称（可能来自文件夹名称或标签）和相关的食物图片，为每个食物项生成一个**独特且具体**的相关食物术语列表。

        ## 处理流程：
        1.  **解析输入**：
            * 根据提供的名称和图片，准确识别其代表的核心食物或菜肴。
            * 如果是缩写（如 'pbj'），推断出其完整且最常用的名称。
            * 如果是描述性名称（如 'carrot_stick'），识别出主要的食物成分和可能的制备方式。
        2.  **生成关联词**：基于解析出的食物信息，创建一个相关食物术语的**唯一**列表，应优先包含以下类型的术语：
            * **完整/标准名称**：该食物最常用或标准的名称（例如，'pbj' -> 'Peanut Butter and Jelly sandwich'）。如果输入名称已经是标准名称，则包含它。
            * **关键/定义性成分**：制作该食物或构成该食物核心的关键原料（例如，'pbj' -> 'peanut butter', 'jelly', 'bread'；'carrot_stick' -> 'carrot'；'apple_pie' -> 'apple', 'pie crust'）。
            * **具体的相关菜品/制备方式**：直接使用该食物制作的特定常见菜肴、变体或特定的制备形式（例如，'apple' -> 'apple pie', 'apple sauce', 'baked apple'；'carrot_stick' -> 'raw carrot', 'sliced carrot'）。
            * **特定的食物子分类**（可选，仅当有助于区分时）：如果适用，可以包含更具体的食物类别（例如，'cheddar_cheese' -> 'hard cheese', 'dairy product'）。
        3.  **约束条件**：
            * **唯一性**：生成的列表中**不得包含重复**的术语。
            * **相关性与具体性**：术语必须与核心食物**直接且具体相关**。避免过于宽泛或模糊的类别（如 'food', 'snack', 'fruit', 'yellow fruit'），除非它们是公认的标准名称的一部分（例如 'fruit salad'）。
            * **关注食物本身**：主要关联食物名称、成分和直接衍生的菜品/形式。

        ## 输出格式要求：
        * **严格的JSON格式**：输出必须是一个有效的JSON对象。
        * **键值对结构**：
            * **键 (Key)**：原始的输入名称（字符串）。
            * **值 (Value)**：一个**字符串列表**，包含所有生成的、独特的、具体的关联食物术语。
        * **无额外内容**：除了必需的JSON结构外，不应包含任何介绍、解释、注释或其他文字。

        ## 示例：
        输入为:
        多个食物名称的字符串和一张图片

        **期望的输出格式 (示例):**
        ```json
        {
          "pbj": ["Peanut Butter and Jelly sandwich", "peanut butter", "jelly", "bread", "sandwich"],
          "carrot_stick": ["carrot stick", "carrot", "raw carrot", "sliced carrot"],
          "celery": ["celery", "celery stalk"]
        }
        """
        text2 = ""
        try:
            with open(local_image_path, "rb") as image_file:
                image_data = image_file.read()
        except FileNotFoundError:
            print(f"错误：找不到文件 {local_image_path}")
            exit()
        base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(local_image_path)
        image_url_data = f"data:{mime_type};base64,{base64_encoded_image}"
        text = "文件夹名称为:" + name
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        try:
            completion = client.chat.completions.create(
                model="qwen-vl-max",  # 已更新模型名称，请根据需要调整
                messages=[
                    {"role": "system", "content": system_prompt1},
                    {"role": "user", "content": [{"type": "text", "text": text},
                                                 {"type": "image_url", "image_url": {"url": image_url_data}}]}
                ]
            )
            if completion.choices:
                message_content = completion.choices[0].message.content
                text2 = message_content
                print(f"原始响应内容:\n{message_content}")

        except Exception as e:
            print(f"调用API时出错: {e}")
        try:
            completion = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {"role": "system", "content": system_prompt2},
                    {"role": "user", "content": [{"type": "text", "text": text2},
                                                 {"type": "image_url", "image_url": {"url": image_url_data}}]}
                ]
            )
            if completion.choices:
                message_content = completion.choices[0].message.content
                print(f"原始响应内容:\n{message_content}")
                cleaned_content = message_content.strip()
                if cleaned_content.startswith("```json"):

                    cleaned_content = cleaned_content[len("```json"):].strip()
                elif cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content[len("```"):].strip()

                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-len("```")].strip()

                print(f"\n清理后的内容:\n{cleaned_content}")
                cleaned_content = eval(cleaned_content)
                keys = cleaned_content.keys()
                key_list = list(keys)
                for key in key_list:
                    print(cleaned_content[key])
            else:
                print("错误：API响应中未找到任何 'choices'。")
                print("完整响应:", completion)
        except Exception as e:
            print(f"调用API时出错: {e}")

        # cfg
        config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
        grounded_checkpoint = "models/groundingdino_swint_ogc.pth"  # change the path of the model
        sam_version = "vit_h"
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        sam_hq_checkpoint = ""
        use_sam_hq = False
        image_path =local_image_path
        text_prompt_list = key_list
        output_dir = os.path.dirname(local_image_path)
        box_threshold = 0.1
        text_threshold = 0.5
        device = "cuda"

        # make dir
        os.makedirs(output_dir, exist_ok=True)
        # load image
        original_image_pil, original_image = load_image(image_path)
        original_image_cv = cv2.imread(image_path)
        original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
        # load model
        model = load_model(config_file, grounded_checkpoint, device=device)

        current_image_pil = original_image_pil
        current_image = original_image

        for text_prompt in text_prompt_list:
            # load image
            current_image_pil, current_image = load_image(image_path)

            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, current_image, text_prompt, box_threshold, text_threshold, device=device
            )

            # initialize SAM
            if use_sam_hq:
                predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
            else:
                predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = current_image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )

            # -------------------------------
            from process import postprocess

            p = postprocess()
            pilimg = Image.open(image_path)
            color_scores = []
            color_scores_format = []
            for mask in masks:
                mask_bin = mask.cpu().numpy().squeeze().astype(int)
                ro = p.run2(pilimg, mask_bin)
                color_scores.append(ro)
                color_scores_format.append('food({:.2f})'.format(ro))
            highest_score_index = np.argmax(color_scores)
            masks = masks[highest_score_index:highest_score_index + 1]
            boxes_filt = boxes_filt[highest_score_index:highest_score_index + 1]
            pred_phrases = [pred_phrases[highest_score_index]]
            color_scores_format = [color_scores_format[highest_score_index]]
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for i, mask in enumerate(masks):
                if color_scores[highest_score_index] > 0.45:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, color_scores_format):
                show_box(box.numpy(), plt.gca(), label)
            print("boxes_filt", boxes_filt)
            txt_filename = f"{text_prompt}.txt"
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, 'w') as f:
                for box in boxes_filt:
                    box_str = ' '.join([str(x.item()) for x in box])
                    f.write(box_str + '\n')
            plt.axis('off')
            save_mask_data(output_dir, masks, boxes_filt, pred_phrases, text_prompt, original_image_cv)
            mask = masks[0].cpu().numpy().squeeze().astype(np.uint8)
            removed_image = image.copy()
            removed_image[mask == 1] = 0
            removed_image_bgr = cv2.cvtColor(removed_image, cv2.COLOR_RGB2BGR)
            removed_image_path = os.path.join(output_dir, f"{text_prompt}_removed.jpg")
            cv2.imwrite(removed_image_path, removed_image_bgr)
            cv2.imwrite(image_path, removed_image_bgr)
            system_prompt = cleaned_content
        json_str = json.dumps(system_prompt, ensure_ascii=False, indent=2)

        food_categories = list(system_prompt.keys())

        categories_str = ", ".join(food_categories)

        examples = ' 或 '.join([f'"{category}"' for category in food_categories])
        result = f"""
        {json_str}
        请根据提供的图片，判断图片中的食物类别属于以下哪个列表项：{categories_str}。
        输出仅为食物类别名称，如 {examples}。
        """
        system_prompt = result
        food_names = food_categories
        path = local_image_path
        parent_directory = os.path.dirname(path)
        for food_image_name in food_names:
            local_image_path =  parent_directory + "/"+food_image_name+"_masked.jpg"
            food_names = food_categories
            try:
                with open(local_image_path, "rb") as image_file:
                    image_data = image_file.read()
            except FileNotFoundError:
                print(f"错误：找不到文件 {local_image_path}")
                exit()
            base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(local_image_path)
            image_url_data = f"data:{mime_type};base64,{base64_encoded_image}"
            text = f"图片对应的食物可能是 {', '.join(food_names)}"
            client = OpenAI(
                api_key="sk-ab02fa43004641d29cb7ea8eb19b36ef",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            try:
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
            new_image_path = parent_directory+"/"+message_content+".jpg"
            print(new_image_path)

            cv2.imwrite(new_image_path, image)



