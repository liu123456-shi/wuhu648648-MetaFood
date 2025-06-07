# git
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

model_path = 'tencent/Hunyuan3D-2'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
rembg = BackgroundRemover()

food_list = [["bagel", "cream_cheese"], ["breaded_fish", "lemon", "broccoli"], ["burger", "hot_dog"],
             ["cheesecake", "strawberry", "raspberry"], ["energy_bar", "cheddar_cheese", "banana"],
             ["grilled_salmon", "broccoli"], ["pasta", "garlic_bread"],
             ["pbj", "carrot_stick", "apple", "celery"], ["pizza", "chicken_wing"],
             ["quesadilla", "guacamole", "salsa"], ["roast_chicken_leg", "biscuit"], ["sandwich", "cookie"],
             ["steak", "mashed_potatoes"], ["toast", "sausage", "fried_egg"]]

for idx, sublist in enumerate(food_list, start=1):
    folder_name = f"../food_mask/{idx}"
    for img_name in sublist:
        image_path = f"{folder_name}/{img_name}_cropped.jpg"
        image = Image.open(image_path).convert("RGBA")
        pixels = image.load()
        width, height = image.size
        # 将黑色背景替换为透明背景
        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                if r == 0 and g == 0 and b == 0:
                    pixels[x, y] = (0, 0, 0, 0)
        if image.mode == 'RGB':
            image = rembg(image)

        mesh = pipeline_shapegen(image=image)[0]
        # mesh = pipeline_texgen(mesh, image=image)
        obj_path = f"{folder_name}/{img_name}.obj"
        mesh.export(obj_path)