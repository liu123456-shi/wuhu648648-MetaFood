from PIL import Image

food_list = [["bagel", "cream_cheese"], ["breaded_fish", "lemon", "broccoli"], ["burger", "hot_dog"],
             ["cheesecake", "strawberry", "raspberry"], ["energy_bar", "cheddar_cheese", "banana"],
             ["grilled_salmon", "broccoli"], ["pasta", "garlic_bread"],
             ["pbj", "carrot_stick", "apple", "celery"], ["pizza", "chicken_wing"],
             ["quesadilla", "guacamole", "salsa"], ["roast_chicken_leg", "biscuit"], ["sandwich", "cookie"],
             ["steak", "mashed_potatoes"], ["toast", "sausage", "fried_egg"]]

for idx, sublist in enumerate(food_list, start=1):
    folder_name = f"./food_mask/{idx}-{'_'.join(sublist)}"
    for img_name in sublist:
        img_path = f"{folder_name}/{img_name}.jpg"
        try:
            img = Image.open(img_path)
            pixels = img.load()
            width, height = img.size
            left, right = width, 0
            top, bottom = height, 0

            for x in range(width):
                for y in range(height):
                    # Check if pixel is not black (assuming no transparency or ignoring transparency)
                    if pixels[x, y][:3] != (0, 0, 0):
                        if x < left:
                            left = x
                        if x > right:
                            right = x
                        if y < top:
                            top = y
                        if y > bottom:
                            bottom = y

            if left < right and top < bottom:
                cropped_img = img.crop((left, top, right, bottom))
                cropped_path = f"{folder_name}/{img_name}_cropped.jpg"
                cropped_img.save(cropped_path)
                print(f"Successfully processed: {img_path}")
            else:
                print(f"Warning: No valid subject area found in {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
