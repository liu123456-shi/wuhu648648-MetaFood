import math
import trimesh
from scipy.spatial import ConvexHull


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            numbers = [float(num) for num in line.split()]
            return numbers
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
    except ValueError:
        print(f"Error: Content in file {file_path} is not valid floating-point numbers!")
    except Exception as e:
        print(f"Error: An unknown error occurred: {e}")
    return None


def rotating_calipers(hull_vertices):
    n = len(hull_vertices)
    if n < 2:
        return 0, None, None
    if n == 2:
        p1, p2 = hull_vertices[0], hull_vertices[1]
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2)), p1, p2
    max_distance = 0
    p1, p2 = None, None
    j = 1
    for i in range(n):
        next_i = (i + 1) % n
        while True:
            next_j = (j + 1) % n
            # Calculate current point pair distance
            current_distance = math.sqrt(((hull_vertices[i][0] - hull_vertices[j][0]) ** 2) +
                                         ((hull_vertices[i][1] - hull_vertices[j][1]) ** 2) +
                                         ((hull_vertices[i][2] - hull_vertices[j][2]) ** 2))
            if current_distance > max_distance:
                max_distance = current_distance
                p1, p2 = hull_vertices[i], hull_vertices[j]
            # Calculate next point pair distance
            next_distance = math.sqrt(((hull_vertices[next_i][0] - hull_vertices[next_j][0]) ** 2) +
                                      ((hull_vertices[next_i][1] - hull_vertices[next_j][1]) ** 2) +
                                      ((hull_vertices[next_i][2] - hull_vertices[next_j][2]) ** 2))
            if next_distance < current_distance:
                break
            j = next_j
    return max_distance, p1, p2


food_list = [["bagel", "cream_cheese"], ["breaded_fish", "lemon", "broccoli"], ["burger", "hot_dog"],
             ["cheesecake", "strawberry", "raspberry"], ["energy_bar", "cheddar_cheese", "banana"],
             ["grilled_salmon", "broccoli"], ["pasta", "garlic_bread"],
             ["pbj", "carrot_stick", "apple", "celery"], ["pizza", "chicken_wing"],
             ["quesadilla", "guacamole", "salsa"], ["roast_chicken_leg", "biscuit"], ["sandwich", "cookie"],
             ["steak", "mashed_potatoes"], ["toast", "sausage", "fried_egg"]]
k = 1
flag = True
r = 0.1
real_plate = 19
for idx, sublist in enumerate(food_list, start=1):
    plate_file_path = f"food_mask/{idx}-{'_'.join(sublist)}/{idx}_boxes_filt.txt"

    for img_name in sublist:
        food_file_path = f"food_mask/{idx}-{'_'.join(sublist)}/{''.join(img_name)}.txt"
        # 1. Load model and get current width
        obj_path = f"food_mask/{idx}/{''.join(img_name)}.obj"

        plate_numbers = read_file(plate_file_path)
        food_numbers = read_file(food_file_path)

        plate_a = plate_numbers[2] - plate_numbers[0]
        plate_b = plate_numbers[3] - plate_numbers[1]

        plate_center = (plate_numbers[1] + plate_numbers[3]) / 2

        plate_max = max(plate_b, plate_a)
        plate_min = min(plate_b, plate_a)

        proportion = plate_min / plate_max

        food_a = food_numbers[2] - food_numbers[0]
        food_b = food_numbers[3] - food_numbers[1]

        food_center = (food_numbers[1] + food_numbers[3]) / 2

        scaled_width = food_center - plate_center
        scaled_width_p = scaled_width / (plate_min / 2)

        max_length = math.sqrt(food_a ** 2 + food_b ** 2)

        real_proportion = real_plate / plate_max
        real_length = max_length * real_proportion

        # print("Real length", real_length)

        mesh = trimesh.load(obj_path, force='mesh', process=False)

        # Calculate the convex hull of the model
        vertices = mesh.vertices
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]

        # Use rotating calipers algorithm to find the two farthest points on the convex hull
        max_distance, p1, p2 = rotating_calipers(hull_vertices)

        # Create a line segment representing the longest edge
        line = trimesh.load_path([p1, p2])

        # 2. Set target width and calculate scaling factor
        target_max_length = real_length  # Target longest edge length (centimeters)

        scaled_p = (1 - scaled_width_p * proportion * r)
        scale_factor = target_max_length / max_distance * scaled_p * 10

        # 3. Scale the model
        scaled_mesh = mesh.copy()
        scaled_mesh.apply_scale(scale_factor)

        # Scale the line segment
        scaled_line = line.copy()
        scaled_line.apply_scale(scale_factor)

        # 4. Calculate the scaled volume (unit: milliliters, 1 cm³ = 1 mL)
        scaled_volume_ml = scaled_mesh.volume

        # print(f"Original longest edge length: {max_distance:.2f} cm")
        # print(img_name)
        # print(f"Scaled longest edge length: {math.sqrt(((scaled_line.vertices[0][0] - scaled_line.vertices[1][0]) ** 2) +((scaled_line.vertices[0][1] - scaled_line.vertices[1][1]) ** 2) +((scaled_line.vertices[0][2] - scaled_line.vertices[1][2]) ** 2))/10:.2f} cm")
        # print(f"Scaled volume: {scaled_volume_ml * 1e-3:.2f} mL")
        # print(f"{scaled_volume_ml * 1e-3:.2f}")

        # Create a scene and add the model and longest edge line segment
        scene = trimesh.Scene()
        scene.add_geometry(scaled_mesh)
        scene.add_geometry(scaled_line)

        # Display the scene
        # scene.show()

        # Save the scaled model
        if k == 15 and flag:
            flag = False
            continue
        k += 1
        save_path = f"food_mask/{idx}/{''.join(img_name)}_real.obj"
        scaled_mesh.export(save_path)
        print(f"The scaled model has been saved to {save_path}")
