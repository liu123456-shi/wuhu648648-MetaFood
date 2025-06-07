import numpy as np
from collections import defaultdict


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0.0


def calculate_dice(box1, box2):
    """Calculate Dice coefficient (variant of F1 score for segmentation/detection tasks)"""
    iou = calculate_iou(box1, box2)
    return 2 * iou / (iou + 1) if iou > 0 else 0.0


def calculate_giou(box1, box2):
    """Calculate Generalized Intersection over Union (GIoU)"""
    # Calculate coordinates of the smallest enclosing rectangle
    x1_c = min(box1[0], box2[0])
    y1_c = min(box1[1], box2[1])
    x2_c = max(box1[2], box2[2])
    y2_c = max(box1[3], box2[3])

    # Calculate area of the enclosing rectangle
    closure_area = (x2_c - x1_c) * (y2_c - y1_c)

    # Calculate IoU
    iou = calculate_iou(box1, box2)

    # Calculate GIoU
    giou = iou - (closure_area - (box1[2] - box1[0]) * (box1[3] - box1[1]) -
                  (box2[2] - box2[0]) * (box2[3] - box2[1])) / closure_area

    return giou


def calculate_distance_metrics(box1, box2):
    """Calculate distance between box centers and size error"""
    # Calculate center points
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

    # Calculate Euclidean distance
    euclidean_dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    # Calculate size error (width/height ratio)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    size_error = abs(w1 / w2 - 1) + abs(h1 / h2 - 1)

    return euclidean_dist, size_error


def read_box_coordinates(file_path):
    """Read bounding box coordinates from file"""
    try:
        with open(file_path, 'r') as f:
            coords = list(map(float, f.readline().strip().split()))
            return (coords[0], coords[1], coords[2], coords[3])
    except:
        return None


def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Calculate Average Precision (AP) - simplified version"""
    if not pred_boxes or not gt_boxes:
        return 0.0

    # Sort predicted boxes by confidence (assuming all have the same confidence here)
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[1], reverse=True)

    # Mark ground truth boxes as matched or not
    gt_matched = [False] * len(gt_boxes)

    # Calculate TP/FN for each predicted box
    tp = np.zeros(len(pred_boxes_sorted))
    fp = np.zeros(len(pred_boxes_sorted))

    for i, (_, pred_box) in enumerate(pred_boxes_sorted):
        max_iou = -1
        max_idx = -1

        # Find ground truth box with highest IoU with current predicted box
        for j, (_, gt_box) in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        # Determine if it's a true positive
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1

    # Calculate cumulative TP and FP
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)

    # Calculate precision and recall
    precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
    recall = cumsum_tp / len(gt_boxes)

    # Calculate AP (11-point interpolation method)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0

    return ap


food_list = [["bagel", "cream_cheese"], ["breaded_fish", "lemon", "broccoli"], ["burger", "hot_dog"],
             ["cheesecake", "strawberry", "raspberry"], ["energy_bar", "cheddar_cheese", "banana"],
             ["grilled_salmon", "broccoli"], ["pasta", "garlic_bread"],
             ["pbj", "carrot_stick", "apple", "celery"], ["pizza", "chicken_wing"],
             ["quesadilla", "guacamole", "salsa"], ["roast_chicken_leg", "biscuit"], ["sandwich", "cookie"],
             ["steak", "mashed_potatoes"], ["toast", "sausage", "fried_egg"]]

# Collect global metrics
global_metrics = {
    'iou': [], 'dice': [], 'giou': [],
    'euclidean_dist': [], 'size_error': [],
    'ap_50': [], 'ap_75': []
}

for idx, sublist in enumerate(food_list, start=1):
    combo_name = '_'.join(sublist)
    print(f"\n==== Processing food combination {idx}: {combo_name} ====")

    # Store all bounding boxes for current combination
    pred_boxes = []
    gt_boxes = []

    for img_name in sublist:
        # Construct file paths
        pred_path = f"./food_mask/{idx}-{combo_name}/{img_name}.txt"
        gt_path = f"./food_mask_GT/{idx}-{combo_name}/{img_name}.txt"

        # Read coordinates
        box_p = read_box_coordinates(pred_path)
        box_t = read_box_coordinates(gt_path)

        if box_p and box_t:
            pred_boxes.append((img_name, box_p))
            gt_boxes.append((img_name, box_t))

            # Calculate various metrics
            iou = calculate_iou(box_p, box_t)
            dice = calculate_dice(box_p, box_t)
            giou = calculate_giou(box_p, box_t)
            euclidean_dist, size_error = calculate_distance_metrics(box_p, box_t)

            # Save to global statistics
            global_metrics['iou'].append(iou)
            global_metrics['dice'].append(dice)
            global_metrics['giou'].append(giou)
            global_metrics['euclidean_dist'].append(euclidean_dist)
            global_metrics['size_error'].append(size_error)

            # Output detailed metrics
            print(f"✔ {img_name}:")
            print(f"  ├─ IoU: {iou:.4f}")
            print(f"  ├─ Dice: {dice:.4f}")
            print(f"  ├─ GIoU: {giou:.4f}")
            print(f"  ├─ Center Distance: {euclidean_dist:.2f}px")
            print(f"  └─ Size Error: {size_error:.4f}")
        else:
            print(f"✖ Could not read coordinates for {img_name}")

    # Calculate AP for current combination
    if pred_boxes and gt_boxes:
        ap_50 = calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5)
        ap_75 = calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.75)

        global_metrics['ap_50'].append(ap_50)
        global_metrics['ap_75'].append(ap_75)

        print(f"\nCombination-level metrics:")
        print(f"  ├─ AP@50: {ap_50:.4f}")
        print(f"  └─ AP@75: {ap_75:.4f}")

# Output global statistics
print("\n\n==== Global Statistics ====")
if global_metrics['iou']:
    print(f"Average IoU: {np.mean(global_metrics['iou']):.4f}")
    print(f"Average Dice: {np.mean(global_metrics['dice']):.4f}")
    print(f"Average GIoU: {np.mean(global_metrics['giou']):.4f}")
    print(f"Average Center Distance: {np.mean(global_metrics['euclidean_dist']):.2f}px")
    print(f"Average Size Error: {np.mean(global_metrics['size_error']):.4f}")

    if global_metrics['ap_50']:
        print(f"mAP@50: {np.mean(global_metrics['ap_50']):.4f}")
        print(f"mAP@75: {np.mean(global_metrics['ap_75']):.4f}")
else:
    print("Warning: No valid data found for evaluation")
