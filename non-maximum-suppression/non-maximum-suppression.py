def nms(boxes, scores, iou_threshold):
    def compute_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        intersection = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection

        if union == 0:
            return 0.0
        return intersection / union

    if not boxes:
        return []

    # Sort indices by score descending
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)

    keep = []
    while order:
        current = order[0]
        keep.append(current)
        remaining = []
        for idx in order[1:]:
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold:
                remaining.append(idx)
        order = remaining

    return keep