import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size

    i = np.arange(feature_size)
    j = np.arange(feature_size)
    cy = (i + 0.5) * stride  # shape (F,)
    cx = (j + 0.5) * stride  # shape (F,)

    s = np.array(scales, dtype=float)
    r = np.array(aspect_ratios, dtype=float)
    w = (s[:, None] * np.sqrt(r)[None, :]).ravel()  # shape (S*R,)
    h = (s[:, None] / np.sqrt(r)[None, :]).ravel()  # shape (S*R,)

    # meshgrid for all (i,j) combos, then flatten
    grid_cy, grid_cx = np.meshgrid(cy, cx, indexing='ij')  # both (F, F)
    grid_cy = grid_cy.ravel()  # (F*F,)
    grid_cx = grid_cx.ravel()  # (F*F,)

    # expand dims for broadcasting: (F*F, 1) vs (1, S*R)
    grid_cy = grid_cy[:, None]  # (F*F, 1)
    grid_cx = grid_cx[:, None]  # (F*F, 1)
    w = w[None, :]              # (1, S*R)
    h = h[None, :]              # (1, S*R)

    x1 = (grid_cx - w / 2).ravel()
    y1 = (grid_cy - h / 2).ravel()
    x2 = (grid_cx + w / 2).ravel()
    y2 = (grid_cy + h / 2).ravel()

    return np.stack([x1, y1, x2, y2], axis=-1).tolist()

# Test 1 — Single cell, single anchor
generate_anchors(1, 8, [4], [1.0])


# Test 2 — 2x2 grid, single scale/ratio
generate_anchors(2, 8, [2], [1.0])
