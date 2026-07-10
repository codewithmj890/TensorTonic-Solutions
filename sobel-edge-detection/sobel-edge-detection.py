def sobel_edges(image):
    rows = len(image)
    cols = len(image[0]) if rows > 0 else 0

    # Zero-pad the image with a 1-pixel border
    padded = [[0.0] * (cols + 2) for _ in range(rows + 2)]
    for i in range(rows):
        for j in range(cols):
            padded[i + 1][j + 1] = image[i][j]

    Kx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Ky = [[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]

    result = [[0.0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            gx = 0.0
            gy = 0.0
            for di in range(3):
                for dj in range(3):
                    pixel = padded[i + di][j + dj]
                    gx += Kx[di][dj] * pixel
                    gy += Ky[di][dj] * pixel
            result[i][j] = (gx ** 2 + gy ** 2) ** 0.5

    return result