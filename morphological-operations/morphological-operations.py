def morphological_op(image, kernel, operation):
    rows = len(image)
    cols = len(image[0]) if rows > 0 else 0
    krows = len(kernel)
    kcols = len(kernel[0]) if krows > 0 else 0

    pad_r = krows // 2
    pad_c = kcols // 2

    def get_pixel(r, c):
        if 0 <= r < rows and 0 <= c < cols:
            return image[r][c]
        return 0

    output = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if operation == "erode":
                result = 1
                for ki in range(krows):
                    for kj in range(kcols):
                        if kernel[ki][kj] == 1:
                            ri = i + ki - pad_r
                            rj = j + kj - pad_c
                            if get_pixel(ri, rj) == 0:
                                result = 0
                                break
                    if result == 0:
                        break
                output[i][j] = result
            else:  # dilate
                result = 0
                for ki in range(krows):
                    for kj in range(kcols):
                        if kernel[ki][kj] == 1:
                            ri = i + ki - pad_r
                            rj = j + kj - pad_c
                            if get_pixel(ri, rj) == 1:
                                result = 1
                                break
                    if result == 1:
                        break
                output[i][j] = result

    return output