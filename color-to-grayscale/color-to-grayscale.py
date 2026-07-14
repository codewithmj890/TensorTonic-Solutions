def color_to_grayscale(image):
    output = []
    for row in image:
        gray_row = []
        for pixel in row:
            r, g, b = pixel
            y = 0.299 * r + 0.587 * g + 0.114 * b
            gray_row.append(y)
        output.append(gray_row)
    return output