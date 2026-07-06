def image_histogram(image):
    hist = [0] * 256
    
    # Iterate through every row in the 2D image matrix
    for row in image:
        # Iterate through every pixel value in the current row
        for pixel in row:
            # The pixel value acts as the index; increment its count
            hist[pixel] += 1
            
    return hist