import cv2
import os
import numpy as np

# where new image is saved at
save_dir = 'test/test_contrast'
save_path = os.path.join(save_dir, '10.5.jpg')

# access file
file_dir = "test/cropped"
filename = os.path.join(file_dir, "07.5.jpg")
image = cv2.imread(filename)

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#127 128 is the middle values

def contrast(grayImage):
    matrix = np.array(grayImage)
    q3 = np.quantile(matrix, 0.75) # 75% quartile
    q4 = np.quantile(matrix, 0.90)

    # Things that you should change bright factor, dark factor, and split to change how you want to contrast it
    brightFactor = 5
    darkFactor = 0.001
    split = q3 + (q4 - q3)
    rows = grayImage.shape[0]
    cols = grayImage.shape[1]

    if grayImage.dtype != np.uint16:
        grayImage = grayImage.astype(np.uint16)

    for i in range(rows):
        for j in range(cols):
            if (grayImage[i, j] <= split): # darker
                grayImage[i, j] = max(int(grayImage[i, j] * darkFactor), 0)
            elif (grayImage[i, j] >= split + 1): # brighter
                grayImage[i, j] = min(int(grayImage[i, j] * brightFactor), 255)
    
    grayImage = grayImage.astype(np.uint8)
    return grayImage

    
highContrast = contrast(grayImage)

cv2.namedWindow('contrast', cv2.WINDOW_NORMAL)
cv2.imshow('contrast', highContrast)

# matrix = np.array(grayImage)

# q1 = np.quantile(matrix, 0.25)
# q3 = np.quantile(matrix, 0.75)
# iqr = q3 - q1
# q4 = np.quantile(matrix, 0.93)

# print(q1)
# print(q3)
# print(np.median(matrix))
# print(q4)

cv2.waitKey(0)
cv2.destroyAllWindows()