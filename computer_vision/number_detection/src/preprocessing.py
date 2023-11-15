import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    crop_img = image[350:600, 200:450]
    cv2.imwrite('./datasets/test/cropped.png', crop_img)

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)
    cv2.imwrite('./datasets/test/gray.png', thresh1)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    return edged

def get_contours(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

    return None

def save_digits(warped, displayCnt):
    vert_padding = 7
    hz_padding = 5
    num_rows, num_cols = warped.shape
    thresh = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('thresh.png', thresh)

    matrix = np.asarray(thresh)
    row = int(num_rows / 2)
    block_tuples, blobs, l_cnt, r_cnt, flag = [], [], 0, 0, False

    for c in range(matrix.shape[1]):
        if (matrix[row, c] == 0):
            if not flag:
                l_cnt = r_cnt
                flag = not flag
        else:
            if flag:
                block_tuples.append(l_cnt)
                blobs.append(r_cnt - l_cnt)
                flag = not flag
        r_cnt = r_cnt + 1

    top_three = sorted(zip(blobs, block_tuples), reverse=True)[:3]

    for i in range(2):
        len, idx = top_three[i]
        post_len, post_idx = top_three[i + 1]
        if i == 1:
            cv2.imwrite('dig_{index}.png'.format(index=i + 1),
                        warped[vert_padding:num_rows - vert_padding, idx + len - hz_padding:post_idx - 10 + hz_padding])
        else:
            cv2.imwrite('dig_{index}.png'.format(index=i + 1),
                        warped[vert_padding:num_rows - vert_padding, idx + len - hz_padding:post_idx + hz_padding])

    idx_3, len_3 = top_three[2]
    cv2.imwrite('dig_3.png', warped[vert_padding:num_rows - vert_padding, idx_3 + len_3 - hz_padding:num_cols + hz_padding])

def main():
    image_path = "sample2.jpeg"
    edged = preprocess_image(image_path)
    displayCnt = get_contours(edged)

    if displayCnt is not None:
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        save_digits(warped, displayCnt)

if __name__ == "__main__":
    main()