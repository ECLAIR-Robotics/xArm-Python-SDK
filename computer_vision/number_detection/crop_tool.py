import cv2
import numpy as np
import threading
import ctypes

_initialized = False

def init():
    global lib
    # Define the argument and return types of the C function
    try:
        lib = ctypes.CDLL('./library/number-detection-pkg.dylib')
        lib.prefix_min.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.prefix_min.restype = None
        _initialized = True
    except OSError as e:
        print(f"Failed to load the shared library: {e}")
        lib = None

if not _initialized:
    init()
    _initialized = True


def prefix_min(arr, results, axis=0):
    if lib is None:
        raise RuntimeError("Shared library not loaded. Cannot call prefix_min.")

    arr_ctypes = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    prefix_min_arr = np.empty_like(arr,dtype=np.uint8)
    prefix_min_ctypes = prefix_min_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    #call the C function
    
    lib.prefix_min(arr_ctypes,prefix_min_ctypes,arr.shape[0],arr.shape[1],axis)

    results[axis] = prefix_min_arr

def get_mask(results:dict):
    prefix_min_tb = results[0]
    prefix_min_lr = results[1]
    prefix_min_bt = results[2]
    prefix_min_rl = results[3]

    lowerThresh = 0
    upperThresh = 25

    combined_mask = ((prefix_min_tb >= lowerThresh) & (prefix_min_tb <= upperThresh)) & \
            (prefix_min_lr >= lowerThresh) & (prefix_min_lr <= upperThresh) & \
            (prefix_min_rl >= lowerThresh) & (prefix_min_rl <= upperThresh) & \
            (prefix_min_bt >= lowerThresh) & (prefix_min_bt <= upperThresh) 

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=2)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=2)

    return combined_mask


#NOTE: this was code take from https://github.com/wjbmattingly/ocr_python_textbook/blob/main/02_02_working%20with%20opencv.ipynb

def get_skew_angle(contour):

    min_area_rect = cv2.minAreaRect(contour)
    angle = min_area_rect[-1]
    (h,w) = min_area_rect[1]

    # Adjust the angle based on the w and height
    if w < h:
        tmp = h
        h = w
        w = tmp

    
    if w > h:
        angle = 90 - angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Round the angle to the nearest hundredth degree
    angle = round(angle, 2)
    
    return angle

# Rotate the image around its center
def rotate_image(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def crop_image(img):
    # Call the function and display the result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = {}
    #NOTE: changed to 4 to calculate all scan direction 
    NUM_THREADS = 4
    threads = [None] * NUM_THREADS

    # create and start threads
    for i in range(0,NUM_THREADS):
        threads[i] = threading.Thread(target=prefix_min, args=(gray,results,i))
        threads[i].start()

    # stop and join them back
    for i in range(0,NUM_THREADS):
        threads[i].join()


    #NOTE: try changing the mask to include right->left instead of 
    # left->right, will it change the crop bias????

    #NOTE: Doesn't matter.....

    combined_mask = get_mask(results=results)


    max_area = 0
    largest_bbox = None
    largest_contour = None


    #NOTE: add another heuristic that eliminates distinctly small rectangles (get area of pipette when detected)


    # countour selection heuristics
    lwr_bound = 3.5
    high_bound = 4.4

    # area selection
    area_bound = 1000

    copy_img  = img.copy()
    
    contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w*h

        ratio = w/h

        if (ratio >= lwr_bound and ratio <= high_bound and area > max_area and area > area_bound):
            max_area = area
            largest_bbox = (x,y,w,h)
            largest_contour = contour

    if largest_bbox is not None:
        x,y,w,h = largest_bbox

        # print(f"RATION w/h: {w/h}")

        skew_angle = get_skew_angle(largest_contour)
        # cv2.drawContours(img,largest_contour,-1,(0,255,0),2)

        # print(f"SKEW: {skew_angle} \t CORRECTION: {skew_angle*-1.0}")
        
        if skew_angle !=  0.0:
            rotated_img = rotate_image(img,-1.0*skew_angle)
        else:
            rotated_img = img
        cv2.rectangle(copy_img,(x,y),(x+w,y+h),(255,0,0),2)
        # NOTE: applied heursitic to right and bottom sides
        heuristic = 5
        
        cropped_img = rotated_img[y+heuristic:y+h-heuristic, x+heuristic:x+w-heuristic]

    else:
        cropped_img = img

    # cv2.imshow('bounding box',copy_img)

    return cropped_img, copy_img