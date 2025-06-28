import cv2
import numpy as np

def erase_object(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask > 127).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) , inpainted   
