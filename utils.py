import cv2
import os

def incremental_save(folder, base_name, extension, image):
    if not os.path.exists(folder):
        os.makedirs(folder)
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}_{i}{extension}")):
        i += 1
    filename = f"{base_name}_{i}{extension}"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    return filepath

def show_image(image, window_name="Result"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
