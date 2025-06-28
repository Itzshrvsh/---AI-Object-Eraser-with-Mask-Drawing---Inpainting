import cv2
import numpy as np
from segment import segment_object
# Make sure erase.py exists in the same directory as main.py
from erase import erase_object
from display import show_image
import os

def rescale(image, scale_percent):
    """
    Rescale the image to a given percentage of its original size.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Load the image
image = cv2.imread("img/pic1.jpg")
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(image1, (1024, 1024))

overlay = image.copy()
resized = rescale(image2 , 90)
clone = resized.copy()
drawing = False
ix, iy = -1, -1
mask = np.zeros(resized.shape[:2], dtype=np.uint8)

# Mouse callback
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(resized, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(mask, (x, y), 5, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def save_output_image(result, output_folder="images", base_name="output"):
    # Make sure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Find the next available filename
    i = 1
    while True:
        output_path = os.path.join(output_folder, f"{base_name}_{i}.jpg")
        if not os.path.exists(output_path):
            break
        i += 1

    # Save the image
    cv2.imwrite(output_path, result)
    print(f"[✅] Saved: {output_path}")

# Start GUI
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_circle)
while True:
    cv2.imshow("Image", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Optional: Use SAM for better masking
refined_mask = segment_object(clone, mask)

# Erase object
result = erase_object(clone, refined_mask)

# Show result
show_image(result)
#cv2.imwrite("images/output.jpg", result)
save_output_image(result)