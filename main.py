import cv2
import numpy as np
import os
from segment import segment_object
from utils import incremental_save, show_image

# Mouse interaction state
drawing = False
ix, iy = -1, -1
mask = None

def draw_mask(event, x, y, flags, param):
    """Draws the user-defined mask for object selection."""
    global drawing, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), 20, 255, -1)  # Draw filled circle on mask
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), 10, 255, -1)

# Load the image
image = cv2.imread("img/cat1.jpeg")
if image is None:
    raise FileNotFoundError("Image not found.")
image = cv2.resize(image, (0, 0), fx=5, fy=5)
clone = image.copy()

# Create empty mask (same size as image, 1 channel)
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Interactive mask drawing
cv2.namedWindow("Draw Mask (ESC to Confirm)")
cv2.setMouseCallback("Draw Mask (ESC to Confirm)", draw_mask)

while True:
    temp = image.copy()
    temp[mask > 0] = (0, 0, 255)  # Red overlay for visual feedback
    cv2.imshow("Draw Mask (ESC to Confirm)", temp)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to confirm
        break

cv2.destroyAllWindows()

# Refine mask using segment_object
refined_mask, *_ = segment_object(clone, mask)

# Convert to OpenCV format
refined_mask = (refined_mask > 0).astype(np.uint8) * 255

# Erase using OpenCV inpainting
output = cv2.inpaint(clone, refined_mask, 3, cv2.INPAINT_TELEA)

# Save and show
filename = incremental_save("images", "output", ".jpg", output)
show_image(output)
print(f"Saved as {filename}")
