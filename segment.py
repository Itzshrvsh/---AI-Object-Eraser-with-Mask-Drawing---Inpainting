import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def segment_object(image, mask):
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        return mask

    input_point = np.array([[int(np.mean(x)), int(np.mean(y))]])
    input_label = np.array([1])

    sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b.pth")
    predictor = SamPredictor(sam)   
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    kernel = np.ones((15, 15), np.uint8)
    clean_mask = cv2.morphologyEx(masks[0].astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return (masks[0].astype(np.uint8) * 255) , clean_mask.astype(np.uint8) * 255
