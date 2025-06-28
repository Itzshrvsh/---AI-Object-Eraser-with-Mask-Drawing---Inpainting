# Load model
from segment_anything import sam_model_registry, SamPredictor
import torch

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)
predictor.set_image(img)

# Predict mask from your drawn point (or box)
input_box = np.array([ix, iy, x, y])  # from mouse
masks, scores, logits = predictor.predict(box=input_box[None, :], multimask_output=False)
