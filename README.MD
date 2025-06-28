# AI Object Remover

AI Object Remover is a Python-based desktop application that allows users to seamlessly erase objects from images using a combination of manual masking and automatic segmentation. The erased areas are filled with background-matching pixels to create realistic, artifact-free edits. This project integrates traditional OpenCV techniques with Segment Anything Model (SAM) for precise object segmentation and inpainting.

---

## 🚀 Features

* **Interactive Object Selection**: Draw on the image to mark the object to be removed.
* **AI-Assisted Segmentation**: Optional refinement using SAM for accurate segmentation.
* **Realistic Inpainting**: Removed areas are filled using OpenCV's inpainting to match background texture and color.
* **Auto File Saving**: Outputs are saved incrementally without overwriting previous images.
* **Minimal UI**: OpenCV windows provide a simple way to interact with images directly.

---

## 📁 Project Structure

```
RUBAN/
├── main.py               # Main file to run the object remover
├── segment.py            # Handles AI-based segmentation (e.g., SAM)
├── utils.py              # Helper functions: saving, image display, etc.
├── img/
│   └── pic1.jpg          # Input image(s)
├── images/               # Automatically saved output images
└── README.md             # Project documentation
```

---

## 📚 How It Works

1. Load an image from `img/`.
2. Draw over the object you want to remove using the mouse.
3. Press `ESC` to confirm your mask.
4. Optionally, the SAM model refines your mask for cleaner edges.
5. OpenCV inpaints the region, generating a realistic replacement.
6. The result is saved in the `images/` folder.

---

## ⚙️ Requirements

* Python 3.8+
* OpenCV
* NumPy
* SAM or any compatible segmentation model (optional)

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## 🌐 Demo

![Before and After](https://user-images.githubusercontent.com/your-demo-link)

---

## 🚩 Use Cases

* Remove unwanted people/objects from images.
* Erase logos, signs, or text.
* Create training data by masking elements.
* Prototype for advanced photo editing tools.

---

## 📦 Future Features

* Auto image enhancement after inpainting
* Multi-object removal support
* Batch processing mode
* Web-based interface (Flask or Streamlit)

---

## 🌟 Credits

Created by Sharvesh

Inspired by:

* [OpenCV Inpainting](https://docs.opencv.org/)
* [Meta Segment Anything Model](https://github.com/facebookresearch/segment-anything)

---

## ✉️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
