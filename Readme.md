# Jitter Aniso Centers

This program is a simple generative art tool that warps one image using another as a bias map. It uses OpenCV for the image processing and PyQt5 for the interface. You can load a source image and a bias image, then the program finds bright spots in the bias image and uses those as centers for jitter-based distortion.

You can adjust or reload the images, automatically detect new centers, or add and delete them manually. Each center influences how the source image bends and shifts, creating unpredictable organic patterns. It is not a precise scientific tool. It’s more like a playground for watching structure fall apart into movement.

### How to Run

1. Install Python 3.10+
2. Create and activate a virtual environment (optional)
3. Install dependencies:

   ```bash
   pip install PyQt5 opencv-python numpy
   ```
4. Run the script:

   ```bash
   python jitter_aniso_centers_safe.py
   ```

### Controls

* **Load Images:** Choose a bias and source image
* **Auto Detect:** Find intensity peaks in the bias map
* **Add / Delete Centers:** Manually control where warping happens
* **Reload:** Replace the images and reset centers

### Purpose

This isn’t meant to be efficient or realistic. It’s for people who like watching an image breathe or dissolve according to another image’s brightness. It makes textures that look a little broken, sometimes like rippling reflections or heat distortion. It’s for experimenting, not for finishing.
