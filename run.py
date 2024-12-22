import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from tkinter import filedialog, Tk
import os

def mask_detect(imdb, debug=0):
    # Placeholder for mask detection logic
    mask = np.mean(imdb, axis=2) > 128  # Example logic
    p = None  # Placeholder for additional parameters
    return mask, p

# Hide the main Tkinter window
root = Tk()
root.withdraw()

# Image path
filetypes = [("Image files", "*.tif *.tiff *.jpg *.png *.bmp *.gif")]
DialogTitle = 'Select multiple images to generate mask...'
file_paths = filedialog.askopenfilenames(title=DialogTitle, filetypes=filetypes)

# Number of images
Nim = len(file_paths)

if Nim == 0:
    raise ValueError("No images selected")

# Read a sample image
sample = io.imread(file_paths[0])

# Load the images
imdb = np.zeros((sample.shape[0], sample.shape[1], Nim), dtype=sample.dtype)
good_im = 0

for ii, file_path in enumerate(file_paths):
    print(f'Loading image {ii + 1} of {Nim} ({os.path.basename(file_path)})')
    try:
        img = io.imread(file_path)
        # Convert rgb to gray
        if img.ndim == 3:
            img = color.rgb2gray(img)
        imdb[:, :, ii] = img
        good_im += 1
    except Exception as e:
        print(f'Skipping image {os.path.basename(file_path)} because of {e}')
        continue

if good_im < 2:
    raise ValueError('You must select at least 2 images to run the script')

if good_im < 50:
    print('Warning: The number of images selected might be insufficient for the script to work')

# Generate the mask showing the debug information
mask, p = mask_detect(imdb, debug=1)

# Save the mask
filename = f'mask_{os.path.basename(file_paths[0])}'
io.imsave(filename, mask.astype(np.uint8) * 255)
print(f'The mask file was saved as {filename}')