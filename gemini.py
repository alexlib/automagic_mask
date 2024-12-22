
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_simulated_ensemble(image, num_frames=5):
    height, width = image.shape
    ensemble = []
    for i in range(num_frames):
        shift_x = np.random.randint(-5, 5)
        shift_y = np.random.randint(-5, 5)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_image = cv2.warpAffine(image, M, (width, height))
        ensemble.append(shifted_image)
    return np.array(ensemble)


def dynamic_masking(image_ensemble):
    inverted_ensemble = 255 - image_ensemble
    min_background = np.min(inverted_ensemble, axis=0)
    subtracted_ensemble = inverted_ensemble - min_background
    
    masks = []
    for frame in subtracted_ensemble:
        median_filtered = cv2.medianBlur(frame, 9)
        closing_kernel = np.ones((5, 5), np.uint8)
        closed_image = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, closing_kernel, iterations=10)
        
        _, thresholded = cv2.threshold(closed_image, 125, 255, cv2.THRESH_BINARY) # Threshold step 1
        thresholded = 255-thresholded # Pixel inversion
        _, thresholded = cv2.threshold(thresholded, 10, 255, cv2.THRESH_BINARY)  # Threshold step 2
        erosion_kernel = np.ones((3, 3), np.uint8)
        eroded_image = cv2.erode(thresholded, erosion_kernel, iterations=2)

        _, mask = cv2.threshold(eroded_image, 10, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    
    masked_images = []
    for i, frame in enumerate(image_ensemble):
        masked_image = cv2.bitwise_and(frame, frame, mask=masks[i])
        masked_images.append(masked_image)
    return masked_images, masks

# Load the image
image = cv2.imread("images/00000203.tif", cv2.IMREAD_GRAYSCALE)
image_ensemble = create_simulated_ensemble(image, num_frames=5)

masked_images, masks = dynamic_masking(image_ensemble)

# Plotting the results
fig, axes = plt.subplots(len(image_ensemble), 2, figsize=(10, 5 * len(image_ensemble)))

for i, (original_frame, masked_frame, mask) in enumerate(zip(image_ensemble, masked_images, masks)):
    axes[i, 0].imshow(original_frame, cmap='gray')
    axes[i, 0].set_title(f'Original Frame {i+1}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(masked_frame, cmap='gray')
    axes[i, 1].set_title(f'Masked Frame {i+1}')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(masks), figsize=(10, 5))
for i, mask in enumerate(masks):
    axes[i].imshow(mask, cmap='gray')
    axes[i].set_title(f'Mask {i+1}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()