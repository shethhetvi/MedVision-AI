import cv2
import numpy as np
import tensorflow as tf

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    """
    # Load the original image
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert the colored heatmap to RGB
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Resize the heatmap to be the same size as the original image
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    return cam_path
