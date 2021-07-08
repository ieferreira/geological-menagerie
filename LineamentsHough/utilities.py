import numpy as np
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from PIL import Image
import matplotlib.pyplot as plt


def load_im(fname: str) -> np.array:
    image_file = Image.open(fname)  # open colour image
    image_file = image_file.convert('1')
    img = np.array(image_file)
    return img


def plot_hough(im, out, angles, d):
    fix, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(im, cmap=plt.cm.gray, alpha=0.2)
    axes[0].set_title('Input image', fontsize=22)

    angle_step = 0.5 * np.rad2deg(np.diff(angles).mean())
    d_step = 0.5 * np.diff(d).mean()
    bounds = (np.rad2deg(angles[0]) - angle_step,
              np.rad2deg(angles[-1]) + angle_step,
              d[-1] + d_step, d[0] - d_step)

    axes[1].imshow(out, cmap=plt.cm.bone, extent=bounds, vmax=10)
    axes[1].set_title('Hough transform', fontsize=22)
    axes[1].set_xlabel('Angle (degree)', fontsize=14)
    axes[1].set_ylabel('Distance (pixel)', fontsize=14)

    axes[2].imshow(im, cmap=plt.cm.gray, alpha=0.2)
    for _, angle, dist in zip(*hough_line_peaks(out, angles, d)):
        #print (_, angle, dist)
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        axes[2].axline((x0, y0), slope=np.tan(
            angle + np.pi/2), alpha=1, c='red', lw=0.5)
    axes[2].set_title('Detected lines (red)', fontsize=22)

    plt.suptitle('Straight Line Hough Transform', fontsize=24)
    plt.tight_layout()
    plt.show()
