import numpy as np
from agx_emulsion.utils.crop_resize import crop_image
from skimage.transform import resize

def make_image(H=100, W=150):
    img = np.zeros((H, W, 3), dtype=float)
    for y in range(H):
        v = y / (H - 1)
        img[y, :, :] = v
    return img

def main():
    img = make_image()
    # Python crop
    cropped = crop_image(img, center=(0.5, 0.5), size=(0.3, 0.2))
    # Python bilinear resize: older/newer skimage compatibility
    try:
        resized = resize(cropped, (64, 96), order=1, channel_axis=2, anti_aliasing=False, preserve_range=True)
    except TypeError:
        resized = resize(cropped, (64, 96), order=1, anti_aliasing=False, preserve_range=True)
    print('PY crop shape:', cropped.shape)
    print('PY resized shape:', resized.shape)

if __name__ == '__main__':
    main()


