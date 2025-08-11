import os, json, numpy as np
from agx_emulsion.utils.crop_resize import crop_image

def make_image(H=100, W=150):
    img = np.zeros((H, W, 3), dtype=float)
    for y in range(H):
        v = y / (H - 1)
        img[y, :, :] = v
    return img

def main():
    # Ensure C++ wrote the crop JSON
    cpp_path = os.path.abspath('cpp/tests/crop_resize/tmp_crop_cpp.json')
    if not os.path.exists(cpp_path):
        # run the C++ test to generate it
        root = os.path.abspath('.')
        os.system(os.path.join(root, 'cpp/build/independent_crop_resize_test') + ' >/dev/null 2>&1')
    if not os.path.exists(cpp_path):
        raise SystemExit(f'Missing C++ crop JSON: {cpp_path}')

    with open(cpp_path) as f:
        d = json.load(f)
    H, W = int(d['H']), int(d['W'])
    flat = np.array(d['data'], dtype=float)
    cpp = flat.reshape(H, W, 3)

    # Python crop for same parameters
    img = make_image()
    py = crop_image(img, center=(0.5, 0.5), size=(0.3, 0.2))

    if cpp.shape != py.shape:
        print('Shape mismatch:', cpp.shape, py.shape)
    diff = np.abs(cpp - py)
    print('Crop parity: max abs diff =', float(diff.max()))
    print('Mean abs diff =', float(diff.mean()))

if __name__ == '__main__':
    main()


