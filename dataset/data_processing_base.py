import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def random_rot_flip_images(image, label, is_rot=True):
    if is_rot:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a


def RGBtoHSVTransform(image, hue=.1, sat=1.2, val=1.2):
    '''
        对图像进行HSV变换
        :param
        image: numpy, b,g,r
        :return: 有颜色色差的图像image
        '''
    image = image[:, :, ::-1].copy()
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(image / 255.)  # RGB
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image = hsv_to_rgb(x) * 255  # 0 to 1
    return image.astype(np.uint8)[:, :, ::-1].copy()


def random_rot_flip_images_both(image, label_checked, label_uncheck, is_rot=True):
    if is_rot:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label_checked = np.rot90(label_checked, k)
        label_uncheck = np.rot90(label_uncheck, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label_checked = np.flip(label_checked, axis=axis).copy()
    label_uncheck = np.flip(label_uncheck, axis=axis).copy()
    return image, label_checked, label_uncheck
