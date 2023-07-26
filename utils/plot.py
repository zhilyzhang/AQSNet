import os, sys
import os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import skimage.measure

from PIL import Image
from glob import glob
from tqdm import tqdm
plt.rc('font', family='Times New Roman')


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def visualize_list_data(list_data, M, N, list_polys=[], list_titles=[], show_data=False, save_dirs='', main_title=None):
    '''
    list_data 必须先存放image，然后其他栅格label等；list_polys是在image画点画线呈现
    :param list_data:
    :param M:
    :param N:
    :param list_polys:
    :param show_data:
    :param save_dirs:
    :return:
    '''
    height, width = list_data[0].shape[:2]
    times = 6
    if height > width:
        dt = int(height / width + 0.5)
        fig = plt.figure(figsize=(N * times, M * times * dt))
    else:
        dt = int(width / height + 0.5)
        fig = plt.figure(figsize=(N * times * dt, M * times))
    if main_title is not None:
        fig.suptitle(main_title, fontsize=times * 5)
    NUM = M * 10 + N
    cur_num_id = NUM * 10
    id_titles = 0
    for i, im in enumerate(list_data):
        cur_num_id += 1
        # plt.axis('off')
        plt.subplot(cur_num_id)
        if len(list_titles) > 0 :
            plt.title(list_titles[id_titles], fontdict={'weight': 'normal', 'size': times * 5})  # 改变图标题字体
            id_titles += 1
        plt.imshow(im)

    if len(list_polys) > 0:
        for list_poly in list_polys:
            cur_num_id += 1
            plt.subplot(cur_num_id)
            # plt.axis('off')
            if len(list_titles) > 0:
                plt.title(list_titles[id_titles], fontdict={'weight': 'normal', 'size': times * 5})  # 改变图标题字体
                id_titles += 1
            plt.imshow(list_data[0].copy())
            if len(list_poly) == 0: continue
            for poly in list_poly:
                poly = close_contour(poly)
                poly = skimage.measure.approximate_polygon(poly, 1.0)
                plt.plot(poly[:, 0], poly[:, 1], linewidth=2)
                plt.plot(poly[:, 0], poly[:, 1], 'o', linewidth=1)
                # plt.text(np.mean(poly[:, 0]), np.mean(poly[:, 1]), f'PN: {poly.shape[0]}')
                # plt.plot(poly[:3, 0], poly[:3, 1], 'o', linewidth=1, color='b')
                # plt.plot(poly[3:4, 0], poly[3:4, 1], 'o', linewidth=1, color='g')
    if show_data:
        plt.show()
    else:
        plt.savefig(save_dirs, bbox_inches='tight', dpi=100)
    plt.close()


def mask2polygon(mask, min_area=20, value=255):
    contours, _ = cv.findContours((mask == value).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    polys = []

    for c in contours:
        if cv.contourArea(c) < min_area: continue
        contour = close_contour(c.squeeze(1))
        contour = skimage.measure.approximate_polygon(contour, 1.0)
        polys.append(contour)
    return polys


def image_mask_with_rgb(im, mask, rgb=[255, 0, 0]):
    for i in range(3):
        im[:, :, i][mask == 255] = rgb[i] * 0.55 + im[:, :, i][mask == 255] * 0.45
    return im


def vis_image_with_questioned_pixels(res_miss, res_error, label_uncheck, image, save_tif_path, show_data=False, dpi=150):
    ld_rgb = [0, 255, 0]
    err_rgb = [0, 0, 255]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    im_label_res = image_mask_with_rgb(image.copy(), res_miss, rgb=ld_rgb)
    im_label_res = image_mask_with_rgb(im_label_res, res_error, rgb=err_rgb)
    polys_uncheck = mask2polygon(label_uncheck, min_area=10, value=255)
    plt.imshow(im_label_res)
    for poly in polys_uncheck:
        poly = close_contour(poly)
        plt.plot(poly[:, 0], poly[:, 1], linewidth=2, color='r')
        # plt.plot(poly[:, 0], poly[:, 1], 'o', linewidth=1, color='b')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0.008, hspace=0.008)  # 修改子图之间的间隔 512×512

    if show_data:
        plt.show()
    else:
        plt.savefig(save_tif_path, bbox_inches='tight', dpi=dpi)
    plt.close()


