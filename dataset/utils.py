import cv2 as cv
import numpy as np
import os.path as osp
import os


def mask2polygon(mask):
    contours, _ = cv.findContours((mask == 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        contour = close_contour(c.squeeze(1))
        polys.append(contour)
    return polys


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def shift_point(poly, m, n, edge=10):
    if np.max(poly[:, 0]) < n - edge and np.min(poly[:, 0]) > edge:
        sign_x = -1 if np.random.randint(2) == 1 else 1
        x = np.random.randint(edge) * sign_x
    else:
        x = 0

    if np.max(poly[:, 1]) < m - edge and np.min(poly[:, 1]) > edge:
        sign_y = -1 if np.random.randint(2) == 1 else 1
        y = np.random.randint(edge) * sign_y
    else:
        y = 0

    poly[:, 0] += x
    poly[:, 1] += y
    poly[:, 0] = np.clip(poly[:, 0], 0, n)
    poly[:, 1] = np.clip(poly[:, 1], 0, m)
    return poly


def draw_poly(mask, poly, value=1):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    mask = cv.fillPoly(mask.copy(), [poly.astype(np.int32)], value)
    return mask


def draw_polys(mask, polys, value=1):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    poly = polys[0]
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    mask = cv.fillPoly(mask.copy(), [poly.astype(np.int32)], value)
    if len(polys) > 1:
        for i in range(1, len(polys)):
            poly = polys[i]
            if not isinstance(poly, np.ndarray):
                poly = np.array(poly)
            mask = cv.fillPoly(mask, [poly.astype(np.int32)], 0)
    return mask

def appear_polygon(poly, m, n):
    max_x, min_x = np.max(poly[:, 0]), np.min(poly[:, 0])
    max_y, min_y = np.max(poly[:, 1]), np.min(poly[:, 1])
    poly[:, 0] -= min_x
    poly[:, 1] -= min_y
    x_rand = np.random.randint((n-max_x))
    y_rand = np.random.randint((m-max_y))
    poly[:, 0] += x_rand
    poly[:, 1] += y_rand
    poly[:, 0] = np.clip(poly[:, 0], 0, n)
    poly[:, 1] = np.clip(poly[:, 1], 0, m)
    return poly


def poly2box(poly):
    """
    :param poly: Numpy, (N, 2)
    :return: x-min, x-max, y-min, y-max
    """
    x_min, x_max = min(poly[:, 0]), max(poly[:, 0])
    y_min, y_max = min(poly[:, 1]), max(poly[:, 1])
    return (x_min, x_max, y_min, y_max)


def is_hollow(coor0, coorn):
    """
    :param coor0: x-min, x-max, y-min, y-max (large)
    :param coorn: x-min, x-max, y-min, y-max
    :return: True/False
    """
    return (coorn[0]>coor0[0] and coorn[1]<coor0[1]) and (coorn[2]>coor0[2] and coorn[3]<coor0[3])


def shift_point_multi_polys(list_poly, m, n, edge=10):
    if np.max(list_poly[0][:, 0]) < n - edge and np.min(list_poly[0][:, 0]) > edge:
        sign_x = -1 if np.random.randint(2) == 1 else 1
        x = np.random.randint(edge) * sign_x
    else:
        x = 0

    if np.max(list_poly[0][:, 1]) < m - edge and np.min(list_poly[0][:, 1]) > edge:
        sign_y = -1 if np.random.randint(2) == 1 else 1
        y = np.random.randint(edge) * sign_y
    else:
        y = 0
    for i in range(len(list_poly)):
        list_poly[i][:, 0] += x
        list_poly[i][:, 1] += y
        list_poly[i][:, 0] = np.clip(list_poly[i][:, 0], 0, n)
        list_poly[i][:, 1] = np.clip(list_poly[i][:, 1], 0, m)
    return list_poly


def appear_polygons(polys, m, n):
    poly = polys[0]
    max_x, min_x = np.max(poly[:, 0]), np.min(poly[:, 0])
    max_y, min_y = np.max(poly[:, 1]), np.min(poly[:, 1])
    x_rand = np.random.randint((n-max_x))
    y_rand = np.random.randint((m-max_y))
    for i in range(len(polys)):
        polys[i][:, 0] -= min_x
        polys[i][:, 1] -= min_y
        polys[i][:, 0] += x_rand
        polys[i][:, 1] += y_rand
        polys[i][:, 0] = np.clip(polys[i][:, 0], 0, n)
        polys[i][:, 1] = np.clip(polys[i][:, 1], 0, m)
    return polys


def mask2mask(mask, vanish_r=0.15, move_r=0.15, appear_r=0.15, res=False, vis=False):
    '''丢失、错误（移位、完全错误（无中生有））'''
    contours, _ = cv.findContours((mask == 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    m, n = mask.shape
    if vis:
        list_contours = []
        old_contours = []
    for c in contours:
        random_value = np.random.random()
        contour = close_contour(c.squeeze(1))
        if vis: old_contours.append(contour.copy())
        if random_value < vanish_r:
            pass
        elif random_value < vanish_r+move_r:
            contour = shift_point(contour, m, n, edge=10)
            new_mask = draw_poly(new_mask, poly=contour, value=255)
            if vis: list_contours.append(contour)
        elif random_value < vanish_r + move_r+appear_r:
            if vis: list_contours.append(contour)
            new_mask = draw_poly(new_mask, poly=contour, value=255)
            contour = appear_polygon(contour, m, n)
            if vis: list_contours.append(contour)
            new_mask = draw_poly(new_mask, poly=contour, value=255)
        else:
            new_mask = draw_poly(new_mask, poly=contour, value=255)
            if vis: list_contours.append(contour)
    if res:
        mask_res = mask - new_mask
        mask_err = new_mask - mask
        if vis:
            res_contours = mask2polygon(mask_res)
            err_contours = mask2polygon(mask_err)
            return new_mask, old_contours, list_contours, res_contours, err_contours, (new_mask, mask_res, mask_err)

        return new_mask, mask_res, mask_err

    return new_mask


def mask2mask_waterbody(mask, vanish_r=0.15, move_r=0.15, appear_r=0.15, res=False, vis=False):
    '''丢失、错误（移位、完全错误（无中生有））'''
    contours, _ = cv.findContours((mask == 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    m, n = mask.shape
    list_contours = []
    old_contours = []
    tmp_poly = []
    ratio = 0.2 #0.95
    vanish_area_thresh = 2000 #20000
    for c in contours:
        area = cv.contourArea(c)
        random_value = np.random.random()
        contour = close_contour(c.squeeze(1))
        coor = poly2box(contour)

        if len(tmp_poly) != 0 and not is_hollow(tmp_poly[0][0], coor):
            cur_polys = [x[1] for x in tmp_poly]
            if vis: old_contours.extend(cur_polys)
            if tmp_poly[0][2] > m * n * ratio:  # 0.05, 0.2
                if random_value < move_r*2:
                    cur_polys = shift_point_multi_polys(cur_polys, m, n, edge=30)
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                    if vis: list_contours.extend(cur_polys)
                else:
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                    if vis: list_contours.extend(cur_polys)
            else:
                if random_value < vanish_r:
                    if tmp_poly[0][2] < vanish_area_thresh:
                        pass
                    else:
                        new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                        if vis: list_contours.extend(cur_polys)

                elif random_value < vanish_r + move_r:
                    cur_polys = shift_point_multi_polys(cur_polys, m, n, edge=10)
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                    if vis: list_contours.extend(cur_polys)

                elif random_value < vanish_r + move_r + appear_r:
                    if vis: list_contours.extend(cur_polys)
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)

                    if tmp_poly[0][2] < vanish_area_thresh:
                        cur_polys = appear_polygons(cur_polys, m, n)
                        if vis: list_contours.extend(cur_polys)
                        new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                else:
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                    if vis: list_contours.extend(cur_polys)
            tmp_poly = []
        tmp_poly.append((coor, contour, area))

    if len(tmp_poly) != 0:
        random_value = np.random.random()
        cur_polys = [x[1] for x in tmp_poly]
        if vis: old_contours.extend(cur_polys)
        if tmp_poly[0][2] > m * n * ratio:
            if random_value < move_r:
                cur_polys = shift_point_multi_polys(cur_polys, m, n, edge=10)
                new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                if vis: list_contours.extend(cur_polys)
            else:
                new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                if vis: list_contours.extend(cur_polys)
        else:
            if random_value < vanish_r:
                if tmp_poly[0][2] < vanish_area_thresh:
                    pass
                else:
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                    if vis: list_contours.extend(cur_polys)

            elif random_value < vanish_r + move_r:
                cur_polys = shift_point_multi_polys(cur_polys, m, n, edge=10)
                new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                if vis: list_contours.extend(cur_polys)

            elif random_value < vanish_r + move_r + appear_r:
                if vis: list_contours.extend(cur_polys)
                new_mask = draw_polys(new_mask, polys=cur_polys, value=255)

                if tmp_poly[0][2] < vanish_area_thresh:
                    cur_polys = appear_polygons(cur_polys, m, n)
                    if vis: list_contours.extend(cur_polys)
                    new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
            else:
                new_mask = draw_polys(new_mask, polys=cur_polys, value=255)
                if vis: list_contours.extend(cur_polys)
        tmp_poly = []
    if res:
        mask_res = mask - new_mask
        mask_err = new_mask - mask
        # if vis:
        #     res_contours = mask2polygon(mask_res)
        #     err_contours = mask2polygon(mask_err)
            # return new_mask, old_contours, list_contours, res_contours, err_contours, (new_mask, mask_res, mask_err)
            # return new_mask, old_contours, list_contours, res_contours, err_contours, (new_mask, mask_res, mask_err)

        return new_mask, mask_res, mask_err
    if vis:
        return new_mask, old_contours, list_contours
    else:
        return new_mask
