import torch
import numpy as np

def iou(y_pred, y_true, threshold=0.5):
    assert len(y_pred.shape) == len(y_true.shape), "Input tensor shapes should be same."
    mask_pred = threshold < y_pred
    mask_true = threshold < y_true
    intersection = torch.sum(mask_pred * mask_true, dim=[-2, -1])
    union = torch.sum(mask_pred + mask_true, dim=[-2, -1])
    r = intersection.float() / union.float()
    r[union == 0] = 1
    return r.mean()

def iou_np(y_pred, y_true, threshold=0.5):
    assert len(y_pred.shape) == len(y_true.shape), "Input tensor shapes should be same."
    if len(y_pred.shape) == 2:
        y_pred = np.array([y_pred], dtype=np.uint8)
        y_true = np.array([y_true], dtype=np.uint8)
    mask_pred = threshold < y_pred
    mask_true = threshold < y_true
    intersection = np.sum(mask_pred * mask_true, axis=(-2, -1))
    union = np.sum(mask_pred + mask_true, axis=(-2, -1))
    r = intersection.astype(np.float32) / union.astype(np.float32)
    r[union == 0] = 1
    return r.mean()

def calculate_tf_fm(pred, gt, threshold):
    assert len(pred.shape) == len(gt.shape)
    mask_pred = threshold < pred
    mask_gt = threshold < gt
    tf = torch.sum(mask_pred * mask_gt)
    fm = torch.sum(mask_pred + mask_gt)
    return tf, fm


def calculate_tp_pn_gn_accuracy(pred, gt, threshold):
    assert len(pred.shape) == len(gt.shape)
    m, n = pred.shape[2:]
    mask_pred = threshold < pred
    mask_gt = threshold < gt
    tp = torch.sum(mask_pred * mask_gt)
    # fm = torch.sum(mask_pred + mask_gt)
    pn = torch.sum(mask_pred)
    gn = torch.sum(mask_gt)

    pred_mask = mask_pred[:, 0, ...] + mask_pred[:, 1, ...]
    gt_mask = mask_gt[:, 0, ...] + mask_gt[:, 1, ...]
    accuracy = torch.sum(pred_mask==gt_mask) / (m * n)

    return tp, pn, gn, accuracy


def calculate_tp_pn_gn_accuracyV2(pred, gt, threshold):
    assert len(pred.shape) == len(gt.shape)
    m, n = pred.shape[2:]
    mask_pred = threshold < pred
    mask_gt = threshold < gt
    tp = torch.sum(mask_pred * mask_gt)
    # fm = torch.sum(mask_pred + mask_gt)
    pn = torch.sum(mask_pred)
    gn = torch.sum(mask_gt)

    accuracy = torch.sum(mask_pred==mask_gt) / (m * n)
    return tp, pn, gn, accuracy


def calculate_tp_pn_gn_accuracy_both(pred, gt, threshold):
    assert len(pred.shape) == len(gt.shape)
    m, n = pred.shape[2:]
    mask_pred = threshold < pred
    mask_gt = threshold < gt
    tp = torch.sum(mask_pred * mask_gt)
    # fm = torch.sum(mask_pred + mask_gt)
    pn = torch.sum(mask_pred)
    gn = torch.sum(mask_gt)

    accuracy = torch.sum(mask_pred==mask_gt) / (m * n)
    return tp, pn, gn, accuracy


def calculate_tf_fm_with_numpy(pred, gt, threshold):
    assert len(pred.shape) == len(gt.shape)
    mask_pred = threshold < pred
    mask_gt = threshold < gt
    tf = np.sum(mask_pred * mask_gt)
    fm = np.sum(mask_pred + mask_gt)
    return tf, fm

def calculate_error(mask, threshold):
    assert len(mask.shape) == 4
    mask = (mask > threshold)
    m, n = mask.shape[2:]
    mask = mask[:, 0,...] + mask[:, 1,...]
    error = torch.sum(mask, (1,2)) / (m * n)
    return error


def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    """

    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param smooth:
    @param eps:
    @return: (N, C)
    """
    numerator = 2 * torch.sum(y_true * y_pred, dim=(-1, -2))
    denominator = torch.sum(y_true, dim=(-1, -2)) + torch.sum(y_pred, dim=(-1, -2))
    return 1 - (numerator + smooth) / (denominator + smooth + eps)


def main():
    # import kornia
    # spatial_gradient_function = kornia.filters.SpatialGradient()
    #
    # image = torch.zeros((7, 7))
    # image[2:5, 2:5] = 1
    # print(image)
    #
    # grads = spatial_gradient_function(image[None, None, ...])[0, 0, ...] / 4
    # print(grads[0])
    # print(grads[1])
    #torch.tensor
    y_true = np.array([[[
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ]]])
    y_pred = np.array([[[
        [1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ]]])
    print(y_true.shape)
    print(y_pred.shape)
    r = iou_np(y_pred, y_true, threshold=0.5)
    print(r)


if __name__ == "__main__":
    # import numpy as np
    # a = torch.from_numpy(np.arange(2*2*3*4).reshape((2,2,3,4)))
    # res = calculate_error(a, 1)
    # print(res)
    main()

