from .data_processing_base import *
from .utils import mask2mask
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image


class ToLabel(object):

    def __call__(self, label):
        label = np.asarray(label, dtype=np.float32) / 255
        return torch.from_numpy(label).float().unsqueeze(0)


class ToLabel2(object):

    def __call__(self, label):
        label = np.asarray(label, dtype=np.float32)
        return torch.from_numpy(label).float().unsqueeze(0)


class AQSDataset(Dataset):
    '''EC: error check'''
    def __init__(self, txt_file, mode, use_reality_data=False,
                 normal_imagenet=True, compared_mode=False):
        super(AQSDataset, self).__init__()
        self.txt_file = txt_file
        self.mode = mode
        self.ids = []
        self.read_files(self.txt_file)
        self.use_reality_data = use_reality_data
        self.compared_mode = compared_mode

        if self.mode == 'debug':
            self.ids = self.ids[:64]
        elif self.mode == 'val':
            np.random.seed(666)
            np.random.shuffle(self.ids)
            np.random.seed(None)
        else:
            np.random.seed(666)
            np.random.shuffle(self.ids)
            np.random.seed(None)

        if normal_imagenet:
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            self.mean, self.std = (0, 0, 0), (1, 1, 1)

        self.label_transform = ToLabel()
        self.label_transform2 = ToLabel2()

    def normalize(self,im):
        mean, std = self.mean, self.std
        mean = np.array(mean).reshape((-1, 1, 1))
        std = np.array(std).reshape((-1, 1, 1))
        im = np.transpose(im, (2, 0, 1)).astype(np.float32)
        im = (im / 255 - mean) / std
        im = torch.from_numpy(im).float()
        return im

    def read_files(self, txt_file):
        with open(txt_file) as f:
            lines = f.readlines()
        for line in lines:
            s = line.strip()
            if s:
                self.ids.append(s)

    def prepare_segmentation(self, img_path, label_path):
        img = np.asarray(Image.open(img_path))
        label = np.asarray(Image.open(label_path))
        if self.mode == 'train':
            if random.random() > 0.75:
                img, label = random_rot_flip_images(img, label, is_rot=False)

            if random.random() > 0.75:
                img[:, :, :3] = RGBtoHSVTransform(img[:, :, :3])

        if self.mode == 'train':
            gt_check, res_gt, res_err = mask2mask(mask=label, vanish_r=0.1, move_r=0.1, appear_r=0.1, res=True)

        else:
            np.random.seed(123)
            gt_check, res_gt, res_err = mask2mask(mask=label, vanish_r=0.1, move_r=0.1, appear_r=0.1, res=True)
            np.random.seed(None)
        '''
        gt_mask: 真值mask
        res_gt: 漏的
        res_err: 错误的
        '''
        img = self.normalize(img)
        gt_mask = self.label_transform(label)
        gt_check = self.label_transform(gt_check)
        res_gt = self.label_transform(res_gt)
        res_err = self.label_transform(res_err)
        res = torch.cat((res_gt, res_err), dim=0)
        # img_check = torch.cat((img, gt_check), dim=0)
        return img, gt_check, gt_mask, res

    def prepare_segmentation_test(self, img_path, unchecked_label_path):
        img = np.asarray(Image.open(img_path))
        img = self.normalize(img)

        unchecked_label = np.asarray(Image.open(unchecked_label_path))
        unchecked_label = self.label_transform(unchecked_label)
        return img, unchecked_label

    def prepare_segmentation_both(self, img_path, checked_label_path, uncheck_label_path):
        img = np.asarray(Image.open(img_path))
        label_checked = np.asarray(Image.open(checked_label_path))
        label_uncheck = np.asarray(Image.open(uncheck_label_path))
        # img, label = self.resize(img, label)
        if self.mode == 'train':
            if random.random() > 0.75:
                img, label_checked, label_uncheck = random_rot_flip_images_both(img, label_checked,
                                                                                label_uncheck, is_rot=True)
            if random.random() > 0.75:
                img = RGBtoHSVTransform(img)
        res_gt = label_checked - label_uncheck
        res_err = label_uncheck - label_checked
        '''
        gt_mask: 真值mask
        res_gt: 漏的
        res_err: 错误的
        '''
        img = self.normalize(img)
        gt_mask = self.label_transform(label_checked)
        uncheck_mask = self.label_transform(label_uncheck)
        res_gt = self.label_transform(res_gt)
        res_err = self.label_transform(res_err)
        res = torch.cat((res_gt, res_err), dim=0)
        return img, uncheck_mask, gt_mask, res


    def __getitem__(self, index):
        pth = self.ids[index]
        list_dir = pth.split()
        if self.mode == 'test':
            return self.prepare_segmentation_test(*list_dir)
        else:
            if self.use_reality_data:
                return self.prepare_segmentation_both(*list_dir)
            else:
                return self.prepare_segmentation(*list_dir[:2])

    def __len__(self):
        return len(self.ids)