import os.path as osp
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import random
import numpy as np


def generate_train_val_txt_with2inputs():
    train_root = '/data02/zzl/dataset/BuildingDatas/building_labeling_check/BLCDataset/train'
    val_root = '/data02/zzl/dataset/BuildingDatas/building_labeling_check/BLCDataset/val'

    save_path = '/data02/zzl/dataset/BuildingDatas/building_labeling_check/txt_files/BLCDataset_with_simulated'
    os.makedirs(save_path, exist_ok=True)

    with open(osp.join(save_path, 'train.txt'), 'w') as f:
        label_pathes = glob(osp.join(train_root, 'label_checked', '*.tif'))
        for pth in tqdm(label_pathes):
            basename = osp.basename(pth)
            im_path = osp.join(train_root, 'image', basename)
            if not os.path.exists(im_path):
                print(f'{basename} dose not exist!')
                continue
            f.write('%s %s\n' % (im_path, pth))

    with open(osp.join(save_path, 'val.txt'), 'w') as f:
        label_pathes = glob(osp.join(val_root, 'label_checked', '*.tif'))
        for pth in tqdm(label_pathes):
            basename = osp.basename(pth)
            im_path = osp.join(val_root, 'image', basename)
            if not os.path.exists(im_path):
                print(f'{basename} dose not exist!')
                continue
            f.write('%s %s\n' % (im_path, pth))


def generate_train_val_txt_with3inputs():
    train_root = '/data02/zzl/dataset/WaterbodyDatas/HBWData/clip_data/train512'
    val_root = '/data02/zzl/dataset/WaterbodyDatas/HBWData/clip_data/test512'

    save_path = '/data02/zzl/dataset/WaterbodyDatas/HBWData/dataset_txt_files/HBWata_size512'
    os.makedirs(save_path, exist_ok=True)

    with open(osp.join(save_path, 'train.txt'), 'w') as f:
        label_pathes = glob(osp.join(train_root, 'label_checked', '*.tif'))
        for pth in tqdm(label_pathes):
            basename = osp.basename(pth)
            im_path = osp.join(train_root, 'image', basename)
            if not os.path.exists(im_path):
                print(f'image: {basename} dose not exist!')
                continue
            label_uncheck_path = osp.join(train_root, 'label_uncheck', basename)
            if not os.path.exists(label_uncheck_path):
                print(f'label_uncheck: {basename} dose not exist!')
                continue
            f.write('%s %s %s\n' % (im_path, pth, label_uncheck_path))

    with open(osp.join(save_path, 'test.txt'), 'w') as f:
        label_pathes = glob(osp.join(val_root, 'label_checked', '*.tif'))
        for pth in tqdm(label_pathes):
            basename = osp.basename(pth)
            im_path = osp.join(val_root, 'image', basename)
            if not os.path.exists(im_path):
                print(f'image: {basename} dose not exist!')
                continue
            label_uncheck_path = osp.join(val_root, 'label_uncheck', basename)
            if not os.path.exists(label_uncheck_path):
                print(f'label_uncheck: {basename} dose not exist!')
                continue
            f.write('%s %s %s\n' % (im_path, pth, label_uncheck_path))


def generate_test_txt():
    # for assessing unchecked samples
    list_data_root = [
                      '/data02/zzl/dataset/HBD4W/HLCSD-5']
    save_root = '/data02/zzl/dataset/HBD4W/txt_files/unchecked_samples'
    for data_root in list_data_root:
        filename = osp.basename(data_root)
        print(f'processing {filename}')
        save_path = osp.join(save_root, filename)
        os.makedirs(save_path, exist_ok=True)
        n = 0
        with open(osp.join(save_path, 'sample.txt'), 'w') as f:
            label_pathes = glob(osp.join(data_root,  'label', '*.tif'))
            for pth in tqdm(label_pathes):
                basename = osp.basename(pth)
                im_path = osp.join(data_root, 'image', basename)
                if not os.path.exists(im_path):
                    print(f'{basename} dose not exist!')
                    continue
                f.write('%s %s\n' % (im_path, pth))
                n += 1
        print(f'total num: {n}')


def copy_dataset():
    from shutil import copyfile
    data_root = '/data01/zzl/datasets/landcover.ai.dataset'
    save_root = '/data01/zzl/datasets/landcover.ai.dataset/water_samples'

    # os.makedirs(osp.join(save_root, 'train', 'image'), exist_ok=True)
    # os.makedirs(osp.join(save_root, 'train', 'prediction'), exist_ok=True)
    # os.makedirs(osp.join(save_root, 'train', 'gt'), exist_ok=True)
    # list_tif_path = glob(osp.join(data_root, 'train', 'img', '*.tif'))
    # n = 0
    # for tif_path in tqdm(list_tif_path):
    #     basename = osp.basename(tif_path)
    #     gt_path = osp.join(data_root, 'train', 'label.water', basename)
    #     prediction_path = osp.join(data_root, 'train', 'pred.water', basename)
    #     copyfile(tif_path, osp.join(save_root, 'train', 'image', f'w_{n}.tif'))
    #     copyfile(gt_path, osp.join(save_root, 'train', 'gt', f'w_{n}.tif'))
    #     copyfile(prediction_path, osp.join(save_root, 'train', 'prediction', f'w_{n}.tif'))
    #     n += 1

    os.makedirs(osp.join(save_root, 'test', 'image'), exist_ok=True)
    os.makedirs(osp.join(save_root, 'test', 'prediction'), exist_ok=True)
    os.makedirs(osp.join(save_root, 'test', 'gt'), exist_ok=True)
    list_tif_path = glob(osp.join(data_root, 'val', 'img', '*.tif'))

    n = 0
    for tif_path in tqdm(list_tif_path):
        basename = osp.basename(tif_path)
        gt_path = osp.join(data_root, 'val', 'label.water', basename)
        prediction_path = osp.join(data_root, 'val', 'pred.water', basename)

        copyfile(tif_path, osp.join(save_root, 'test', 'image', f'w_{n}.tif'))
        copyfile(gt_path, osp.join(save_root, 'test', 'gt', f'w_{n}.tif'))
        copyfile(prediction_path, osp.join(save_root, 'train', 'prediction', f'w_{n}.tif'))
        n += 1


if __name__ == '__main__':
    # generate_train_val_txt_with2inputs()
    # generate_train_val_txt_with3inputs()
    # generate_test_txt()

    copy_dataset()
