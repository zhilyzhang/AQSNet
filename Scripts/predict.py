import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
import argparse
import PIL.Image
import torch
from models.AQSNet import AQSNet
from torch.utils.data import DataLoader
import logging

from dataset.data_load import AQSDataset
import time
from tqdm import tqdm
import random
import numpy as np
import cv2 as cv
import utils.measures as measures
import json
import utils.plot as plot
import warnings
from glob import glob
import os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))

warnings.filterwarnings(action='ignore')


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def denormalize(data, mean, std):
    mean = torch.Tensor(mean).type_as(data)
    std = torch.Tensor(std).type_as(data)
    return data.mul(std[..., None, None]).add(mean[..., None, None])


def get_confusion_matrix(label, pred, num_classes, ignore=-1):
    """
    calculate the confusion matrix by label and pred.
    """
    output = pred.cpu().numpy()#.transpose(0, 2, 3, 1)
    # pred_seg = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    pred_seg = np.asarray(output, dtype=np.uint8)
    gt_seg = np.asarray(label.cpu().numpy(), dtype=np.int32)

    ignore_index = gt_seg != ignore
    gt_seg = gt_seg[ignore_index]
    pred_seg = pred_seg[ignore_index]

    index = (gt_seg*num_classes+pred_seg).astype(np.int32)

    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred in range(num_classes):
            cur_index = i_label * num_classes + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def get_data_loaders(opts, Dataset):
    print('building dataloaders')
    dataset_val = Dataset(txt_file=opts['test_data_txt'],
                          mode='val',
                          use_reality_data=opts['use_reality_data'],
                          normal_imagenet=opts['normal_imagenet'])

    val_loader = DataLoader(dataset=dataset_val, batch_size=opts['batch_size'],
                            shuffle=False, num_workers=opts['num_workers'])
    list_filename = [os.path.basename(p.split()[0]) for p in dataset_val.ids]
    return val_loader, list_filename


class Prediction(object):
    def __init__(self, args):
        self.best_performance = 0
        try:
            self.opts = json.load(open(args.exp, 'r'))
        except:
            self.opts = json.load(open(osp.join(root, args.exp), 'r'))
        print(self.opts)
        self.device = torch.device('cuda:{}'.format(self.opts['GPU']))
        self.network_name = self.opts['network_name']
        self.val_loader, self.list_filename = get_data_loaders(self.opts, AQSDataset)
        self.model = AQSNet(in_channels=4, backbone='hrnet48',
                            with_mode='scam', with_aux=self.opts['with_aux']).to(self.device)

        if self.opts['is_vis_map']:
            self.vis_path = os.path.join(self.opts['exp_dir'], self.network_name, 'vis_dirs')
            os.makedirs(self.vis_path, exist_ok=True)

        self.resume(self.opts['weight_path'])

    def resume(self, path):
        pre_weight = torch.load(path, map_location='cpu')

        model_dict = self.model.state_dict()
        num = 0
        for k, v in pre_weight.items():
            if 'model.' in k:
                k = k[6:]
            if k in model_dict.keys():
                model_dict[k] = v
                num += 1
        print(f'model-load-weight:{num}/{len(pre_weight.keys())}/{len(model_dict.keys())}')
        self.model.load_state_dict(model_dict)

    def image_mask_with_rgb(self, im, mask, rgb=[255, 0, 0]):
        for i in range(3):
            im[:, :, i][mask == 255] = rgb[i]
        return im

    def predict(self, vis=False):
        start_time = time.time()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        IoU, Recall, Precision, Accuracy, F1, errors = self.validation(vis=vis)
        print('IoU: %.3f%%' % (IoU.item() * 100))
        print('Recall: %.3f%%' % (Recall.item() * 100))
        print('Precision: %.3f%%' % (Precision.item() * 100))
        print('Accuracy: %.3f%%' % (Accuracy.item() * 100))
        print('F1: %.3f%%' % (F1.item() * 100))
        print('error: %.3f%%' % (errors.item() * 100))
        duration1 = time.time() - start_time
        print('Val-used-time:', duration1)

    def vis(self, im, label_uncheck, label_checked, label_res, pred_res):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        mean = torch.Tensor(mean).type_as(im)
        std = torch.Tensor(std).type_as(im)
        im = im[0].mul(std[..., None, None]).add(mean[..., None, None])
        im = np.transpose(im.cpu().numpy() * 255, (1, 2, 0)).astype(np.uint8)
        label_uncheck = (label_uncheck[0].squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        label_checked = (label_checked[0].squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        label_res = (label_res[0].detach().cpu().numpy() * 255).astype(np.uint8)
        pred_res = (pred_res[0].detach().cpu().numpy() * 255).astype(np.uint8)
        label_res[label_res>127] = 255
        label_res[label_res<=127] = 0

        pred_res[pred_res > 127] = 255
        pred_res[pred_res <= 127] = 0
        ld_rgb = [0, 0, 255]
        err_rgb = [255, 0, 0]
        im_label_res = self.image_mask_with_rgb(im.copy(), label_res[0], rgb=ld_rgb)
        im_label_res = self.image_mask_with_rgb(im_label_res, label_res[1], rgb=err_rgb)

        im_pred_res = self.image_mask_with_rgb(im.copy(), pred_res[0], rgb=ld_rgb)
        im_pred_res = self.image_mask_with_rgb(im_pred_res, pred_res[1], rgb=err_rgb)
        return im, label_uncheck, label_checked, im_label_res, im_pred_res

    def validation(self, vis=False):
        self.model.eval()
        TP, Acc, PN, GN = [], [], [], []
        error = []
        for i_batch, (img, label_uncheck, label_checked, label_res) in tqdm(enumerate(self.val_loader)):
            img = img.to(self.device)
            label_uncheck = label_uncheck.to(self.device)
            label_checked = label_checked.to(self.device)
            label_res = label_res.to(self.device)
            with torch.no_grad():
                pred_res, _ = self.model(img, label_uncheck)  #

            if vis:
                im, label_uncheck, label_checked, im_label_res, im_pred_res = self.vis(img, label_uncheck, label_checked, label_res, pred_res)
                plot.visualize_list_data(list_data=[im, label_checked, label_uncheck, im_label_res, im_pred_res],
                                         M=2, N=3,
                                         list_titles=['im', 'label_checked', 'label_uncheck', 'im_label_res', 'im_pred_res'],
                                         save_dirs=os.path.join(self.vis_path, self.list_filename[i_batch]),
                                         show_data=False)
            tp, pn, gn, acc = measures.calculate_tp_pn_gn_accuracy(pred_res, label_res, threshold=0.5)
            er = measures.calculate_error(pred_res, threshold=0.5)
            TP.append(tp)
            PN.append(pn)
            GN.append(gn)
            Acc.append(acc)
            error.append(er)

        IoU = sum(TP) / (sum(PN) + sum(GN) - sum(TP) + 1e-9)
        Recall = sum(TP) / (sum(GN) + 1e-9)
        Precision = sum(TP) / (sum(PN) + 1e-9)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        Accuracy = sum(Acc) / len(Acc)
        errors = sum(error) / (len(error))
        return IoU, Recall, Precision, Accuracy, F1, errors

    def vis_results(self):
        label_uncheck_path = '/data02/zzl/dataset/BuildingDatas/building_labeling_check/BLCDataset/val/label_uncheck'
        save_path = '/data02/zzl/model_results/quality_check/whu_analysis_results/ablation_backbone/hrnet'
        image_path = '/data02/zzl/dataset/BuildingDatas/building_labeling_check/BLCDataset/val/image'

        os.makedirs(save_path, exist_ok=True)
        self.resume(self.opts['weight_path'])
        # self.resume_alter(self.opts['weight_path'])
        self.model.eval()
        for i_batch, (img, label_uncheck, label_checked, label_res) in tqdm(enumerate(self.val_loader)):
            img = img.to(self.device)
            label_uncheck = label_uncheck.to(self.device)

            with torch.no_grad():
                # pred_res = self.model(img, label_uncheck)
                pred_res = self.model(img, label_uncheck)

            pred_res = (pred_res[0].detach().cpu().numpy() * 255).astype(np.uint8)
            pred_res[pred_res > 127] = 255
            pred_res[pred_res <= 127] = 0
            basename = self.list_filename[i_batch]

            image = np.asarray(PIL.Image.open(osp.join(image_path, basename)))
            label_uncheck = cv.imread(osp.join(label_uncheck_path, basename), cv.IMREAD_GRAYSCALE)
            plot.vis_image_with_questioned_pixels(res_miss=pred_res[0],
                                                  res_error=pred_res[1],
                                                  label_uncheck=label_uncheck,
                                                  image=image,
                                                  save_tif_path=osp.join(save_path, basename))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,
                        default='Experiments/aqs_metrics/water_reality.json')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print('==> Init Params')
    trainer = Prediction(args)
    print('==> Start predicting')
    trainer.predict(vis=False)

    '''
    "test_data_txt": "/data02/zzl/dataset/WaterbodyDatas/HBWData/dataset_txt_files/HBWata_size512/test.txt"
    "test_data_txt": "/data02/zzl/dataset/BuildingDatas/building_labeling_check/txt_files/BLCDataset/val.txt"
    "test_data_txt": "/data02/zzl/model_results/quality_check/whu_analysis_results/ablation_model/vis_sample.txt",
    /data02/zzl/model_results/quality_check/HBW_results/result_analysis_vis/ablation_backbone/vis_sample.txt
    
    '''
