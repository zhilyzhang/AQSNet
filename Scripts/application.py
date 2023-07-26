import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
import argparse
import torch
from models.AQSNet import AQSNet
from torch.utils.data import DataLoader

from dataset.data_load import AQSDataset
from tqdm import tqdm
import random
import numpy as np
import json
import os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))

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
                          mode='test',
                          use_reality_data=opts['use_reality_data'],
                          normal_imagenet=opts['normal_imagenet'])

    val_loader = DataLoader(dataset=dataset_val, batch_size=opts['batch_size'],
                            shuffle=False, num_workers=opts['num_workers'])
    list_filename = [p.strip().split()[0] for p in dataset_val.ids]
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

        self.model = AQSNet(in_channels=4, backbone=self.opts['backbone'], with_aux=self.opts['with_aux']).to(self.device)
        self.save_checked_samples_path = os.path.join(self.opts['save_checked_samples_path'], self.network_name)
        os.makedirs(self.save_checked_samples_path, exist_ok=True)
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

    def applying(self):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        self.validation()

    def validation(self):
        self.model.eval()
        # threshold = 0.3
        threshold = 0.25
        validated_num = 0
        with open(osp.join(self.save_checked_samples_path, 'checked_sample.txt'), 'w') as f:
            for i_batch, (img, label_uncheck) in tqdm(enumerate(self.val_loader)):
                img = img.to(self.device)
                label_uncheck = label_uncheck.to(self.device)

                with torch.no_grad():
                    pred_res, _ = self.model(img, label_uncheck)  # , pred_mask
                mask_pred = (pred_res > 0.5).float()
                target_mask = label_uncheck.squeeze(1) + mask_pred[:, 0, ...] - mask_pred[:, 1, ...]
                # print(target_mask.max(), target_mask.min())
                target_mask = torch.clamp(target_mask, 0, 1)
                pred_mask = mask_pred[:, 0, ...] + mask_pred[:, 1, ...]
                # print(pred_mask.max(), pred_mask.min())
                pred_mask = torch.clamp(pred_mask, 0, 1)
                if torch.sum(pred_mask) / (torch.sum(target_mask) + 1e-9) < threshold:
                    basename = osp.basename(self.list_filename[i_batch])
                    # uncheck_path = self.list_filename[i_batch].replace('image', 'label_uncheck')
                    uncheck_path = self.list_filename[i_batch].replace('image', 'label')
                    im_path = self.list_filename[i_batch]
                    if not os.path.exists(im_path):
                        print(f'{basename} dose not exist!')
                        continue
                    f.write('%s %s\n' % (im_path, uncheck_path))
                    validated_num += 1
        print(f'total num/validate num: {len(self.val_loader)}/{validated_num}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,
                        default='/data02/zzl/codes2022/AQSNet/Experiments/aqs_applications/building_data_checking.json')
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print('==> Init Params')
    t = Prediction(args)
    print('==> Start predicting')
    t.applying()
