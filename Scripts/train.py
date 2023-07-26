import os, sys
import os.path as osp
root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
# print(root)
sys.path.insert(0, root)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from models.AQSNet import AQSNet
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import datetime
import logging
from dataset.data_load import AQSDataset
from utils.losses import BinSegLoss
import time
from tqdm import tqdm
import random
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn


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


class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss=None, aux_loss=None, with_aux=False):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss
    self.aux_loss = aux_loss
    self.with_ab = with_aux
    if aux_loss == 'l1':
        self.r_loss = nn.L1Loss()
    elif aux_loss == 'l2':
        self.r_loss = nn.MSELoss()
    else:
        self.r_loss = None

  def forward(self, inputs, check_inputs, labels, gt_res):
    out, seg = self.model(inputs, check_inputs)
    loss = 0
    loss += self.loss(out[:, :1, ...], gt_res[:, :1, ...])  # 漏的
    loss += self.loss(out[:, 1:, ...], gt_res[:, 1:, ...])  # 错的
    if self.with_ab:
        loss += self.loss(seg, labels)

    if self.aux_loss is not None:
        re_check_res = check_inputs + out[:, :1, ...] - out[:, 1:, ...]
        loss += self.r_loss(re_check_res, labels)

    return torch.unsqueeze(loss, 0), out


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


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


def train(args):
    cudnn.benchmark = False
    cudnn.deterministic = False
    cudnn.enabled = True
    distributed = args.distributed
    device = torch.device('cuda:{}'.format(args.local_rank))
    process_writer = not distributed or (distributed and args.local_rank == 0)
    snapshot_path = args.save_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    os.makedirs(snapshot_path, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if process_writer:
        tb_writer = SummaryWriter(snapshot_path + '/log')
        fileName = datetime.datetime.now().strftime('log_' + '%m_%d_%H')
        logger = get_logger(os.path.join(snapshot_path, fileName + ".log"))
        logger.info(str(args))

    use_reality_data = not args.with_simulated
    train_dataset = AQSDataset(txt_file=args.train_txt_path, mode='train',
                               normal_imagenet=True, use_reality_data=use_reality_data)
    #
    # valid_dataset = AQSDataset(txt_file=args.val_txt_path, mode='val',
    #                            normal_imagenet=True, use_reality_data=use_reality_data)

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        train_sampler = DistributedSampler(train_dataset)
        # val_sampler = DistributedSampler(valid_dataset)
    else:
        train_sampler = None
        # val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, drop_last=True, sampler=train_sampler)

    # val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
    #                         num_workers=args.num_workers, drop_last=True, sampler=val_sampler)

    if process_writer:
        logger.info("The length of Train set is: {}".format(len(train_loader) * args.batch_size))

    with_aux = args.with_aux
    model = AQSNet(in_channels=4, backbone='hrnet48', with_aux=with_aux)  #espa
    if os.path.exists(args.pre_weights):
        model.init_weights(model_pretrained=args.pre_weights)
    else:
        print('without loading pretrained weights!')

    model = FullModel(model, loss=BinSegLoss(), aux_loss='l1', with_aux=with_aux)
    model.to(device)
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.90, weight_decay=5e-4, nesterov=True)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    max_epoch = args.max_epochs
    num_per_iter = len(train_loader)
    max_iterations = args.max_epochs * num_per_iter  # max_epoch = max_iterations // len(trainloader) + 1
    num_print = num_per_iter//100 + 1
    if process_writer:
        logger.info("{} iterations per epoch. {} max iterations ".format(num_per_iter, max_iterations))
    iter_num = 0
    best_performance = 0.0
    num_batches = len(train_loader)

    start_time = time.time()
    iterator = tqdm(range(1, max_epoch+1), ncols=70)
    num_val_iter = 1000
    tmp_times = num_per_iter // num_val_iter
    is_pre_val = True if tmp_times >= 2 else False

    # training!
    for epoch_num in iterator:
        if distributed:
            train_sampler.set_epoch(epoch_num)
            # val_sampler.set_epoch(epoch_num)
        model.train()
        if process_writer:
            logger.info('training epoch: %d' % epoch_num)
        for i_batch, (images, uncheck_mask, labels, gt_res) in enumerate(train_loader):
            # img, uncheck_mask, gt_mask, res
            optimizer.lr = args.l_rate * (1 - float(iter_num) / (num_batches * args.max_epochs)) ** 0.9

            images = images.to(device)
            uncheck_mask = uncheck_mask.to(device)
            labels = labels.to(device)
            gt_res = gt_res.to(device)

            b, _, m, n = images.shape
            losses, out = model(images, uncheck_mask, labels, gt_res)
            predict = out > 0.5
            gt = gt_res
            acc = (predict == gt).sum().item() / (b * m * n) / 2
            loss = losses.mean()

            if distributed:
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss

            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 40)
            optimizer.step()

            if process_writer:
                tb_writer.add_scalar('info/total_loss', reduced_loss.item(), iter_num)
                tb_writer.add_scalar('info/acc', acc, iter_num)

            if i_batch % num_print == 0:
                if process_writer:
                    logger.info(
                        'i/each_iter/epoch %d/%d/%d : loss : %.4f ,acc : %.4f'
                        % (i_batch, num_per_iter, epoch_num, reduced_loss.item(), acc))
                # show_data(images, labels, out, save_fig=os.path.join(snapshot_path, f'train_epoch{epoch_num}_{i_batch%10}.png'))
            iter_num += 1

        if process_writer:
            if epoch_num >= max_epoch - 5:
                save_mode_path = os.path.join(snapshot_path, f'weight_{epoch_num}.pth')
                if distributed:
                    torch.save(model.module.state_dict(), save_mode_path)
                else:
                    torch.save(model.state_dict(), save_mode_path)

                logger.info("save model to {}".format(save_mode_path))

        duration1 = time.time() - start_time
        start_time = time.time()
        if process_writer:
            logger.info('Train running time: %.2f(minutes)' % (duration1 / 60))

    if process_writer:
        iterator.close()
        tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=7)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--distributed", action='store_true', default=False, help='if or not with distributed')
    parser.add_argument("--with_simulated", action='store_true', default=False, help='if or not with simulated data')
    parser.add_argument("--with_aux", action='store_true', default=True, help='if or not with an aux branch')

    parser.add_argument(
        "--pre_weights",
        type=str,
        # default='')
        default='/data02/zzl/codes2022/AQSNet/trained_weights/baqs/aqsnet_baqs_hrnet.pth')

    parser.add_argument("--train_txt_path", type=str,
                        default='/data02/zzl/dataset/BuildingDatas/building_labeling_check/txt_files/BLCDataset/train.txt')

    # parser.add_argument("--val_txt_path", type=str, default=')

    parser.add_argument("--save_path", type=str, default='/data02/zzl/model_results/quality_check/HBW_results/DQCNet_simulated_data/')

    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--max_epochs', nargs='?', type=int, default=36,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=14,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=4e-3,
                        help='Learning Rate')
    parser.add_argument('--ratio', nargs='?', type=float, default=0.5,
                        help='Learning Rate')

    args = parser.parse_args()

    print(f'GPU: {args.local_rank}')

    args.max_epochs = 36
    args.batch_size = 8
    args.img_size = 512

    train(args)

# python Scripts/train/train.py
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 29503 --nproc_per_node=2 --nnodes=1 ./Scripts/train.py

