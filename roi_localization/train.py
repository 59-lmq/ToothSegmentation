import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from utils.losses import dice_loss
# from dataloaders.toothLoader import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from dataloaders.CToothLoader import CToothImageLoader, RandomCrop, CenterCrop, \
    RandomRotFlip, ToTensor, TwoStreamBatchSampler, RandomNoise


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='..', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='exp_1028_01', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=7001, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--num_workers', type=int,  default=0, help='num-workers to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../experiments/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
# print(batch_size)
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (128, 128, 128)
num_classes = 2
data_path = {
    "train": r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI\train.list',
    "test": r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI\test.list',
    "nii": r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI\nii_train_h5'
}


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()
    # save_mode_path = os.path.join(
    # '/hpc/data/home/bme/v-cuizm/project/NC/model/binary_seg_ROI_HZ_02_(142data_256size)/iter_10000.pth')
    # net.load_state_dict(torch.load(save_mode_path))

    db_train = CToothImageLoader(base_dir=train_data_path,
                                 data_path=data_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     RandomCrop(patch_size),
                                     RandomNoise(),
                                     RandomRotFlip(),
                                     ToTensor(),
                                     ]))
    db_test = CToothImageLoader(base_dir=train_data_path,
                                data_path=data_path,
                                split='test',
                                transform=transforms.Compose([
                                     RandomCrop(patch_size),
                                     ToTensor()
                                     ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True,  num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log/run%d' % args.max_iterations)
    logging.info("{} itertations per epoch".format(len(trainloader)))
    # print(f'len(trainloader): {len(trainloader)}')
    # print(f'trainloader: {trainloader}')
    # print(f'db_train: {db_train}')

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = net(volume_batch)

            loss_seg = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            loss = 0.5*(loss_seg+loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f , dice_acc : %f' % (iter_num, loss.item(), 1-loss_seg_dice.item()))

            # change lr
            if iter_num % 5000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num % 200 == 0:
                net.eval()
                test_loss = 0
                iter_test = 0
                for i_batch, sampled_batch in enumerate(testloader):
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    with torch.no_grad():
                        outputs = net(volume_batch)

                    loss_seg = F.cross_entropy(outputs, label_batch)
                    outputs_soft = F.softmax(outputs, dim=1)
                    loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
                    loss = 0.5*(loss_seg+loss_seg_dice)
                    print('---test for seg:', 1 - loss_seg_dice.item())
                    test_loss = test_loss + loss
                    iter_test = iter_test + 1
                writer.add_scalar('loss_test/test_loss', test_loss/iter_test, iter_num)
                net.train()
                del volume_batch, label_batch, loss_seg, outputs_soft, loss_seg_dice

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
