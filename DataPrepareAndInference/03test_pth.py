import math
import os.path

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import nibabel as nib
from medpy import metric
from main_program.roi_localization.networks.vnet import VNet

# 测试训练出来的 .pth
# 找的别人的代码，中间不知道发生了啥，反正输出就是 原图、原标签和预测标签的nii.gz


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map, score_map


def test_all_case(net, image_list, num_classes=2, patch_size=(112, 112, 80),
                  stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for ith, image_path in enumerate(image_list):
        print(f'now: {image_path}')
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_pred.nii.gz" % (ith))
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_img.nii.gz" % (ith))
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_gt.nii.gz" % (ith))
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def make_dir(dir_path):
    import os
    """
    创建空文件夹
    :param dir_path:
    :return:
    """
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)


if __name__ == '__main__':
    data_path = r'E:\Jupter\ctooth\TrainData\test.list'  # 存放数据名称的列表
    test_save_path = 'predictions12/'  # 模型出来后的nii.gz的保存位置
    make_dir(test_save_path)
    save_mode_path = r'E:\Jupter\ctooth\main_program\model\run1025_01\iter_10002.pth'  # 需要读取的模型位置

    # 读取网络
    net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    # 开始验证
    net.eval()
    with open(data_path, 'r') as f:
        image_list = f.readlines()

    nii_norm_path = r'E:\Jupter\ctooth\TrainData\nii_train_h5'

    # 从
    image_list = [os.path.join(nii_norm_path, item.replace('\n', '').split('.')[0]+"_norm.h5") for item in image_list]
    # 滑动窗口法
    print(image_list)
    avg_metric = test_all_case(net, image_list, num_classes=2,
                               patch_size=(128, 128, 128), stride_xy=128, stride_z=128,
                               save_result=True, test_save_path=test_save_path)
