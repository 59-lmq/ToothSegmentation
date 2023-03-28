import math
import os.path
import time

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import nibabel as nib
from medpy import metric
from networks.vnet import VNet
from loguru import logger


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_single_case(net, image, stride_xy, stride_z, ps, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < ps[0]:
        w_pad = ps[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < ps[1]:
        h_pad = ps[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < ps[2]:
        d_pad = ps[2]-d
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

    sx = math.ceil((ww - ps[0]) / stride_xy) + 1
    sy = math.ceil((hh - ps[1]) / stride_xy) + 1
    sz = math.ceil((dd - ps[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-ps[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-ps[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-ps[2])
                test_patch = image[xs:xs+ps[0], ys:ys+ps[1], zs:zs+ps[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+ps[0], ys:ys+ps[1], zs:zs+ps[2]] += y
                cnt[xs:xs+ps[0], ys:ys+ps[1], zs:zs+ps[2]] += 1
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
        if ith == 0:
            continue
        st = time.time()
        print(f'now: {image_path}')
        logger.info(f'现在处理的是：{image_path}')
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        # break
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        logger.info(f' prediction： {prediction}')
        # break

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        # print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
        logger.info(f'第 {ith} 个数据中，dice: {single_metric[0]}, jc: {single_metric[1]}, '
                    f'hd: {single_metric[2]} asd: {single_metric[3]}')
        total_metric += np.asarray(single_metric)

        if save_result:
            sub_name = image_path.split('\\')[-1].split('_')[0]
            pred_name = os.path.join(test_save_path, sub_name + "%02d_pred.nii.gz" % (ith))
            orin_name = os.path.join(test_save_path, sub_name + "%02d_img.nii.gz" % (ith))
            grou_name = os.path.join(test_save_path, sub_name + "%02d_gt.nii.gz" % (ith))
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), pred_name)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), orin_name)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), grou_name)

            logger.info(f'原图保存到：{orin_name}')
            logger.info(f'预测标签保存到：{pred_name}')
            logger.info(f'原标签保存到：{grou_name}')
        logger.info(f'第{ith}个数据，耗时：{time.time() - st} s')

    avg_metric = total_metric / len(image_list)
    logger.info(f'平均的 dice, jc, hd, asd为：{avg_metric}')
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
        os.makedirs(dir_path)


def teeth_predict():
    exp_name = 'exp_name'
    snapshot_path = "../experiments/" + exp_name

    log_path = os.path.join(snapshot_path, 'log/predict_')
    logger.add(log_path + '{time}.log', rotation='00:00')

    data_path = r'.\nii_data\data_list'  # 存放数据名称的列表
    data_name = os.path.join(data_path, 'test_binary.list')
    test_save_path = '../results/' + exp_name  # 模型出来后的nii.gz的保存位置
    make_dir(test_save_path)
    save_mode_path = r'../experiments/' + exp_name + '/iter_501.pth'  # 需要读取的模型位置
    patch_size = (48, 48, 48)
    num_class = 33
    stride_xy = 48
    stride_z = 48
    
    logger.info(f'开始进行预测。')
    logger.info(f'类别：{num_class}，窗口尺寸：{patch_size}, xy方向上的步长：{stride_xy}, z方向上的步长：{stride_z}')
    logger.info(f'测试的列表路径：{data_name}')

    # 读取网络
    net = VNet(n_channels=1, n_classes=num_class, normalization='batchnorm', has_dropout=True).cuda()
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    logger.info(f'模型初始权重来自于：{save_mode_path}')

    # 开始验证
    net.eval()
    with open(data_name, 'r') as f:
        image_list = f.readlines()

    # 从
    image_list = [item.replace('\n', '') for item in image_list]
    # 滑动窗口法
    print(image_list)
    logger.info(f'测试集中的数据分别为：{image_list}')
    avg_metric = test_all_case(net, image_list, num_classes=num_class,
                               patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                               save_result=True, test_save_path=test_save_path)
    logger.info(f'预测结束！最终平均的dice, jc, hd, asd分别为： {avg_metric}')


if __name__ == '__main__':
    teeth_predict()
