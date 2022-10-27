
import os
from tqdm import tqdm

import numpy as np  # 转换格式
import nibabel as nib  # 读取数据
from nibabel.viewers import OrthoSlicer3D  # 可视化
import h5py
import matplotlib
matplotlib.use('TkAgg')  # 用于滚动查看nii.gz

# 用来将nii.gz -> .h5


def make_dir(dir_path):
    """
    创建空文件夹
    :param dir_path:
    :return:
    """
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)


def save_h5(x_data, y_data, file_name):
    """
    保存 .h5 文件
    :param x_data:  保存的image数据
    :param y_data: 保存的label数据
    :param file_name: 保存的文件名
    :return:
    """
    f = h5py.File(file_name, 'w')
    f.create_dataset('image', data=x_data, compression="gzip")
    f.create_dataset('label', data=y_data, compression="gzip")
    f.close()


def load_h5(file_path):
    """
    读取 .h5文件
    :param file_path:需要读取的文件路径名
    :return:
    """
    # 读取h5文件
    h5f = h5py.File(file_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    print(type(image), type(label))
    print(image.shape, label.shape)


def GainNorm(data):
    """
    归一化数据
    :param data:需要归一化的数据
    :return: 返回归一化后的数据
    """
    # 归一化
    norm_data = (data - np.mean(data)) / (np.std(data))
    return norm_data


def RandomCropNumpy(x_data, y_data, out_shape):
    # 随机裁剪 266x266x200 -> 128x128x128
    d, h, w = x_data.shape
    crop_depth = np.random.randint(0, d - out_shape[0])
    crop_height = np.random.randint(0, h - out_shape[1])
    crop_width = np.random.randint(0, w - out_shape[2])
    print(f'random (d, w, h): ({crop_depth, crop_height, crop_width})')

    crop_x_data = x_data[crop_depth:crop_depth + out_shape[0],
                        crop_height:crop_height + out_shape[1],
                        crop_width:crop_width + out_shape[2]]
    crop_y_data = y_data[crop_depth:crop_depth + out_shape[0],
                        crop_height:crop_height + out_shape[1],
                        crop_width:crop_width + out_shape[2]]
    return crop_x_data, crop_y_data


def covert_h5():
    """
    转成 h5形式
    :return:
    """
    # out_shape = [128, 128, 128]  # 输出的大小
    image_path = r'../image'  # 原图像路径
    label_path = r'../label'  # 原标签路径

    output_path = r'../nii_Train_4dim'  # 输出是 1x128x128x128的 .h5文件路径
    output_path2 = r'../nii_Train_3dim'  # 输出是 128x128x128的 .h5文件路径
    output_path3 = r'../nii_Train_3dim_norm'  # 输出是 image.shape 的 .h5文件路径  我训练用的是这个

    make_dir(output_path)
    make_dir(output_path2)
    make_dir(output_path3)
    # if os.path.exists(output_path) is False:
    #     os.mkdir(output_path)
    # if os.path.exists(output_path2) is False:
    #     os.mkdir(output_path2)
    # if os.path.exists(output_path3) is False:
    #     os.mkdir(output_path3)

    image_list = os.listdir(image_path)
    for case in tqdm(image_list):
        if case.split('.')[1] != 'nii':
            continue
        # print(case)
        # 0、读取nii.gz文件
        image_object = nib.load(os.path.join(image_path, case))
        label_object = nib.load(os.path.join(label_path, case))
        # 1、读取数据
        x_data = image_object.get_fdata()
        y_data = label_object.get_fdata()

        # 2、归一化
        x_norm = GainNorm(x_data)
        y_norm = GainNorm(y_data)

        # # 3、随机裁剪到[128, 128, 128]
        # x_crop, y_crop = RandomCropNumpy(x_norm, y_norm, out_shape)

        # # 4、增加一个维度，方便获取
        # x_transpose = x_crop[np.newaxis, :, :, :]
        # y_transpose = y_crop[np.newaxis, :, :, :]

        # # 保存到输出是 1x128x128x128的 .h5文件路径
        # case_dir = os.path.join(output_path, case.split('.')[0]+'_norm.h5')
        # save_h5(x_transpose, y_transpose, case_dir)

        # # 保存到输出是 128x128x128的 .h5文件路径  我训练用的是这个
        # case_dir2 = os.path.join(output_path2, case.split('.')[0] + '_norm.h5')
        # save_h5(x_crop, y_crop, case_dir2)

        # 保存到输出是 128x128x128的 .h5文件路径  我训练用的是这个
        case_dir3 = os.path.join(output_path3, case.split('.')[0] + '_norm.h5')
        save_h5(x_norm, y_norm, case_dir3)
        print(f'norm_h5 save at:{case_dir3}')

        """
        print(f'case_dir:{case_dir}')
        f = h5py.File(case_dir, 'w')
        f.create_dataset('image', data=x_transpose, compression="gzip")
        f.create_dataset('label', data=y_transpose, compression="gzip")
        f.close()

        print(f'case_dir2:{case_dir2}')
        f = h5py.File(case_dir2, 'w')
        f.create_dataset('image', data=x_crop, compression="gzip")
        f.create_dataset('label', data=y_crop, compression="gzip")
        f.close()

        print(f'case_dir2:{case_dir3}')
        f = h5py.File(case_dir3, 'w')
        f.create_dataset('image', data=x_norm, compression="gzip")
        f.create_dataset('label', data=y_norm, compression="gzip")
        f.close()
        """

        # print(type(image_object))
        # image, img_header = nrrd.read(os.path.join(data_path,case, 'lgemri.nrrd'))
        # label, gt_header = nrrd.read(os.path.join(data_path,case, 'laendo.nrrd'))
        # label = (label == 255).astype(np.uint8)
        # w, h, d = label.shape
        # # 返回label中所有非零区域（分割对象）的索引
        # tempL = np.nonzero(label)
        # # 分别获取非零区域在x,y,z三轴的最小值和最大值，确保裁剪图像包含分割对象
        # minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        # miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        # minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        # # 计算目标尺寸比分割对象多余的尺寸
        # px = max(output_size[0] - (maxx - minx), 0) // 2
        # py = max(output_size[1] - (maxy - miny), 0) // 2
        # pz = max(output_size[2] - (maxz - minz), 0) // 2
        # # 在三个方向上随机扩增
        # minx = max(minx - np.random.randint(10, 20) - px, 0)
        # maxx = min(maxx + np.random.randint(10, 20) + px, w)
        # miny = max(miny - np.random.randint(10, 20) - py, 0)
        # maxy = min(maxy + np.random.randint(10, 20) + py, h)
        # minz = max(minz - np.random.randint(5, 10) - pz, 0)
        # maxz = min(maxz + np.random.randint(5, 10) + pz, d)
        # # 图像归一化，转为32位浮点数（numpy默认是64位）
        # image = (image - np.mean(image)) / np.std(image)
        # image = image.astype(np.float32)
        # # 裁剪
        # image = image[minx:maxx, miny:maxy, minz:maxz]
        # label = label[minx:maxx, miny:maxy, minz:maxz]
        # print(label.shape)
        #
        # case_dir = os.path.join(out_path, case)
        # os.mkdir(case_dir)
        # f = h5py.File(os.path.join(case_dir, 'mri_norm2.h5'), 'w')
        # f.create_dataset('image', data=image, compression="gzip")
        # f.create_dataset('label', data=label, compression="gzip")
        # f.close()


def new_covert_h5():
    """
    转成h5格式
    :return:
    """
    # out_shape = [128, 128, 128]  # 输出的大小
    image_path = r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI/image'  # 原图像路径
    label_path = r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI/label'  # 原标签路径
    # 输出是 image.shape 的 .h5文件路径  我训练用的是这个
    output_path3 = r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI/nii_train_h5'

    make_dir(output_path3)

    image_list = os.listdir(image_path)
    for case in tqdm(image_list):
        # print(case.split('_')[0], case.split('_'))
        if case.split('.')[1] != 'nii':
            continue
        # if case.split('_')[0] != 'Teeth':
        #     continue
        # print(case)
        # 0、读取nii.gz文件
        image_object = nib.load(os.path.join(image_path, case))
        label_object = nib.load(os.path.join(label_path, case))
        # 1、读取数据
        x_data = image_object.get_fdata()
        y_data = label_object.get_fdata()

        # 2、归一化
        x_norm = GainNorm(x_data)
        y_norm = GainNorm(y_data)
        # print(case, x_data.shape, y_data.shape, x_data.max(), y_data.max())
        # break

        # 3、保存到 .h5文件路径  我训练用的是这个
        case_dir3 = os.path.join(output_path3, case.split('.')[0] + '_norm.h5')
        save_h5(x_norm, y_norm, case_dir3)
        print(f'norm_h5 save at:{case_dir3}')


def load_nii(nii_path):
    nii_object = nib.load(nii_path)
    nii_numpy = nii_object.get_fdata()
    print(nii_numpy.shape)


if __name__ == '__main__':
    # 1、查看单个.nii.gz文件
    # nii_path = r'E:\Jupter\ctooth\OriginalData\image\CBCT2209135.nii.gz'
    # load_nii(nii_path)

    # 2、对图像和标签进行转换
    # new_covert_h5()

    # 3、查看转换后的数据
    file_path = r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI/nii_train_h5\CBCT2208200_norm.h5'
    load_h5(file_path)

