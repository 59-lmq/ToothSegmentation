import numpy as np  # 转换格式
import nibabel as nib  # 读取数据
from nibabel.viewers import OrthoSlicer3D  # 可视化
import matplotlib
matplotlib.use('TkAgg')  # 用于滚动查看nii.gz
import os
import nrrd

import SimpleITK as sitk


def make_dir(dir_path):
    """
    创建空文件夹
    :param dir_path:
    :return:
    """
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)


def dicom2nii():
    reader = sitk.ImageSeriesReader()

    folderPath = r'E:\Jupter\ctooth\瑞通data\CBCT-data\CT\\'  # dicom图片所在文件夹
    output_path = r'E:\Jupter\ctooth\OriginalData\image'  # 文件保存的路径
    folder_list = os.listdir(folderPath)
    for folder in folder_list:
        # print(folder)
        dicom_images_path = os.path.join(output_path, folder+'.nii.gz')
        # print(dicom_images_path)
        folder_path = os.path.join(folderPath, folder)
        # print(folder_path)
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
        reader.SetFileNames(dicom_names)

        dicom_images = reader.Execute()
        sitk.WriteImage(dicom_images, dicom_images_path)
        print(f'save successfully! folder path:{dicom_images_path}')


def nrrd2nii():
    folderPath = r'E:\Jupter\ctooth\瑞通data\CBCT-data\Label\\'  # nrrd标签图片所在文件夹
    output_path = r'E:\Jupter\ctooth\OriginalData\label'  # 文件保存的路径

    folder_list = os.listdir(folderPath)
    # print(folder_list)
    for folder in folder_list:
        if len(folder.split('_')) == 1:
            continue
        # print(folder.split('_'))
        # print(folder)
        label_path = os.path.join(folderPath, folder)
        label_numpy, label_header = nrrd.read(label_path)
        # # label_unique = np.unique(label_numpy)
        # # print(type(label_numpy), label_numpy)
        label_nib = nib.Nifti1Image(label_numpy, np.eye(4))
        label_nii_name = os.path.join(output_path, "CBCT"+folder.split('.')[0].split('l')[1]+".nii.gz")
        print(label_nii_name)
        # # print(f'label_numpy min & max:[{label_numpy.min(), label_numpy.max()}]')
        # # print(f'unique label:{label_unique}')
        # # print(f'type unique label:{type(label_unique)}')
        # # print(f'len unique label:{len(label_unique)}')
        # # print(f'len unique:{len(label_unique)==2}')
        nib.save(label_nib, label_nii_name)
        # break


def visual3d():

    new_image_object = nib.load('./1_image.nii.gz')
    OrthoSlicer3D(new_image_object.dataobj).show()


def print_type(data, name):
    """
    打印输出类型和shape
    :param data: 打印的目标
    :param name: 打印的名字
    :return: 无
    """
    print(f'type({name}): {type(data)}, {name}.shape:{data.shape}')


def fix_size(num_min, num_max, compare):
    """
    裁剪核心代码，用于判断边界
    :param num_min: 最小值
    :param num_max: 最大值
    :param compare: 比较值
    :return: 差值与比较值相同的 最大最小值
    """
    delta_num = num_max - num_min
    delta_compare = compare - delta_num

    num_min_new = num_min
    num_max_new = num_max
    if delta_compare > 0:
        num_min_new = num_min - (delta_compare // 2)
        num_max_new = num_max + (delta_compare // 2)

    delta_new = num_max_new - num_min_new
    delta_new_compare = compare - delta_new
    if delta_new_compare > 0:
        num_max_new = num_max_new + delta_new_compare
    else:
        num_max_new = num_max_new - delta_new_compare

    return num_min_new, num_max_new


def CropTeeth(label_nd, patch_size):
    """
    裁剪核心代码
    :param label_nd:输入的标签 ndarray
    :param patch_size:  输入的patch_size tuple
    :return: x_min_new, x_max_new, y_min_new, y_max_new, z_min_new, z_max_new
    """
    label_one = np.where(label_nd == 1)
    label_x, label_y, label_z = label_one[0], label_one[1], label_one[2]
    label_x_array = np.array(label_x)
    label_y_array = np.array(label_y)
    label_z_array = np.array(label_z)
    x_min, x_max = label_x_array.min(), label_x_array.max()
    y_min, y_max = label_y_array.min(), label_y_array.max()
    z_min, z_max = label_z_array.min(), label_z_array.max()

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    delta_z = z_max - z_min

    print(f'minus(label_x): {delta_x}, min:{x_min}, max:{x_max}')
    print(f'minus(label_y): {delta_y}, min:{y_min}, max:{y_max}')
    print(f'minus(label_z): {delta_z}, min:{z_min}, max:{z_max}')

    x_min_new, x_max_new = fix_size(x_min, x_max, patch_size[0])
    y_min_new, y_max_new = fix_size(y_min, y_max, patch_size[1])
    z_min_new, z_max_new = fix_size(z_min, z_max, patch_size[2])

    return x_min_new, x_max_new, y_min_new, y_max_new, z_min_new, z_max_new


def dicom2half():
    """
    将dicom 从 (667, 667, 666)裁剪到牙齿附近的patch size
    :return:
    """
    reader = sitk.ImageSeriesReader()

    folderPath = r'F:\pythonProject\Datasets\RT_data\CBCT-data\CT'  # dicom图片所在文件夹
    labelPath = r'F:\pythonProject\Datasets\RT_data\CBCT-data\Label'  # dicom标签图片所在文件夹
    image_outputPath = r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI\image'
    label_outputPath = r'F:\pythonProject\Datasets\ForROI\RT_CBCT_ROI\label'

    make_dir(image_outputPath)
    make_dir(label_outputPath)

    folder_list = os.listdir(folderPath)
    label_list = os.listdir((labelPath))

    patch_size = (256, 220, 220)

    for idx, folder in enumerate(folder_list):

        # 只读取二值化的标签 如Binary_Labelxxxx.nrrd
        # 多标签的为 Labelxxxx.nrrd
        if len(label_list[idx].split('_')) == 1:
            continue

        print(f'第 {idx} 开始处理')

        # 1、获取image和label的文件名字
        folder_path = os.path.join(folderPath, folder)
        label_path = os.path.join(labelPath, label_list[idx])
        print(f'folder_path:{folder_path}, label_path:{label_path}')

        # 2、读取dicom 文件和 label 文件
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
        label_numpy, label_header = nrrd.read(label_path)  # x, y, z
        # print(f'dicom_names:{dicom_names}')

        x_min, x_max, y_min, y_max, z_min, z_max = CropTeeth(label_nd=label_numpy, patch_size=patch_size)
        print("x:[%d, %d], y:[%d, %d], z:[%d, %d]" % (x_min, x_max, y_min, y_max, z_min, z_max))
        # print("x_delta:%d, y_delta:%d, z_delta:%d" % (-x_min+x_max, -y_min+y_max, -z_min+z_max))
        # 3、转为 ndarray 类型
        reader.SetFileNames(dicom_names)
        dicom = reader.Execute()

        # 3.1、 进行转置，从 (z, y, x) to (x, y, z)
        image = sitk.GetArrayFromImage(dicom)  # z, y, x
        flip_image = image.transpose((2, 1, 0))  # (z, y, x) to (x, y, z)

        print_type(image, 'image')
        print_type(flip_image, 'flip_image')
        print_type(label_numpy, 'label_numpy')

        # 3.2、 进行裁切

        flip_crop_image = flip_image[x_min:x_max, y_min:y_max, z_min:z_max]
        crop_label = label_numpy[x_min:x_max, y_min:y_max, z_min:z_max]
        print_type(crop_label, 'crop_label')
        print_type(flip_crop_image, 'flip_crop_image')

        # 4、 从ndarray 转为 .nii.gz格式
        new_image_object = nib.Nifti1Image(flip_crop_image, np.eye(4))
        new_label_object = nib.Nifti1Image(crop_label, np.eye(4))

        # 5、定义输出名字
        image_output_path = os.path.join(image_outputPath, folder+'.nii.gz')
        label_nii_name = os.path.join(label_outputPath, "CBCT"+label_list[idx].split('.')[0].split('l')[1]+".nii.gz")

        # 6、输出为 .nii.gz
        print(f'image_output_path:{image_output_path}, label_nii_name:{label_nii_name}')
        nib.save(new_image_object, image_output_path)
        nib.save(new_label_object, label_nii_name)

        # # 7、进行可视化
        # OrthoSlicer3D(new_image_object.dataobj).show()
        # OrthoSlicer3D(new_label_object.dataobj).show()

        # break


if __name__ == '__main__':
    # dicom2nii()
    # nrrd2nii()
    dicom2half()
