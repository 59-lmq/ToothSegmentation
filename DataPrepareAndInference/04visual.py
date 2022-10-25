import numpy as np  # 转换格式
import nibabel as nib  # 读取数据
from nibabel.viewers import OrthoSlicer3D  # 可视化
import h5py
import matplotlib
matplotlib.use('TkAgg')  # 用于滚动查看nii.gz
import os
import nrrd

import SimpleITK as sitk


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


if __name__ == '__main__':
    # dicom2nii()
    nrrd2nii()
