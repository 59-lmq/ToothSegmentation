import numpy as np  # 转换格式
import nibabel as nib  # 读取数据
from nibabel.viewers import OrthoSlicer3D  # 可视化
import h5py
import matplotlib
matplotlib.use('TkAgg')  # 用于滚动查看nii.gz

"""
image_path = r"../image/Teeth_0001_0000.nii.gz"
label_path = r"../label/Teeth_0001_0000.nii.gz"

output_shape = [128, 128, 128]

# 1、读取数据
image_object = nib.load(image_path)
print(f'type(image_object):{type(image_object)}')
label_object = nib.load(label_path)
print(f'type(label_object):{type(label_object)}')

image_numpy_data = image_object.get_fdata()
print('image.shape: ', image_numpy_data.shape)  # (266, 266, 200)
print(f'image value range: [{image_numpy_data.min()}, {image_numpy_data.max()}]')  # [-1000.0, 3095.0]

label_numpy_data = label_object.get_fdata()
print('label.shape: ', label_numpy_data.shape)  # (266, 266, 200)
print(f'label value range: [{label_numpy_data.min()}, {label_numpy_data.max()}]')  # [-1000.0, 3095.0]

# 2、归一化
image_min, image_max = image_numpy_data.min(), image_numpy_data.max()
image_normalize_data = (image_numpy_data - image_min) / (image_max - image_min)
image_normalize_data2 = (image_numpy_data - np.mean(image_numpy_data)) / (np.std(image_numpy_data))

print('norm_image.shape: ', image_normalize_data.shape)
print(f'norm_image value range: [{image_normalize_data.min()}, {image_normalize_data.max()}]')

print('norm_image.shape: ', image_normalize_data2.shape)
print(f'norm_image2 value range: [{image_normalize_data2.min()}, {image_normalize_data2.max()}]')

# 3、裁剪
depth, height, width = image_numpy_data.shape
crop_d = np.random.randint(0, depth-output_shape[0])
crop_h = np.random.randint(0, height-output_shape[1])
crop_w = np.random.randint(0, width-output_shape[2])
print(f'random (d, w, h): ({crop_d, crop_h, crop_w})')

image_crop = image_normalize_data2[crop_d:crop_d+output_shape[0],
                                   crop_h:crop_h+output_shape[1],
                                   crop_w:crop_w+output_shape[2]]

print('image_crop.shape: ', image_crop.shape)
print(f'image_crop value range: [{image_crop.min()}, {image_crop.max()}]')

# 4、可视化
new_image_object = nib.Nifti1Image(image_crop, np.eye(4))
# OrthoSlicer3D(new_image_object.dataobj).show()

print(f'type(new_image_object):{type(new_image_object)}')

# 5、转成 (1, 128, 128, 128)
image_transpose = image_crop[np.newaxis, :, :, :]
print('image_transpose.shape: ', image_transpose.shape)

"""


def GainNorm(data):
    norm_data = (data - np.mean(data)) / (np.std(data))
    return norm_data


def RandomCropNumpy(x_data, y_data, out_shape):
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


def GainData(x_path, y_path, out_shape):
    # 0、读入nii.gz文件
    x = nib.load(x_path)
    y = nib.load(y_path)

    # 1、转成numpy.ndarray格式
    x_data = x.get_fdata()
    y_data = y.get_fdata()

    # 2、归一化
    x_norm = GainNorm(x_data)
    y_norm = GainNorm(y_data)

    # 3、随机裁剪至(128, 128, 128)
    x_crop, y_crop = RandomCropNumpy(x_norm, y_norm, out_shape)

    # 4、增加维度到(1, 128, 128)
    x_transpose = x_crop[np.newaxis, :, :, :]
    y_transpose = y_crop[np.newaxis, :, :, :]

    # 5、保存为.h5文件

    f = h5py.File('./test.h5', 'w')
    f.create_dataset('image', data=x_transpose, compression='gzip')
    f.create_dataset('label', data=y_transpose, compression='gzip')
    f.close()


def loadh5():
    # 读取h5文件
    file_path = r'E:\Jupter\ctooth\TrainData\nii_train_h5\CBCT2207156_norm.h5'
    h5f = h5py.File(file_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    print(type(image), type(label))
    print(image.shape, label.shape)


if __name__ == '__main__':
    # image_path = r"../image/Teeth_0001_0000.nii.gz"
    # label_path = r"../label/Teeth_0001_0000.nii.gz"
    #
    # output_shape = [128, 128, 128]
    #
    # GainData(image_path, label_path, output_shape)

    loadh5()



