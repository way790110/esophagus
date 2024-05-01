import pathlib

import cv2
import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def draw_contour(img, mask, color, thickness=1):
    img_withContour = np.array(img.copy(), dtype=np.uint8)

    if len(img_withContour.shape)==2:
        img_withContour = cv2.cvtColor(img_withContour, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_withContour, contours, -1, color, thickness)
    return img_withContour


def window_converter(image, winLevel, winWidth):
    '''
    Convert 16bits Dicom image into 8bits image by window level(WL) and window width(WW)

    Args:  
        image: 16 bits 2D gray scale image
        winLevel: WL 
        winWidth: WW
    Return: 
        image: 8 bits(0-255) image
    '''
    image = np.array(image, dtype = np.float64)
    yMin = (winLevel - 0.5 * winWidth)
    yMax = (winLevel + 0.5 * winWidth)
    image = np.where(image>=yMax, 255, 0) + np.where((image>yMin) & (image<yMax), ((image-yMin)/(yMax-yMin)) * 255, 0)
    return image


def save_h5_several_header(file_path, header_list, data_list):
    hf = h5py.File(file_path, 'w')
    for header, data in zip(header_list, data_list):
        hf.create_dataset(header, data=data)


def get_filename(path):
    while path.suffix in ['.nii', '.gz']:
        path = path.with_suffix('')
    return path.stem


def load_nii_to_numpy(file_path):
    vol = np.array(nib.load(file_path).dataobj)
    vol = vol.transpose(2, 1, 0)
    return np.rot90(vol, 2)


def main():
    root = r'dataset\FLARE_2022\Training\FLARE22_LabeledCase50'
    root = pathlib.Path(root)
    image_root = root / 'images'
    label_root = root / 'labels'

    count = 0

    # setting path 
    for image_path in image_root.glob("*"):
        file_name = get_filename(image_path)
        label_name = file_name[:-5] + '.nii.gz'
        label_path = label_root / label_name
        save_path = pathlib.Path(*[r'dataset\h5\FLARE22', file_name[:-5] + '.h5'])
        save_path.parent.mkdir(exist_ok=True, parents=True)

        # loading image & mask
        image = load_nii_to_numpy(image_path) 
        mask = load_nii_to_numpy(label_path) 
        mask_esophagus = np.where(mask == 10, 1, 0)

        # choose esophagus image
        idx = np.where(np.sum(mask_esophagus, axis=(1, 2)) != 0)
        image = image[idx] 
        mask_esophagus = mask_esophagus[idx]

        # save to h5
        save_h5_several_header(save_path, ['image', 'mask'], [image, mask_esophagus])

        # for show
        image = window_converter(image, 35, 350).astype(np.uint8)
        mask_esophagus = mask_esophagus.astype(np.uint8)
        for i in range(len(image)):
            fusion = draw_contour(image[i], mask_esophagus[i], (0 , 0, 255), 1)
            save_path = pathlib.Path(*[r'dataset\png\FLARE22', file_name[:-5], f'{i}.png'])
            save_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(save_path), fusion)

            


    print(count)

if __name__ == '__main__':
    main()