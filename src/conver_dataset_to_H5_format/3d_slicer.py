import copy
import glob
import os
from os.path import isfile
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pydicom
from numba import jit
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydicom.misc import is_dicom

import GEcolormap as GEcolormap


def save_h5_several_header(file_path, header_list, data_list):
    hf = h5py.File(file_path, 'w')
    for header, data in zip(header_list, data_list):
        hf.create_dataset(header, data=data)


def is_axial_CT(ds:pydicom.FileDataset):
    """
    Filter coronal/sagittal and localizer
    """
    modality = ds['Modality'].value
    imageType = ds['ImageType'].value
    imageOrientationPatient = ds[0x20, 0x37].value

    is_CT  = modality=='CT' 
    try:
        valid_image_type = (imageType[2]=='AXIAL' or imageType[2]=='REFORMATTED' or imageType[2]=='MPR')
    except IndexError:
        return False
    is_axial = imageOrientationPatient==[1,0,0,0,1,0] # add REFORMATTED for dual energy CT

    return is_CT and valid_image_type and is_axial


def is_image_dicom(ds:pydicom.FileDataset):
    """
    Assume a legal dicom image has Pixel Data and RescaleIntercept attribute.
    Filter for report file and missing Pixel Data
    """
    return ('SeriesNumber' in ds) and ('InstanceNumber' in ds) and \
            ('RescaleIntercept' in ds) and ([0x7fe0, 0x10] in ds)


def PETCT_header_check(ds):
    '''
    Check PETCT image is valid and is PET or CT
    '''
    try:
        series_num = ds['SeriesNumber'].value
        instance_num = ds['InstanceNumber'].value
        modality = ds['Modality'].value
        imageType = ds['ImageType'].value
        imageOrientationPatient = ds[0x20, 0x37].value
    except KeyError:
        return False

    try:
        seriesDes = ds['SeriesDescription'].value
    except KeyError:
        seriesDes = ""

    # Check attributes according to modality
    if ('RescaleIntercept' in ds) and (imageOrientationPatient==[1,0,0,0,1,0]) and ([0x7fe0, 0x10] in ds): # filter for report file
        if modality=='CT':
            convolutionKernel = ds[0x18, 0x1210].value
            if imageType[2]!='AXIAL': # filter for localizer
                return False
            if convolutionKernel=="Q AC": # filter for AC image
                return False
            if 'ctac' in seriesDes.lower(): # No CTAC CT image
                return False
            # About ReconstructionTargetCenterPatient ref following link
            # Skip 2 center not consist cuz image will be slightly not aligned like RadiAnt
            # https://dicom.nema.org/medical/Dicom/2016e/output/chtml/part03/sect_C.8.15.3.6.html
            if "ReconstructionTargetCenterPatient" in ds: 
                reconstruct_center = ds["ReconstructionTargetCenterPatient"].value
                data_collect_center = ds["DataCollectionCenterPatient"].value
                if reconstruct_center != data_collect_center:
                    return False
            return True

        elif modality=='PT':
            try:
                units = ds[0x54, 0x1001].value
                manufacturer = ds[0x08, 0x70].value
                attenuationCorrectionMethod = ds[0x54, 0x1101].value
                radiopharma_info = ds[0x54, 0x16][0]
            except KeyError:
                return False

            if units!="BQML" and units!="CNTS":
                return False
            if radiopharma_info.RadiopharmaceuticalStartTime==None or radiopharma_info.RadionuclideTotalDose==None or radiopharma_info.RadionuclideHalfLife==None:
                return False
            return True
        else: 
            return False
    else:
        return False



def group_and_sort_by_attributes(data, *args):
    """Sort and group data by attributes in args.

    Args:
        data (object | dict): objects to be sorted and grouped.
        args: attributes as sorting keys. Last attributes is index in group. 
            Attributes except last one consist group ID.

    Returns:
        list[object | dict]: objects splited into group list.
    """
    # Pack data with group tag and id.
    attrs_and_data = []
    for data_element in data:
        attributes = []
        for attr_name in args:
            attr = data_element.get(attr_name)
            attributes.append(attr)

        row_data = {
            "tag": "".join([str(i) for i in attributes[:-1]]),
            "id" :  attributes[-1],
            "data": data_element
        }
        attrs_and_data.append(row_data)

    # Group data by tag
    group_data = {}
    for aad in attrs_and_data:
        if aad["tag"] not in group_data:
            group_data[aad["tag"]] = [aad]
        else:
            group_data[aad["tag"]].append(aad)

    # Sort by last attribute and unpack data
    group_data = dict(sorted(group_data.items()))
    for tag in group_data:
        sorted_data = sorted(group_data[tag], key=lambda k:k["id"])
        group_data[tag] = [d["data"] for d in sorted_data]

    return list(group_data.values())


def CT_dirpacker_v2(dirpath):
    dicom_files = [
        f for f in glob.glob(os.path.join(dirpath, "**/*"),recursive=True) if isfile(f)
    ] # change for no extension name dicom
    images_in_study = []
    series_vols_in_study = []
    series_headers_in_study = []

    # Gather slice files of the series
    for dcmf in dicom_files:
        try:
            ds = dcmread(dcmf)
            is_axial = is_axial_CT(ds)
            is_image = is_image_dicom(ds)
        except (InvalidDicomError, KeyError) as _:
            #print("Invalid dicom file: {}, skipped.".format(dcmf))
            continue

        if is_image and is_axial:
            images_in_study.append(ds)
        else:
            continue

    # Split images into series
    series_in_study = group_and_sort_by_attributes(
        images_in_study, "SeriesNumber", "InstanceNumber"
    )

    # Build series volume and headers
    for series_ds in series_in_study:
        series_vol = []
        series_headers = []
        for ds in series_ds:
            series_vol.append(ds.pixel_array)
            ds = dcmread(ds.filename, stop_before_pixels=True)
            series_headers.append(ds)

        if len(series_headers) < 2:
            print(f"Series {series_headers[0].SeriesNumber} slices number shouldn't be one, skip")
            continue

        if series_headers[0][0x20,0x32].value[2] < series_headers[1][0x20,0x32].value[2]:
            series_vol = np.flip(series_vol, 0)
            series_headers.reverse()

        series_vols_in_study.append(np.array(series_vol))
        series_headers_in_study.append(series_headers)

    if len(series_headers_in_study) == 0:
        print("ERROR: No valid series found in CT study.")
        raise AttributeError

    return series_vols_in_study, series_headers_in_study


def get_header_and_vol(ds_list):
    img = []
    header = []
    for ds in ds_list:
        img.append(ds.pixel_array)
        del ds.PixelData
        header.append(ds)
    return header, np.array(img)


def PETCT_dirpacker(dicom_files):
    '''
    Load PET-CT dicom image from files list, filter the CT and PET image apart, 
    and return PETCT series volumes and headers set(s) in a list.
    Sort images and headers by header tag 'Instance Number'.
    Reverse image slices order if image is not in head-to-foot direction.

    Args: 
        List of dicom files path
    Return: 
        List[(CT volume, CT header list, PET volume, PET header list)]
    '''        
    # Gather slice files of the series
    study_data = {}
    for dcm_file in dicom_files:
        # Open file
        try:
            ds = dcmread(dcm_file)
        except InvalidDicomError:
            continue
        
        # Check required tags and other attributes
        if not PETCT_header_check(ds):
            continue

        # Append FileDataset to series data list
        series_num = int(ds['SeriesNumber'].value)
        if series_num in study_data:
            study_data[series_num].append(ds)
        else:
            study_data[series_num] = [ds]
        
     # Skip one-slice series
    for series_num, series_data in study_data.items():
        if len(series_data) < 2:
            print(f"WARNING: series {series_num} slices too less: {len(series_data)}")
            del study_data[series_num]

    # Remove duplicate instance
    for series_num, series_data in study_data.items():
        study_data[series_num] = remove_duplicate_instance(study_data[series_num])

    # Split PET and CT series
    PET_data = []
    CT_data = []
    for series_num, series_data in study_data.items():
        modality = series_data[0].Modality
        if modality == "CT":
            CT_data.append(series_data)
        elif modality == "PT":
            PET_data.append(series_data)
        else:
            continue
    
    # Check PET and CT series count, and keep one CT series
    if len(PET_data) == 0:
        print("ERROR: No valid PET series in study")
        raise Exception

    if len(CT_data) == 0:
        print("ERROR: No valid CT series in study")
        raise Exception
    elif len(CT_data) > 1: # Take a CT series with most instances
        series_len = [len(d) for d in CT_data]
        CT_data = CT_data[np.argmax(series_len)]
    else:
        CT_data = CT_data[0]

    # Sort all series by instanceNumber, and flip to head to foot order
    CT_data = sorted(CT_data, key=lambda d: d.InstanceNumber)
    if CT_data[0][0x20,0x32].value[2] < CT_data[1][0x20,0x32].value[2]:
        CT_data.reverse()

    for i, _ in enumerate(PET_data):
        PET_data[i] = sorted(PET_data[i], key=lambda d: d.InstanceNumber)
        if PET_data[i][0][0x20,0x32].value[2] < PET_data[i][1][0x20,0x32].value[2]:
            PET_data[i].reverse()

    # return all set of PET-CT volumes and headers
    CT_header, CT_vol = get_header_and_vol(CT_data)
    return_data = []
    for _, pet_data in enumerate(PET_data):
        PET_header, PET_vol = get_header_and_vol(pet_data)
        return_data.append((CT_vol, CT_header, PET_vol, PET_header))

    return return_data


def remove_duplicate_instance(ds_list):
    unique_instanceNum = []
    unique_instance = []
    for ds in ds_list:
        if ds.InstanceNumber not in unique_instanceNum:
            unique_instanceNum.append(ds.InstanceNumber)
            unique_instance.append(ds)
    return unique_instance


def draw_contour(img, mask, color_name, thickness=1):
    img_withContour = np.array(img.copy(), dtype=np.uint8)
    if len(img_withContour.shape)==2:
        img_withContour = gray_to_RGB(img_withContour)

    color = get_color(color_name)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_withContour, contours, -1, color, thickness)
    return img_withContour


def get_color(color_name, isBGR=False):
    if color_name=='r' or color_name=='red':
        color =  (255, 0, 0)
    elif color_name=='g' or color_name=='green':
        color =  (0, 255, 0)
    elif color_name=='b' or color_name=='blue':
        color =  (0, 0, 255)
    else:
        return False

    if isBGR:
        return tuple([color[2], color[1], color[0]])
    else:
        return color


def gray_to_RGB(img, colormap=None, rescale=None):
    '''
    Convert 1-channel grayscale image to 3-channel colored image.
    Integrates 'rescaling to 0-255' and 'applyColorMap / merge to 3-channels grayscale'.
    Args:
        rescale: set tuple (min, max) to linear rescale between given range,
            set True for auto rescale between max and min value in image,
            set False or not set to not rescale
    Return:
        3-channels numpy array
    '''
    # Rescale image
    if type(rescale) == tuple:
        assert len(rescale) == 2 and (rescale[1]> rescale[0])
        min_bound = rescale[0]
        max_bound = rescale[1]
        img = np.array((img.copy() - min_bound) / (max_bound - min_bound) * 255, np.uint8)
    elif rescale == True:
        img = np.array((img.copy() - np.min(img)) / (np.max(img) - np.min(img)) * 255, np.uint8)
    elif (rescale == False) or (rescale is None):
        pass
    else:
        print(f"ERROR: invalid rescale setting {rescale}")
    
    # Convert to 3-channels
    if isinstance(colormap, (np.ndarray, list)):
        img_3channel = cv2.applyColorMap(img, colormap)
    else:
        img_3channel = cv2.merge((img,img,img))
            
    return img_3channel


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


def fill_padding_value(vol):
    """
    TODO: Should padding by dicom header attribute
    """
    for i in range(len(vol)):
        vol[i] = np.where(vol[i]==-2000, 0, vol[i])   
    return vol


def dicom_rescale(vol, header_list):
    if header_list[0].Modality == 'PT':
        vol_temp = np.zeros(np.shape(vol), dtype=np.float32)
        for i in range(len(vol)):
            header = header_list[i]
            vol_temp[i] = PET_unit_conversion(vol[i], header)
        return vol_temp
            
    elif header_list[0].Modality == 'CT':
        vol = np.array(vol).astype(np.int16)
        for i in range(len(vol)):
            header = header_list[i]
            if header:
                Slope = header.RescaleSlope
                Intercept = header.RescaleIntercept
                vol[i] = vol[i] * int(Slope) + int(Intercept)
    else:
        print("============UNKNOWN MODALITY DETECTED================")

    return vol


def PET_unit_conversion(PET_input, header):
    '''
    Arg:
        PET_input: PET value to be conversion, could be value or np.array
        header: PET header in pydicom format
    '''
    SeriesTime = _dicom_time_convert(header.SeriesTime)
    PatientWeight = float(header.PatientWeight)
    Slope = float(header.RescaleSlope)
    Intercept = float(header.RescaleIntercept)
    RadiopharmaceuticalStartTime = _dicom_time_convert(header.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
    RadionuclideTotalDose = float(header.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    RadionuclideHalfLife = float(header.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
    decay = 2**(-(SeriesTime - RadiopharmaceuticalStartTime) / RadionuclideHalfLife)
    if header.Units=='BQML':
        PET_output = (PET_input * Slope + Intercept) * (PatientWeight*1000) / (RadionuclideTotalDose * decay)
    elif header.Units=='CNTS' and header[0x7053, 0x10].value=='Philips PET Private Group':
        PET_output = (PET_input * Slope + Intercept) * (PatientWeight*1000) / (RadionuclideTotalDose * decay) * float(header[0x7053,0x1009].value)
    else:
        print("Unknown PET unit or group detected, Abort.")
        raise BaseException

    return PET_output


def _dicom_time_convert(time):
    time = str(time)
    hour = float(time[0:2])
    min = float(time[2:4])
    sec = float(time[4:6])
    return hour*3600+min*60+sec


def get_modalities(dicom_files):
    modalities = []
    for dcm_file in dicom_files:
        ds = dcmread(dcm_file)
        modalities.append(ds.Modality)
    return modalities


def PET_resize_and_compress(PET_vol_modified, PET_header_modified, PET_header, CT_vol_modfied):
    '''
    Resize PET_modified slice image size to fit CT slice image size
    CHANGE PET VOL FILE TYPE TO np.float16
    '''
    height, width = CT_vol_modfied.shape[1:]
    PET_vol_modified_temp = np.zeros(((len(PET_vol_modified),) + (height, width)), dtype=np.float32)
    for idx in range(len(PET_vol_modified)):
        img = PET_vol_modified[idx, :, :]
        img_sm = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        PET_vol_modified_temp[idx, :, :] = img_sm
        PET_header_modified[idx][0x28,0x30].value = [PET_header[idx][0x28,0x30].value[0] * PET_header[idx][0x28,0x10].value / height, PET_header[idx][0x28,0x30].value[1] * PET_header[idx][0x28,0x11].value / width]
        PET_header_modified[idx][0x28,0x10].value = height
        PET_header_modified[idx][0x28,0x11].value = width
    return  PET_vol_modified_temp, PET_header_modified


def zAxis_alignment(vol_main, header_main, vol_source, header_source):
    """
    Modify vol 2 as vol 1 length and z axis range,
    Temporary using floor sampling 

    Args:
        vol_main:      image volume
        header_main:   list of pydicom.dataset.FileDatasety
        vol_source:      image volume
        header_source:   list of pydicom.dataset.FileDataset

    Return:

        vol_main:      image volume
        header_main:   list of pydicom.dataset.FileDataset
        vol_source:      image volume
        header_source:   list of pydicom.dataset.FileDataset
    """
    vol_main = np.array(vol_main)
    vol_source = np.array(vol_source)
    slice_thickness_main = header_main[0][0x18,0x50].value
    z_pos_source = [header_source[i][0x20,0x32].value[2] for i in range(len(header_source))] # z position from large to small of source image 

    vol_modified_source = np.zeros((vol_main.shape[0],) + vol_source.shape[1:], dtype=type(vol_source[0][0][0]))
    header_modified_source = []

    for i in range(len(header_main)):
        z_pos_main  = header_main[i][0x20,0x32].value[2]
        if (z_pos_main + 0.5*slice_thickness_main) >= z_pos_source[-1] and (z_pos_main - 0.5*slice_thickness_main) <= z_pos_source[0]:
            nearest_source_img_idx = min(enumerate(z_pos_source), key=lambda x: abs(x[1]-z_pos_main))[0]
            vol_modified_source[i] = vol_source[nearest_source_img_idx]
            header_modified_source.append(header_source[nearest_source_img_idx])
        else:
            nearest_source_img_idx = min(enumerate(z_pos_source), key=lambda x: abs(x[1]-z_pos_main))[0]
            nearest_header = copy.deepcopy(header_source[nearest_source_img_idx])
            header_modified_source.append(nearest_header)

    return vol_main, header_main, vol_modified_source, header_modified_source


@jit
def _fovcrop(vol, new_vol, rows, boundary_shift, fov_ratio):
    for z in range(len(vol)): # z axis
            for i in range(0, rows, 1):
                for j in range(0, rows, 1):
                    new_vol[z][i][j] = vol[z][int(np.floor(boundary_shift+i*fov_ratio))][int(np.floor(boundary_shift+j*fov_ratio))]
    return new_vol

@jit
def _fovpad(vol, new_vol, rows, boundary_shift, fov_ratio):
    for z in range(len(vol)): # z axis
        for i in range(boundary_shift, rows-boundary_shift, 1):
            for j in range(boundary_shift, rows-boundary_shift, 1):
                new_vol[z][i][j] = vol[z][int(np.floor((i-boundary_shift)/fov_ratio))][int(np.floor((j-boundary_shift)/fov_ratio))]
    return new_vol


def CT2PET_coordinate_transform(CT_vol, CT_header_list, PET_vol, PET_header_list, padding_value=0):
    """
    Transform CT to fit PET view coordinate

    This function transform CT coordinate into PET coordinate. The transformation is base on coordinate data in dicom headers.
    WARNNING: This module assert 
        1. All image orientation is 1/0/0/0/1/0
        2. Slice image shape is square(n*n)
        3. Pixel spacing has same value in x and y direction
        4. assume PET and CT has the same centrel point

    Args:
        CT_vol: CT dicom image volume
        CT_header_list: CT volume slices' header files
        PET_vol: PET dicom image volume
        PET_header_list: PET volume slices' header files

    Return:
        CT_vol: CT image volume which is aligned by PET coordinate
        CT_header_list: CT volume slices' header files
        PET_vol: PET image volume
        PET_header_list: PET volume slices' header files
        shift_spacing: the image origin point shift value from origin image to modded img, >0 for cropping, <0 for padding
    """
    # Load data
    # [0x20,0x37]: image orientation
    # [0x28,0x10]: rows
    # [0x28,0x11]: columns

    for CT_header in CT_header_list:
        assert [1,0,0,0,1,0] == CT_header[0x20,0x37].value, print(
            "ERROR: This module assert all image orientation is 1/0/0/0/1/0."
        )
        assert CT_header[0x28,0x10].value == CT_header[0x28,0x11].value, print(
            "ERROR: This module assert slice images shape are square."
        )
    for PET_header in PET_header_list:
        assert [1,0,0,0,1,0] == PET_header[0x20,0x37].value, print(
            "ERROR: This module assert all image orientation is 1/0/0/0/1/0."
        )
        assert PET_header[0x28,0x10].value == PET_header[0x28,0x11].value, print(
            "ERROR: This module assert slice images shape are square."
        )

    pixelSpacing_CT = CT_header_list[0][0x28,0x30].value[0]
    pixelSpacing_PET = PET_header_list[0][0x28,0x30].value[0]
    rows_CT = CT_header_list[0][0x28,0x10].value
    cols_CT = CT_header_list[0][0x28,0x11].value
    rows_PET = PET_header_list[0][0x28,0x10].value

    # Fov(field of view) comparison
    fov_CT = pixelSpacing_CT * rows_CT
    fov_PET = pixelSpacing_PET * rows_PET

    # Count the shift between CT and PET square in PET coordinate
    PET_vol, PET_header_list, CT_vol, CT_header_list = zAxis_alignment(
        PET_vol, PET_header_list, CT_vol, CT_header_list
    )
    # Copy a new object pointer as intention
    new_PET_header = copy.deepcopy(PET_header_list)

    if fov_PET < fov_CT: # Crop CT
        fov_ratio = fov_PET / fov_CT
        # Volume of PET and CT
        new_CT_vol = np.zeros_like(CT_vol)
        if padding_value:
            new_CT_vol.fill(padding_value)
        new_CT_header = copy.deepcopy(CT_header_list)
        boundary_shift = (rows_CT-rows_CT * fov_ratio) / 2 # Shift value in pixel
        boundary_shift_mm = boundary_shift * pixelSpacing_CT # Real world shift distance on origin img, take origin pixelspacing

        new_CT_vol = _fovcrop(CT_vol, new_CT_vol, rows_CT, boundary_shift, fov_ratio)
        for CT_header in new_CT_header:
            if CT_header:
                CT_header['PixelSpacing'].value = [pixelSpacing_CT*fov_ratio, pixelSpacing_CT*fov_ratio]
                CT_position_x, CT_position_y, CT_position_z = CT_header[0x20,0x32].value
                CT_header[0x20,0x32].value = [CT_position_x + boundary_shift_mm, CT_position_y + boundary_shift_mm, CT_position_z]

        return new_CT_vol, new_CT_header, PET_vol, new_PET_header, boundary_shift_mm

    elif fov_PET > fov_CT: # Pad CT
        # Volume of PET and CT
        fov_ratio = fov_CT / fov_PET
        new_CT_vol = np.zeros_like(CT_vol)
        if padding_value:
            new_CT_vol.fill(padding_value)
        new_CT_header = copy.deepcopy(CT_header_list)
        boundary_shift = int(np.ceil((rows_CT - rows_CT * fov_ratio) / 2)) # Shift value in pixel
        boundary_shift_mm = -1 * boundary_shift * pixelSpacing_CT / fov_ratio # Negative shift value for padding. Real world shift distance on new img, take new pixelspacing
        
        new_CT_vol = _fovpad(CT_vol, new_CT_vol, rows_CT, boundary_shift, fov_ratio)

        for CT_header in new_CT_header:
            if CT_header:
                CT_header['PixelSpacing'].value = [pixelSpacing_CT/fov_ratio, pixelSpacing_CT/fov_ratio]
                CT_position_x, CT_position_y, CT_position_z  = CT_header[0x20,0x32].value
                CT_header[0x20,0x32].value = [CT_position_x + boundary_shift_mm, CT_position_y + boundary_shift_mm, CT_position_z]

        return new_CT_vol, new_CT_header, PET_vol, new_PET_header, boundary_shift_mm 

    else:
        return CT_vol, CT_header_list, PET_vol, new_PET_header, float(0)


def grayscale_merge(background_img, layer_img, colormap, layer_rescaled=False):
    if layer_img.shape != background_img.shape:
        layer_img = cv2.resize(layer_img, (background_img.shape[1], background_img.shape[0]))
    background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)
    
    layer_img = gray_to_RGB(layer_img, colormap, layer_rescaled)
    fusioned_img = cv2.addWeighted(background_img, 0.5, layer_img, 0.5, 0)
    return fusioned_img


def main():
    # target_root = Path("/mnt/sda1/Project_data/ESO/20240412_ESO_3DSlicer_mask/1171231")
    target_root = Path(r'dataset\20240412_ESO_3DSlicer_mask\769254')
    output_path = target_root.joinpath("mask_fusion_sample")
    output_path.mkdir(parents=True, exist_ok=True)

    # save_h5
    save_to_h5 = True
    if save_to_h5:
        h5_root = target_root.parent / 'h5' / f'{target_root.stem}.h5'
        h5_root.parent.mkdir(exist_ok=True, parents=True)

    # Load mask file
    file_path = target_root / "mask" / "Segmentation.seg.nrrd"
    mask, option = nrrd.read(file_path)
    mask = np.transpose(mask, (2, 1, 0)) # XYZ to ZYX
    mask = np.flip(mask, axis=0)

    # Load dicom image
    dicom_folder = target_root.joinpath("data")
    dicom_files = [f for f in Path(dicom_folder).rglob("*") if is_dicom(f)]

    modalities = get_modalities(dicom_files)
    if "PT" in modalities:
        CT_vol, CT_header, PET_vol, PET_header = PETCT_dirpacker(dicom_files)[0]
        if save_to_h5:
            save_h5_several_header(h5_root, ['image', 'mask'], [CT_vol, mask])
            exit()
        assert mask.shape[0] == CT_vol.shape[0]

        CT_vol = fill_padding_value(CT_vol)
        CT_vol_modfied, CT_header_modified, PET_vol_modified, PET_header_modified, shiftspacing = CT2PET_coordinate_transform(CT_vol, CT_header, PET_vol, PET_header)
        mask_vol_modfied, _, _, _, _ = CT2PET_coordinate_transform(mask, CT_header, PET_vol, PET_header)

        CT_vol_modfied = dicom_rescale(CT_vol_modfied, CT_header_modified)
        PET_vol_modified = dicom_rescale(PET_vol_modified, PET_header_modified)
        PET_vol_modified, PET_header_modified = PET_resize_and_compress(PET_vol_modified, PET_header_modified, PET_header, CT_vol_modfied)

        for idx, (CT_slice, PET_slice, mask_slice) in enumerate(zip(CT_vol_modfied, PET_vol_modified, mask_vol_modfied)):
            CT_slice = window_converter(CT_slice, 35, 350)
            PET_slice = window_converter(PET_slice, 2.5, 5)
            PETCT_GEcolor = grayscale_merge(
                CT_slice.astype(np.uint8),
                PET_slice.astype(np.uint8),
                GEcolormap.GE_color_opencv()
            )
            contoured_img = draw_contour(PETCT_GEcolor, mask_slice, 'r', thickness=1)
            contoured_img = np.flip(contoured_img, axis=-1) # RGB2BGR for opencv
            cv2.imwrite(str(output_path.joinpath(f"{idx}.png")), contoured_img)

    else:
        series_vols, series_headers = CT_dirpacker_v2(dicom_folder)
        if save_to_h5:
            save_h5_several_header(h5_root, ['image', 'mask'], [series_vols, mask])
            exit() 
        img_vol = series_vols[0]
        img_header = series_headers[0]

        img_vol = fill_padding_value(img_vol)
        img_vol = dicom_rescale(img_vol, img_header)

        # Check image and mask shape consistances
        assert mask.shape[0] == img_vol.shape[0]

        # Fusion image and mask
        for idx, (img_vol_slice, mask_slice) in enumerate(zip(img_vol, mask)):
            plt.imshow(mask_slice)
            plt.show()
            CT_axial = window_converter(img_vol_slice, 35, 350)
            contoured_img = draw_contour(CT_axial, mask_slice, 'r', thickness=1)
            contoured_img = np.flip(contoured_img, axis=-1) # RGB2BGR for opencv
            cv2.imwrite(str(output_path.joinpath(f"{idx}.png")), contoured_img)


if __name__ == '__main__':
    main()
