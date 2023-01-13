import math
import os

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm


def mask_array_to_bounding_box(mask_array: np.ndarray):
    '''
    Determine the bounding box index of the binary mask
    :param mask_array: mask array in numpy ndarray format
    :return: two-dimensional index coordinates ([x_min, x_max],[y_min, y_max]), center index, and mask size of the bounding box
    '''
    x_bound_index = np.where(np.max(mask_array, axis=0))[0]
    y_bound_index = np.where(np.max(mask_array, axis=1))[0]
    x_bound_index = [int(x_bound_index[0]), int(x_bound_index[-1])]
    y_bound_index = [int(y_bound_index[0]), int(y_bound_index[-1])]
    boundary_index = [[x_bound_index[0], x_bound_index[-1]], [y_bound_index[0], y_bound_index[-1]]]
    bb_center = [(x_bound_index[0] + x_bound_index[-1]) / 2, (y_bound_index[0] + y_bound_index[-1]) / 2]
    bb_size = [(-x_bound_index[0] + x_bound_index[-1]), (-y_bound_index[0] + y_bound_index[-1])]
    return boundary_index, bb_center, bb_size


def mask_to_bounding_box(mask_image: sitk.Image, margin_size=5):
    '''
    Determine the bounding box coordinates of the binary mask
    :param mask_image: mask image in SimpleITK format
    :param margin_size: margin size for the bounding box
    :return: two-dimensional coordinates ([x_min, x_max],[y_min, y_max]), center index, and mask size of the bounding box
    '''
    mask_array = sitk.GetArrayFromImage(mask_image)
    boundary_index, bb_center, bb_size = mask_array_to_bounding_box(mask_array)
    boundary_index_abs = [
        list(mask_image.TransformIndexToPhysicalPoint(
            [boundary_index[0][0] - margin_size, boundary_index[0][1] + margin_size])),
        list(mask_image.TransformIndexToPhysicalPoint(
            [boundary_index[1][0] - margin_size, boundary_index[1][1] + margin_size]))
    ]
    bb_center_abs = list(mask_image.TransformContinuousIndexToPhysicalPoint(bb_center))
    bb_size_abs = [boundary_index_abs[0][1] - boundary_index_abs[0][0],
                   boundary_index_abs[1][1] - boundary_index_abs[1][0]]
    return boundary_index_abs, bb_center_abs, bb_size_abs


def mask_to_bounding_box_image(mask_image: sitk.Image):
    '''
    Acquire the bounding box as a binary mask image.
    :param mask_image: the mask image in SimpleITK format
    :return: the bounding box mask in SimpleITK format
    '''
    mask_array = sitk.GetArrayFromImage(mask_image)
    boundary_index, bb_center, bb_size = mask_array_to_bounding_box(mask_array)

    x_bound_left = max(int(boundary_index[0][0] - 1), 0)
    x_bound_right = min(int(boundary_index[0][-1] + 2), mask_array.shape[1])
    y_bound_left = max(int(boundary_index[1][0] - 1), 0)
    y_bound_right = min(int(boundary_index[1][-1] + 2), mask_array.shape[0])

    bounding_box_mask_x = np.zeros(mask_array.shape)
    bounding_box_mask_x[:, x_bound_left:x_bound_right] = 1
    bounding_box_mask_y = np.zeros(mask_array.shape)
    bounding_box_mask_y[y_bound_left:y_bound_right, :] = 1
    bounding_box_mask = np.all([bounding_box_mask_y, bounding_box_mask_x], axis=0)
    bounding_box_mask_image = sitk.GetImageFromArray(bounding_box_mask.astype(int))
    bounding_box_mask_image.CopyInformation(mask_image)
    bounding_box_mask_image = sitk.Cast(bounding_box_mask_image, sitk.sitkUInt8)
    return bounding_box_mask_image


def image_flatten(image):
    '''
    Remove the third dimension of the 2D radiograph
    :param image: image in SimpleITK format
    :return: flattened 2D image
    '''
    if image.GetDimension() == 2:
        return image
    flattened_image_array = sitk.GetArrayFromImage(image)[0, :, :]
    flattened_image = sitk.GetImageFromArray(flattened_image_array)
    flattened_image.SetOrigin(image.GetOrigin()[:-1])
    flattened_image.SetSpacing(image.GetSpacing()[:-1])
    return flattened_image


def alignment_score(image_array, angle):
    '''
    Estimate the degree of patella alignment vertically
    :param image_array: image array in numpy ndarray format
    :param angle: the current rotation angle for the image
    :return: the estimated alignment score for the current rotation angle
    '''
    rotated_array = rotate(image_array, angle, reshape=False, order=1)
    boundary_index, _, _ = mask_array_to_bounding_box(rotated_array)
    score = (boundary_index[0][1] - boundary_index[0][0]) / (boundary_index[1][1] - boundary_index[1][0])
    return score


def image_normalization(image_sitk: sitk.Image, mask_sitk: sitk.Image, scale=100, shift=600, threshold=1200):
    '''
    Image normalization and thresholding
    :param image_sitk: the original image before normalization
    :param mask_sitk: the mask within which the normalization inference is acquired
    :param scale: the re-scale factor
    :param shift: the pixel shift factor
    :param threshold: the upper threshold for pixel cut-off. The lower threshold is set as 0
    :return: the normalized and thresholded image
    '''
    image_array = sitk.GetArrayFromImage(image_sitk).astype(float)
    masked_image_array = image_array[sitk.GetArrayFromImage(mask_sitk).astype(bool)]
    mean = np.mean(masked_image_array)
    std = np.std(masked_image_array)
    if std == 0:
        print('Warning: zero std!')
    image_array = (image_array - mean) / std * scale + shift
    image_array[image_array < 0] = 0
    image_array[image_array > threshold] = threshold
    image = sitk.GetImageFromArray(image_array.astype('uint16'))
    image.CopyInformation(image_sitk)
    return image


def image_preprocessing(cleaned_data_directory, export_directory, image_filename, mask_filename,
                        export_image_filename=None,
                        export_mask_filename=None):
    '''
    The image preprocessing pipeline. Including patella alignment, image normalization, and thresholding
    :param cleaned_data_directory: The source directory where the image and mask of each patient are stored in individual folders
    :param export_directory: The target directory where the preprocessing images and masks are saved
    :param image_filename: the file name of the input image
    :param mask_filename: the file name of the input mask
    :param export_image_filename: the file name of the output image, optional
    :param export_mask_filename: the file name of the output mask, optional
    '''
    if export_image_filename is None:
        export_image_filename = image_filename
    if export_mask_filename is None:
        export_mask_filename = mask_filename
    resolution = [0.5, 0.5]
    image_resampler = sitk.ResampleImageFilter()
    image_resampler.SetOutputSpacing([0.5, 0.5])
    image_resampler.SetOutputPixelType(sitk.sitkUInt16)
    image_resampler.SetInterpolator(sitk.sitkLinear)
    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetOutputSpacing([0.5, 0.5])
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_resampler.SetOutputPixelType(sitk.sitkUInt8)
    index = 0
    for patient_id in tqdm(os.listdir(cleaned_data_directory)):
        image_filepath = os.path.join(cleaned_data_directory, patient_id, image_filename)
        mask_filepath = os.path.join(cleaned_data_directory, patient_id, mask_filename)
        if not os.path.exists(image_filepath) or not os.path.exists(mask_filepath):
            continue
        # index += 1
        # if index > 10:
        #     return
        # if patient_id != "M0007_TDNT":
        #     continue
        image_sitk = sitk.ReadImage(image_filepath)
        image_sitk = image_flatten(image_sitk)
        mask_sitk = sitk.ReadImage(mask_filepath)
        mask_sitk = image_flatten(mask_sitk)
        # crop image and mask based on the mask bounding box with margin for free rotation
        boundary_index_abs, bb_center_abs, bb_size_abs = mask_to_bounding_box(mask_sitk, margin_size=0)
        bb_size_diagonal = np.sqrt(bb_size_abs[0] ** 2 + bb_size_abs[1] ** 2)
        origin = [boundary_index_abs[0][0] - (bb_size_diagonal - bb_size_abs[0]) / 2,
                  boundary_index_abs[1][0] - (bb_size_diagonal - bb_size_abs[1]) / 2]
        size = [int(bb_size_diagonal / resolution[0]), int(bb_size_diagonal / resolution[1])]
        mask_resampler.SetOutputOrigin(origin)
        mask_resampler.SetSize(size)
        cropped_mask = mask_resampler.Execute(mask_sitk)
        #  find the best angle for vertical patella bounding box alignment
        angles = np.arange(-15, 15, 1)
        scores = []
        for angle in angles:
            score = alignment_score(sitk.GetArrayFromImage(cropped_mask), angle)
            scores.append(score)
        # print(dict(zip(angles,scores)))
        best_angle = angles[scores.index(min(scores))]
        best_angle = best_angle / 180 * math.pi
        # rotate image and mask by resampling
        rotation_matrix = [[math.cos(best_angle), -math.sin(best_angle)], [math.sin(best_angle), math.cos(best_angle)]]
        origin = np.dot(np.array(rotation_matrix),
                        np.transpose([np.array(origin) - np.array(bb_center_abs)])).flatten() + np.array(bb_center_abs)
        image_resampler.SetOutputOrigin(origin)
        image_resampler.SetOutputDirection(np.array(rotation_matrix).flatten().tolist())
        image_resampler.SetSize(size)
        mask_resampler.SetOutputOrigin(origin)
        mask_resampler.SetOutputDirection(np.array(rotation_matrix).flatten().tolist())
        cropped_image = image_resampler.Execute(image_sitk)
        cropped_mask = mask_resampler.Execute(mask_sitk)
        bounding_box_mask = mask_to_bounding_box_image(cropped_mask)
        normalized_image = image_normalization(cropped_image, cropped_mask, scale=100, shift=600, threshold=1200)

        patient_export_directory = os.path.join(export_directory, patient_id)
        if not os.path.exists(patient_export_directory):
            os.mkdir(patient_export_directory)

        sitk.WriteImage(normalized_image, os.path.join(patient_export_directory, export_image_filename),
                        useCompression=True)
        sitk.WriteImage(cropped_mask, os.path.join(patient_export_directory, export_mask_filename), useCompression=True)
        sitk.WriteImage(bounding_box_mask, os.path.join(patient_export_directory, 'bounding_box.mha'),
                        useCompression=True)


def roi_generation(preprocessed_data_directory, mask_filename, roi_name='ROI'):
    '''
    Generate the 3x3 rectangular ROIs based on the patella bounding box
    :param preprocessed_data_directory: preprocessed image and mask database directory
    :param mask_filename: mask file name
    :param roi_name: ROI prefix for the file name when saving the ROI masks
    '''
    roi_number = 3
    index = 0
    for patient_id in tqdm(os.listdir(preprocessed_data_directory)):
        # if index >= 3:
        #     break
        patient_directory = os.path.join(preprocessed_data_directory, patient_id)
        mask_filepath = os.path.join(patient_directory, mask_filename)
        mask = sitk.ReadImage(mask_filepath)
        mask_array = sitk.GetArrayFromImage(mask)

        boundary_index, bb_center, bb_size = mask_array_to_bounding_box(mask_array)
        roi_size = [int(bb_size[0] / roi_number), int(bb_size[1] / roi_number)]

        x_starting_index = boundary_index[0][0]

        for i in range(roi_number):
            if i == roi_number - 1:
                x_ending_index = boundary_index[0][1]
            else:
                x_ending_index = x_starting_index + roi_size[0]
            y_starting_index = boundary_index[1][0]
            for j in range(roi_number):
                if j == roi_number - 1:
                    y_ending_index = boundary_index[1][1]
                else:
                    y_ending_index = y_starting_index + roi_size[1]
                rec_mask_1 = np.zeros(mask_array.shape)
                rec_mask_1[y_starting_index:y_ending_index, :] = 1
                rec_mask_2 = np.zeros(mask_array.shape)
                rec_mask_2[:, x_starting_index:x_ending_index] = 1
                roi_mask_array = np.all([mask_array, rec_mask_1, rec_mask_2], axis=0)
                roi_mask = sitk.GetImageFromArray(roi_mask_array.astype(int))
                roi_mask.CopyInformation(mask)
                sitk.WriteImage(roi_mask, os.path.join(patient_directory, roi_name + '{0}{1}_mask.mha'.format(j, i)),
                                useCompression=True)
                y_starting_index = y_ending_index
            x_starting_index = x_ending_index
        index += 1


if __name__ == '__main__':
    cleaned_data_directory = r"../data_cleaning"
    export_directory = r"../preprocessing"
    image_filename = 'image.mha'
    mask_filename = 'unet_mask_jiachen.mha'
    export_mask_filename = 'mask.mha'
    # image_preprocessing(cleaned_data_directory, export_directory, image_filename, mask_filename, export_mask_filename=export_mask_filename)
    mask_filename = 'bounding_box.mha'
    roi_generation(export_directory, mask_filename, roi_name='ROI')
