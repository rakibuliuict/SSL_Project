import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        largestCC = segmentation
    return largestCC

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd

def load_image_and_label(image_path):
    h5f = h5py.File(image_path, 'r')
    t2w = h5f['image']['t2w'][:]
    adc = h5f['image']['adc'][:]
    hbv = h5f['image']['hbv'][:]
    label = h5f['label']['seg'][:].astype(np.uint8)
    image = np.stack([t2w, adc, hbv], axis=0).astype(np.float32)
    return image, label

def var_all_case_LA_mean(model1, model2, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SemiSL/Code/SSL_Project/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset/" + item.strip() + "/" + item.strip() + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        image, label = load_image_and_label(image_path)
        prediction, _ = test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def var_all_case_LA(model, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SemiSL/Code/SSL_Project/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset/" + item.strip() + "/" + item.strip() + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        image, label = load_image_and_label(image_path)
        prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def test_all_case_average(model1, model2, image_list, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list)
    total_metric = 0.0
    for ith, image_path in enumerate(loader):
        image, label = load_image_and_label(image_path)
        if preproc_fn:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
        if nms:
            prediction = getLargestCC(prediction)
        single_metric = calculate_metric_percase(prediction, label) if np.sum(prediction) > 0 else (0, 0, 0, 0)
        total_metric += np.asarray(single_metric)
        if save_result and test_save_path:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_pred.nii.gz")
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_score.nii.gz")
            nib.save(nib.Nifti1Image(image[0].astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_img.nii.gz")
            nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('Average metrics: {}'.format(avg_metric))
    return avg_metric

def test_all_case(model, image_list, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list)
    total_metric = 0.0
    for ith, image_path in enumerate(loader):
        image, label = load_image_and_label(image_path)
        if preproc_fn:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes)
        if nms:
            prediction = getLargestCC(prediction)
        single_metric = calculate_metric_percase(prediction, label) if np.sum(prediction) > 0 else (0, 0, 0, 0)
        total_metric += np.asarray(single_metric)
        if save_result and test_save_path:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[0].astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_img.nii.gz")
            nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('Average metrics: {}'.format(avg_metric))
    return avg_metric

def var_all_case_LA_plus(model1, model2, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SemiSL/Code/SSL_Project/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset/" + item.strip() + "/" + item.strip() + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        image, label = load_image_and_label(image_path)
        prediction, _ = test_single_case_plus(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def test_all_case_plus(model1, model2, image_list, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list)
    total_metric = 0.0
    for ith, image_path in enumerate(loader):
        image, label = load_image_and_label(image_path)
        if preproc_fn:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_plus(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
        if nms:
            prediction = getLargestCC(prediction)
        single_metric = calculate_metric_percase(prediction, label) if np.sum(prediction) > 0 else (0, 0, 0, 0)
        total_metric += np.asarray(single_metric)
        if save_result and test_save_path:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[0].astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_img.nii.gz")
            nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)), f"{test_save_path}/{ith:02d}_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('Average metrics: {}'.format(avg_metric))
    return avg_metric

def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=2):
    return test_single_case_mean(model, model, image, stride_xy, stride_z, patch_size, num_classes)

def test_single_case_plus(model1, model2, image, stride_xy, stride_z, patch_size, num_classes=2):
    return test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
