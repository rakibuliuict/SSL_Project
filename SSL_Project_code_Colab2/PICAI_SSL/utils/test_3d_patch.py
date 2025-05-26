import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import os

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

# def load_image_and_label(image_path):
#     h5f = h5py.File(image_path, 'r')
#     t2w = h5f['image']['t2w'][:]
#     adc = h5f['image']['adc'][:]
#     hbv = h5f['image']['hbv'][:]
#     label = h5f['label']['seg'][:].astype(np.uint8)
#     image = np.stack([t2w, adc, hbv], axis=0).astype(np.float32)
#     return image, label

def load_image_and_label(image_path):
    """Load MRI modalities and segmentation mask from an h5 file."""
    print(f"Loading: {image_path}")  # Debugging line
    if not os.path.exists(image_path):  # Check if the file exists
        raise FileNotFoundError(f"File does not exist: {image_path}")
    
    with h5py.File(image_path, 'r') as h5f:
        t2w = h5f['image']['t2w'][:]
        adc = h5f['image']['adc'][:]
        hbv = h5f['image']['hbv'][:]
        label = h5f['label']['seg'][:].astype(np.uint8)

    image = np.stack([t2w, adc, hbv], axis=0).astype(np.float32)  # [3, H, W, D]
    return image, label

def test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes=1):
    c, h, w, d = image.shape
    image = np.transpose(image, (0, 3, 1, 2))
    add_pad = False
    pad_crops = [(0, 0)]
    for i, dim in enumerate(image.shape[1:]):
        if dim < patch_size[i]:
            pad = patch_size[i] - dim
            pad_crops.append((pad // 2, pad - pad // 2))
            add_pad = True
        else:
            pad_crops.append((0, 0))
    if add_pad:
        image = np.pad(image, ((0, 0),) + tuple(pad_crops[1:]), mode='constant')

    d_, h_, w_ = image.shape[1:]
    sz = math.ceil((d_ - patch_size[2]) / stride_z) + 1
    sy = math.ceil((h_ - patch_size[0]) / stride_xy) + 1
    sx = math.ceil((w_ - patch_size[1]) / stride_xy) + 1

    score_map = np.zeros((num_classes, d_, h_, w_), dtype=np.float32)
    cnt = np.zeros((d_, h_, w_), dtype=np.float32)

    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                zs = min(stride_z * z, d_ - patch_size[2])
                ys = min(stride_xy * y, h_ - patch_size[0])
                xs = min(stride_xy * x, w_ - patch_size[1])
                patch = image[:, zs:zs + patch_size[2], ys:ys + patch_size[0], xs:xs + patch_size[1]]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).cuda()
                with torch.no_grad():
                    y1 = F.softmax(model1(patch_tensor)[0], dim=1)
                    y2 = F.softmax(model2(patch_tensor)[0], dim=1)
                    y = ((y1 + y2) / 2).cpu().numpy()[0]
                score_map[:, zs:zs + patch_size[2], ys:ys + patch_size[0], xs:xs + patch_size[1]] += y
                cnt[zs:zs + patch_size[2], ys:ys + patch_size[0], xs:xs + patch_size[1]] += 1

    score_map = score_map / (cnt + 1e-5)
    pred = np.argmax(score_map, axis=0).astype(np.uint8)

    if add_pad:
        z0, z1 = pad_crops[1][0], d_ - pad_crops[1][1]
        y0, y1 = pad_crops[2][0], h_ - pad_crops[2][1]
        x0, x1 = pad_crops[3][0], w_ - pad_crops[3][1]
        pred = pred[z0:z1, y0:y1, x0:x1]
    return pred, score_map

def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    return test_single_case_mean(model, model, image, stride_xy, stride_z, patch_size, num_classes)

def test_single_case_plus(model1, model2, image, stride_xy, stride_z, patch_size, num_classes=1):
    return test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)

# def var_all_case_LA(model, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
#     with open('/content/drive/MyDrive/SSL/SSL_Project_code_Colab2/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
#         image_list = f.readlines()
#     image_list = ["/content/drive/MyDrive/SSL/Dataset/PICAI_dataset/" + item.strip() + "/" + item.strip() + ".h5" for item in image_list]
#     loader = tqdm(image_list)
#     total_dice = 0.0
#     for image_path in loader:
#         image, label = load_image_and_label(image_path)
#         prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
#         dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
#         total_dice += dice
#     avg_dice = total_dice / len(image_list)
#     print('Average Dice coefficient: {:.4f}'.format(avg_dice))
#     return avg_dice

# def var_all_case_LA(model, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    """Evaluate the model on all test cases and return the average Dice score."""
    image_list = []
    with open('/content/drive/MyDrive/SSL/SSL_Project_code_Colab2/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()

    # Construct valid paths for each patient
    image_list = [os.path.join('/content/drive/MyDrive/SSL/Dataset/PICAI_dataset/', item.strip(), f"{item.strip()}.h5") for item in image_list]
    
    total_dice = 0.0
    loader = tqdm(image_list)

    for image_path in loader:
        try:
            image, label = load_image_and_label(image_path)
        except FileNotFoundError:
            print(f"Skipping missing file: {image_path}")
            continue

        prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice

    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def var_all_case_LA(model, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    """Evaluate the model on all test cases and return the average Dice score."""
    with open('/content/drive/MyDrive/SSL/SSL_Project_code_Colab2/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()

    image_list = [os.path.join('/content/drive/MyDrive/SSL/Dataset/PICAI_dataset/', item.strip(), f"{item.strip()}.h5") for item in image_list]
    
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        try:
            image, label = load_image_and_label(image_path)
        except FileNotFoundError:
            print(f"Skipping missing file: {image_path}")
            continue

        # Ensure the dimensions of prediction and label match
        prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        
        # Check if prediction and label need to be transposed to match
        if prediction.shape != label.shape:
            # Adjust the dimensions if necessary
            label = np.transpose(label, (2, 0, 1))  # Ensure label shape is (D, H, W)
        
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice

    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def var_all_case_LA_mean(model1, model2, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SSL/SSL_Project_code_Colab2/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["/content/drive/MyDrive/SSL/Dataset/PICAI_dataset/" + item.strip() + "/" + item.strip() + ".h5" for item in image_list]
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

def var_all_case_LA_plus(model1, model2, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SSL/SSL_Project_code_Colab2/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["/content/drive/MyDrive/SSL/Dataset/PICAI_dataset/" + item.strip() + "/" + item.strip() + ".h5" for item in image_list]
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



