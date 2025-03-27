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

def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=2):
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
                    y1, _ = model(patch_tensor)
                    y = F.softmax(y1, dim=1).cpu().numpy()[0]
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

def test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes=2):
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

def var_all_case_LA(model, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SemiSL/Code/SSL_Project/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [f"/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset/{item.strip()}/{item.strip()}.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:].astype(np.float32)
        label = h5f['label']['seg'][:].astype(np.uint8)
        prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes)
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def var_all_case_LA_mean(model1, model2, num_classes, patch_size=(256, 256, 16), stride_xy=18, stride_z=4):
    with open('/content/drive/MyDrive/SemiSL/Code/SSL_Project/PICAI_SSL/Datasets/picai/data_split/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [f"/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset/{item.strip()}/{item.strip()}.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:].astype(np.float32)
        label = h5f['label']['seg'][:].astype(np.uint8)
        prediction, _ = test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
        dice = metric.binary.dc(prediction, label) if np.sum(prediction) > 0 else 0
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('Average Dice coefficient: {:.4f}'.format(avg_dice))
    return avg_dice

def test_all_case(model, image_list, num_classes, patch_size, stride_xy, stride_z, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list)
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes)
        if nms:
            prediction = getLargestCC(prediction)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label)
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "%02d_pred.nii.gz" % ith)
        ith += 1
        total_metric += np.asarray(single_metric)
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric

def test_all_case_average(model1, model2, image_list, num_classes, patch_size, stride_xy, stride_z, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list)
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes)
        if nms:
            prediction = getLargestCC(prediction)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label)
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "%02d_pred.nii.gz" % ith)
        ith += 1
        total_metric += np.asarray(single_metric)
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric

def test_all_case_plus(model_l, model_r, image_list, num_classes, patch_size, stride_xy, stride_z, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list)
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes)
        if nms:
            prediction = getLargestCC(prediction)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label)
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "%02d_pred.nii.gz" % ith)
        ith += 1
        total_metric += np.asarray(single_metric)
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric

def test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    add_pad = False
    w_pad = max(0, patch_size[0] - w)
    h_pad = max(0, patch_size[1] - h)
    d_pad = max(0, patch_size[2] - d)
    if w_pad or h_pad or d_pad:
        add_pad = True
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)
    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, 0), 0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    y1_l, _ = model_l(test_patch)
                    y1_r, _ = model_r(test_patch)
                    y1 = (y1_l + y1_r) / 2
                    y = F.softmax(y1, dim=1).cpu().numpy()
                y = y[0, 1, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.uint8)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map
