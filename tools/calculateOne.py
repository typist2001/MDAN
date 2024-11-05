import argparse
import cv2
import numpy as np
from os import path as osp
import pandas as pd

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def calculate_metrics(gt_path, restored_path, crop_border, suffix, test_y_channel, correct_mean_var):
    """Calculate PSNR and SSIM for a single image.

    Args:
        gt_path (str): Path to the ground truth image.
        restored_path (str): Path to the restored image.
        crop_border (int): Crop border for each side.
        suffix (str): Suffix for restored image.
        test_y_channel (bool): If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.
        correct_mean_var (bool): Correct the mean and variance of restored images.

    Returns:
        dict: Dictionary containing the image filename, PSNR, and SSIM.
    """
    result = {}
    basename, ext = osp.splitext(osp.basename(gt_path))
    result['Filename'] = basename

    img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    img_restored = cv2.imread(restored_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

    if correct_mean_var:
        mean_l = []
        std_l = []
        for j in range(3):
            mean_l.append(np.mean(img_gt[:, :, j]))
            std_l.append(np.std(img_gt[:, :, j]))
        for j in range(3):
            # correct twice
            mean = np.mean(img_restored[:, :, j])
            img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
            std = np.std(img_restored[:, :, j])
            img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

            mean = np.mean(img_restored[:, :, j])
            img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
            std = np.std(img_restored[:, :, j])
            img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

    if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
        img_gt = bgr2ycbcr(img_gt, y_only=True)
        img_restored = bgr2ycbcr(img_restored, y_only=True)

    # calculate PSNR and SSIM
    psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
    ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')

    result['PSNR'] = psnr
    result['SSIM'] = ssim

    return result


def main(args):
    """Calculate PSNR and SSIM for images and output as a table.
    """
    results = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    for gt_path, restored_path in zip(img_list_gt, img_list_restored):
        gt_filename, ext = osp.splitext(osp.basename(gt_path))
        restored_filename, ext = osp.splitext(osp.basename(restored_path))
        if not (gt_filename in restored_filename):
            raise ValueError(f"File names do not match: GT: {gt_filename}, Restored: {restored_filename}")

    for img_path_gt, img_path_restored in zip(img_list_gt, img_list_restored):
        result = calculate_metrics(
            img_path_gt, img_path_restored, args.crop_border, args.suffix, args.test_y_channel, args.correct_mean_var
        )
        results.append(result)

    df = pd.DataFrame(results)
    print(df.to_csv(index=True, sep='\t'))


HR_prefix = r'/media/wanbo/24F6D58CF6D55F1C/typ/bsrn/datasets/HR/'
SR_prefix = r'/mnt/g5/超分辨率-庞欣超-毕业资料/sr_res'

tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']


def calutlate(dataset, scale, method):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default=rf'{HR_prefix}/{dataset}/x{scale}', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default=rf'{SR_prefix}/{method}/x{scale}/{dataset}',
                        help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=scale, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        default=True,
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.'
    )
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    dataset = tests[3]
    scale = 2
    method = 'VLESR'
    calutlate(dataset, scale, method)
