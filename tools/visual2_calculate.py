import argparse, os
import cv2
import numpy as np
from os import path as osp
import pandas as pd

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def calculate_metrics(gt_path, restored_path, crop_border, suffix='', test_y_channel=True, correct_mean_var=False):
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


template = r'''
\begin{figure*}
		\centering
		\begin{subcaptiongroup}
			\centering
			\captionsetup{singlelinecheck=off, justification=centering}
			\parbox[c]{0.3\textwidth}
			{
				\includegraphics[width=0.3\textwidth,align=c]{figure/%num/HR-mark.png}
				\caption*{%s($\times$%s):%s}
			}
			\quad
			\parbox[c]{0.12\textwidth}
			{
				
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/HR.png}
				\caption*{Ground Truth\\PSNR/SSIM}
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/5.png}
				\caption*{%method\\ %s}
			}
			\parbox[c]{0.12\textwidth}
			{
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/1.png}
				\caption*{Bicubic\\
					%s}
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/6.png}
				\caption*{%method\\
					%s}
			}
			\parbox[c]{0.12\textwidth}
			{
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/2.png}
				\caption*{%method\\
					%s}
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/7.png}
				\caption*{%method\\
					%s}
			}
			\parbox[c]{0.12\textwidth}
			{
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/3.png}
				\caption*{%method\\
					%s}
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/8.png}
				\caption*{%method\\
					%s}
			}
			\parbox[c]{0.12\textwidth}
			{
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/4.png}
				\caption*{%method\\
					%s}
				\includegraphics[width=0.12\textwidth,align=c]{figure/%num/9.png}
				\caption*{\textbf{%method\\
						%s}}
			}
		\end{subcaptiongroup}
		\label{fig:comp-global}
	\end{figure*}
'''
methods = ["bicubic", "FSRCNN", "CARN", "IMDN", "PAN", "SAFMN", "SwinIR", "SCET", "SCESN"]
tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']

dataset_index = 1  # 数据集索引
scale = 2  # 缩放因子
num = 'barbara'  # 挑选的图片名

template = template.replace("%s", tests[dataset_index], 1)
template = template.replace("%s", str(scale), 1)
template = template.replace("%s", num, 1)
folder = rf"/media/wanbo/24F6D58CF6D55F1C/typ/bsrn/tools/{tests[dataset_index]}_{num}_x{scale}"

excluded_files = ['HR.png', 'HR-mark.png']

res_list = []
file_list = os.listdir(folder)
filtered_files = [file for file in file_list if file not in excluded_files]
for i, method in enumerate(methods):
    for file in filtered_files:
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            if method in file:
                res = calculate_metrics(os.path.join(folder, "HR.png"), full_path, scale)
                res = f"{round(res['PSNR'], 2):.2f}/{round(res['SSIM'], 4):.4f}"
                print(f"{i + 1}\t{method}\t{res}")
                res_list.append(res)
                break
# 生成latex
orders = [5, 1, 6, 2, 7, 3, 8, 4, 9]
res_list2 = []
for i in orders:
    res_list2.append(res_list[i - 1])
    if i != 1:
        template = template.replace("%method", methods[i - 1], 1)
for data in res_list2:
    template = template.replace("%s", data, 1)

print(template.replace("%num", num))
