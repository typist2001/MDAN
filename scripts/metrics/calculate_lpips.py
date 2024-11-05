import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main(method, test, scale=2):
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = fr'E:\project\pyProjects\bsrn\datasets\HR\{test}\x{scale}'
    folder_restored = fr'E:\sr_res\{method}\x{scale}\{test}'
    # crop_border = 4
    # suffix = '_test_bicubic_x2'
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    # print(folder_gt)
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))
    img_list2 = sorted(glob.glob(osp.join(folder_restored, '*')))
    # print(img_list)
    # print(img_list2)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # img_path2 = osp.join(folder_restored, basename + suffix + ext)
        img_path2 = img_list2[i]
        img_restored = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        # print(f'{i + 1:3d}: {basename:25}. \tLPIPS: {lpips_val.item():.6f}.')

        lpips_all.append(lpips_val.item())
    res = f'{sum(lpips_all) / len(lpips_all):.6f}'
    # print(f'Average: LPIPS: {res}')
    return res


if __name__ == '__main__':
    # methods = ['bicubic', 'FSRCNN', 'CARN', 'IMDN', 'PAN', 'BSRN', 'SwinIR', 'SCET', 'SAFMN', 'MDAN', 'SCESN']
    methods = ['PAN']
    tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
    dataset_index = 3  # 数据集索引
    scale = 2  # 缩放因子
    print(tests[dataset_index])
    for method in methods:
        print(method, main(method, tests[dataset_index], scale))
