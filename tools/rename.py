import glob, os
import os.path as osp


def rename(folder_gt, folder_restored):
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))
    img_list2 = sorted(glob.glob(osp.join(folder_restored, '*')))

    for i, img_path in enumerate(img_list):
        img_path2 = img_list2[i]

        # Extract the file name from img_path
        _, filename = osp.split(img_path)

        # Create the new path by joining folder_restored and the extracted filename
        new_img_path2 = osp.join(folder_restored, filename)

        # Rename the file in folder_restored to match the filename in folder_gt
        os.rename(img_path2, new_img_path2)

        print(f'Renamed: {img_path2} to {new_img_path2}')


tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
# tests = ['Set5', 'Set14', 'BSD100', 'Urban100']
method = 'MDAN'
for test in tests:
    for i in (2, 3, 4):
        folder_gt = rf'E:\project\pyProjects\bsrn\datasets\HR\{test}\x{i}'
        folder_restored = rf'E:\sr_res\{method}\x{i}\{test}'
        rename(folder_gt, folder_restored)
