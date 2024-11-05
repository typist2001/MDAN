import os
import glob
import shutil

# methods = ["bicubic", "FSRCNN", "CARN", "IMDN", "PAN", "SAFMN", "SwinIR", "SCET", "MDAN"]
methods = ["SwinIR"]

tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']

dataset_index = 3  # 数据集索引
scale = 4  # 缩放因子
num = 'img_092'  # 挑选的图片名

src_path = rf'/mnt/g5/超分辨率-庞欣超-毕业资料/sr_res/%s/x{scale}/%s'
hr_path = rf'/media/wanbo/24F6D58CF6D55F1C/typ/bsrn/datasets/HR/{tests[dataset_index]}/x{scale}/{num}.png'

folder = rf"/media/wanbo/24F6D58CF6D55F1C/typ/bsrn/tools/{tests[dataset_index]}_{num}_x{scale}"
os.makedirs(folder, exist_ok=True)
shutil.copy(hr_path, os.path.join(folder, "HR.png"))
for i in methods:
    files_with = glob.glob(os.path.join(src_path % (i, tests[dataset_index]), f"*{num}*"))
    for file_path in files_with:
        print(file_path)
        shutil.copy(file_path, rf'{folder}/{num}_{i}.png')
