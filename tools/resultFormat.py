SR_prefix_tmp = r'E:\sr_res\待整理\IDN'
SR_prefix = r'E:\sr_res\IDN'
tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
tests_tmp = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
import os
import shutil

def move_files(source_dir, destination_dir):
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        shutil.move(source_file, destination_file)
        print(f"移动文件：{source_file} -> {destination_file}")
def move():
    # 调用函数进行文件移动
    for i in (2,3,4):
        for j in range(len(tests)):
            source_directory = rf'{SR_prefix_tmp}\{tests_tmp[j]}\{tests_tmp[j]}_x{i}'
            destination_directory = rf'E:\sr_res\IDN\x{i}\{tests[j]}'
            os.makedirs(destination_directory, exist_ok=True)
            move_files(source_directory, destination_directory)
def parent(path):
    return os.path.dirname(path)
def rename_files(directory):
    # 3096x2.png重命名为3096_test_IMDN_x2.png
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            new_filename = 'test_IDN_'.join((filename[:-6],filename[-6:]))


            # new_filename = ('_test_PAN_'+os.path.basename((parent(parent(file_path))))).join((filename[:-4], filename[-4:]))
            new_file_path = os.path.join(dirpath, new_filename)
            os.rename(file_path, new_file_path)
            print(f"重命名文件：{file_path} -> {new_file_path}")

rename_files(SR_prefix)
# move()

