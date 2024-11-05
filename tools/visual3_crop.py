import cv2
import os
import time, shutil

# Urban096
# x1 = 409
# y1 = 181
# x2 = 526
# y2 = 299


# Set14 barbara x2
x1 = 505
y1 = 260
x2 = x1 + 40
y2 = y1 + 40

# Urban100 img_099 x2
# x1 = 31
# y1 = 86
# x2 = x1 + 120
# y2 = y1 + 120

# Urban100 img_004 x2
# x1 = 880
# y1 = 560
# x2 = x1 + 120
# y2 = y1 + 120

# Urban100 img_099 x3
# x1 = 290
# y1 = 490
# x2 = x1 + 100
# y2 = y1 + 100

# Urban100 img_044 x4
# x1 = 585
# y1 = 110
# x2 = x1 + 150
# y2 = y1 + 150


# BSD100 253027 x2
# x1 = 426
# y1 = 116
# x2 = x1 + 44
# y2 = y1 + 44

# BSD100 78004 x4
# x1 = 172
# y1 = 113
# x2 = x1 + 80
# y2 = y1 + 80

# Urban024 x2
x1 = 500
y1 = 100
x2 = x1 + 70
y2 = y1 + 70


# Urban008 x4
# x1 = 305
# y1 = 40
# x2 = x1 + 120
# y2 = y1 + 120

# Urban092 x4
# x1 = 510
# y1 = 270
# x2 = x1 + 60
# y2 = y1 + 60


def get_img(input_dir):
    img_paths = []
    for (path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            print(filename)
            img_paths.append(path + '/' + filename)
    print("img_paths:", img_paths)
    return img_paths


def cut_img(img_paths, output_dir):
    scale = len(img_paths)
    for i, img_path in enumerate(img_paths):
        a = "#" * int(i / 1000)
        b = "." * (int(scale / 1000) - int(i / 1000))
        c = (i / scale) * 100
        time.sleep(0.2)
        print('正在处理图像： %s' % img_path.split('/')[-1])
        img = cv2.imread(img_path)
        weight = img.shape[1]
        if weight > 1600:
            cropImg = img[x1:y1, x2:y2]  # 裁剪【y1,y2：x1,x2】
            # cropImg = cv2.resize(cropImg, None, fx=0.5, fy=0.5,
            # interpolation=cv2.INTER_CUBIC) #缩小图像
            cv2.imwrite(output_dir + '/' + img_path.split('/')[-1], cropImg)
        else:
            cropImg_01 = img[y1:y2, x1:x2]
            path = output_dir + '/' + img_path.split('/')[-1]
            cv2.imwrite(path, cropImg_01)
        print('{:^3.3f}%[{}>>{}]'.format(c, a, b))
    print('100%')


methods = ["bicubic", "FSRCNN", "CARN", "IMDN", "PAN", "SAFMN", "SwinIR", "SCET", "MDAN"]


def renameOutPut(output_dir):
    renames_dir = os.path.join(output_dir, "renames")
    os.makedirs(renames_dir, exist_ok=True)

    # 获取output_dir下的所有文件
    files = os.listdir(output_dir)
    files = [item for item in files if os.path.isfile(os.path.join(output_dir, item))]
    # 对文件进行排序并重命名并复制到renames目录
    for file in files:
        for i, method in enumerate(methods):
            if method in file:
                new_file_name = f"{i + 1}.png"
                src_file = os.path.join(output_dir, file)
                dst_file = os.path.join(renames_dir, new_file_name)
                shutil.copy(src_file, dst_file)
                print("重命名：", src_file, "-->", dst_file)
                break
        else:
            src_file = os.path.join(output_dir, file)
            dst_file = os.path.join(renames_dir, file)
            shutil.copy(src_file, renames_dir)
            print("复制：", src_file, "-->", dst_file)
            print(src_file)


if __name__ == '__main__':
    input_dir = "./Urban100_img_024_x2"  # 读取图片目录表
    output_dir = input_dir + "-out";  # 保存截取的图像目录
    global_dir = input_dir + "/HR.png";
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_paths = get_img(input_dir)
    print('图片获取完成 。。。！')
    cut_img(img_paths, output_dir)
    ima = cv2.imread(global_dir)
    cv2.rectangle(ima, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(output_dir + '/HR-mark.png', ima)
    renameOutPut(output_dir)
