import os
import cv2

LR_prefix = r'E:\project\pyProjects\bsrn\datasets\LR'
SR_prefix = r'E:\sr_res\bicubic'
tests = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']

def bicubic(dataset, scale,keepdims=False):
    lr_image_dir = f'{LR_prefix}\\{dataset}\\x{scale}'
    hr_image_dir = f'{SR_prefix}\\x{scale}\\{dataset}'

    print(f'输入文件夹：{lr_image_dir}')
    print(f'输出文件夹：{hr_image_dir}')

    # create LR image dirs
    os.makedirs(hr_image_dir, exist_ok=True)  # 创建保存结果的文件夹

    supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                             ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                             ".tiff")

    # Upsample LR images
    for filename in os.listdir(lr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        name, ext = os.path.splitext(filename)
        # Read LR image
        lr_img = cv2.imread(os.path.join(lr_image_dir, filename))
        hr_img_dims = (lr_img.shape[1], lr_img.shape[0])

        # Upsample image
        lr_image = cv2.resize(lr_img, (0, 0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        if keepdims:
            lr_image = cv2.resize(lr_image, hr_img_dims, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(
            os.path.join(hr_image_dir + f"\\{name}_test_bicubic_x{scale}{ext}"),
            lr_image)  # 保存高分辨率图像
if __name__ == '__main__':
    for dataset in tests:
        for scale in (2, 3, 4):
            bicubic(dataset, scale)