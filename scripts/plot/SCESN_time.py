def main():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 25
    '''10 - 25'''
    # BSRN, RFDN
    # x = [149, 91, 904, 462, 808, 731, 834, 533, 795]
    # y = [30.48, 30.72, 32.13, 32.21, 32.16, 32.30, 32.35, 32.41, 32.13]
    # 推理时间
    # inference_time = [149, 91, 904, 462, 808, 731, 834, 533, 795, 978]

    # PSNR值
    # psnr_values = [30.48, 30.72, 32.13, 32.21, 32.16, 32.30, 32.35, 32.41, 32.13, 32.55]

    # 方法名称
    # methods = ['SRCNN', 'FSRCNN', 'CARN', 'IMDN', 'RFDN', 'LatticeNet', 'BSRN', 'MDAN(Ours)', 'PAN', 'SCESN']

    result_dict = {
        'SRCNN': {'inference_time': 149, 'psnr': 30.48, 'x_offset': 20, 'y_offset': 0.01, 'mark': False},
        'FSRCNN': {'inference_time': 91, 'psnr': 30.72, 'x_offset': 20, 'y_offset': 0.01, 'mark': False},
        'CARN': {'inference_time': 904, 'psnr': 32.13, 'x_offset': -60, 'y_offset': -0.15, 'mark': False},
        # 'IMDN': {'inference_time': 462, 'psnr': 32.21, 'x_offset': 20, 'y_offset': 0.01, 'mark': False},
        'RFDN': {'inference_time': 808, 'psnr': 32.16, 'x_offset': -300, 'y_offset': -0.1, 'mark': False},
        'LatticeNet': {'inference_time': 731, 'psnr': 32.30, 'x_offset': -350, 'y_offset': 0.08, 'mark': False},
        # 'BSRN': {'inference_time': 834, 'psnr': 32.35, 'x_offset': 20, 'y_offset': 0.01, 'mark': False},
        # 'MDAN(Ours)': {'inference_time': 533, 'psnr': 32.41, 'x_offset': 20, 'y_offset': 0.01, 'mark': False},
        # 'PAN': {'inference_time': 795, 'psnr': 32.13, 'x_offset': -60, 'y_offset': -0.15, 'mark': False},
        'SCESN(Ours)': {'inference_time': 978, 'psnr': 32.55, 'x_offset': 30, 'y_offset': 0.03, 'mark': False},
        'SwinIR-light': {'inference_time': 953, 'psnr': 32.44, 'x_offset': 50, 'y_offset': -0.08, 'mark': False},
        'SCET': {'inference_time': 1124, 'psnr': 32.27, 'x_offset': 30, 'y_offset': -0.12, 'mark': False},
        'EDSR': {'inference_time': 2694, 'psnr': 32.46, 'x_offset': -150, 'y_offset': -0.15, 'mark': False},
    }

    area = (20) * radius ** 2
    inference_time = [values['inference_time'] for values in result_dict.values()]
    psnr_values = [values['psnr'] for values in result_dict.values()]
    ax.scatter(inference_time, psnr_values, s=area, alpha=1.0, marker='.', c='#FFD93D', edgecolors='white',
               linewidths=2.0)
    for method, values in result_dict.items():
        x_offset = values['x_offset']
        y_offset = values['y_offset']
        plt.annotate(method, (values['inference_time'] + x_offset, values['psnr'] + y_offset),
                     fontsize=notation_size)
    '''Ours marker'''
    x = [978]
    y = [32.55]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=700)

    plt.xlim(0, 2800)
    plt.ylim(30.4, 32.75)
    plt.xlabel('Inference time (ms)', fontsize=notation_size)
    plt.ylabel('PSNR (dB)', fontsize=notation_size)
    plt.title('PSNR vs. Inference time', fontsize=notation_size)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize(notation_size)
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize(notation_size)

    ax.grid(b=True, linestyle='-.', linewidth=0.5)
    plt.show()

    fig.savefig('SCESN_time.png')


if __name__ == '__main__':
    main()
