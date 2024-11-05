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
    inference_time = [149, 91, 904, 462, 808, 731, 834, 533, 795]

    # PSNR值
    psnr_values = [30.48, 30.72, 32.13, 32.21, 32.16, 32.30, 32.35, 32.41, 32.13]

    # 方法名称
    methods = ['SRCNN', 'FSRCNN', 'CARN', 'IMDN', 'RFDN', 'LatticeNet', 'BSRN', 'MDAN(Ours)', 'PAN']

    area = (20) * radius ** 2
    ax.scatter(inference_time, psnr_values, s=area, alpha=1.0, marker='.', c='#FFD93D', edgecolors='white',
               linewidths=2.0)
    for i, method in enumerate(methods):
        if method == 'PAN' or method == 'CARN':
            plt.annotate(method, (inference_time[i] - 60, psnr_values[i] - 0.15), fontsize=notation_size)
        elif method == 'LatticeNet':
            plt.annotate(method, (inference_time[i] - 150, psnr_values[i] - 0.15), fontsize=notation_size)
        elif method == 'RFDN':
            plt.annotate(method, (inference_time[i] + 10, psnr_values[i] + 0.03), fontsize=notation_size)
        else:
            plt.annotate(method, (inference_time[i] + 20, psnr_values[i] + 0.01), fontsize=notation_size)

    '''Ours marker'''
    x = [533]
    y = [32.41]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=700)

    plt.xlim(0, 950)
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

    fig.savefig('MDAN_time.png')


if __name__ == '__main__':
    main()
