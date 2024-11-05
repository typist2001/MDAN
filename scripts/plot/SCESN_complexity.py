def main():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 25
    '''0 - 10'''
    # BSRN-S, FSRCNN
    x = [13]
    y = [30.71]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('FSRCNN', (13 + 10, 30.71 + 0.1), fontsize=notation_size)
    '''10 - 25'''
    # BSRN, RFDN
    x = [550]
    y = [32.24]
    area = (45) * radius**2
    ax.scatter(x, y, s=area, alpha=1.0, marker='.', c='#FFD93D', edgecolors='white', linewidths=2.0)
    # plt.annotate('MDAN(Ours)', (330 - 70, 32.41 + 0.16), fontsize=notation_size)
    plt.annotate('RFDN', (550 - 70, 32.24 + 0.1), fontsize=notation_size)
    '''25 - 50'''
    # IDN, IMDN, PAN,SCESN,SwinIR-light,SCET
    x = [553, 715, 272,583,897,683]
    y = [31.82, 32.21, 32.13,32.55,32.44,32.27]
    area = (60) * radius**2
    ax.scatter(x, y, s=area, alpha=0.6, marker='.', c='#95CD41', edgecolors='white', linewidths=2.0)
    plt.annotate('IDN', (553 - 60, 31.82 + 0.1), fontsize=notation_size)
    plt.annotate('IMDN', (715 + 25, 32.21 - 0.05), fontsize=notation_size)
    plt.annotate('PAN', (272 - 70, 32.13 - 0.25), fontsize=notation_size)
    plt.annotate('SCESN(Ours)', (583 - 70, 32.55 + 0.16), fontsize=notation_size)
    plt.annotate('SwinIR-light', (897 - 90, 32.44 + 0.16), fontsize=notation_size)
    # plt.annotate('ELAN-light', (601 - 70, 32.43 + 0.16), fontsize=notation_size)
    plt.annotate('SCET', (683 - 40, 32.27 + 0.14), fontsize=notation_size)
    '''50 - 100'''
    # SRCNN, CARN, LAPAR-A
    x = [57, 1592, 659]
    y = [30.48, 32.13, 32.15]
    area = (75) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#EAE7C6', edgecolors='white', linewidths=2.0)
    plt.annotate('SRCNN', (57 + 30, 30.48 + 0.1), fontsize=notation_size)
    plt.annotate('LAPAR-A', (659 - 75, 32.15 - 0.23), fontsize=notation_size)
    '''1M+'''
    # LapSRN, VDSR, DRRN, MemNet
    x = [502, 666, 298, 678]
    y = [31.54, 31.35, 31.68, 31.74]
    area = (90) * radius**2
    ax.scatter(x, y, s=area, alpha=0.3, marker='.', c='#264653', edgecolors='white', linewidths=2.0)
    plt.annotate('LapSRN', (502 - 70, 31.54 - 0.25), fontsize=notation_size)
    plt.annotate('VDSR', (666 - 50, 31.35 - 0.25), fontsize=notation_size)
    plt.annotate('DRRN', (298 - 60, 31.68 - 0.25), fontsize=notation_size)
    plt.annotate('MemNet', (678 + 20, 31.74 - 0.18), fontsize=notation_size)
    '''Ours marker'''
    x = [583]
    y = [32.55]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=900)

    plt.xlim(0, 1000)
    plt.ylim(29.75, 32.88)
    plt.xlabel('Parameters (K)', fontsize=notation_size)
    plt.ylabel('PSNR (dB)', fontsize=notation_size)
    plt.title('PSNR vs. Parameters vs. Multi-Adds', fontsize=notation_size)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [40, 60, 80, 95, 110], ['#4D96FF', '#FFD93D', '#95CD41', '#EAE7C6', '#264653'], [0.8, 1.0, 0.6, 0.8, 0.3])
    ]
    ax.legend(
        labelspacing=0.1,
        handles=h,
        handletextpad=1.0,
        markerscale=1.0,
        fontsize=notation_size-6,
        title='Multi-Adds',
        title_fontsize=notation_size-3,
        labels=['<10G', '10G-25G', '25G-50G', '50G-100G', '>100G'],
        scatteryoffsets=[0.0],
        loc='lower right',
        ncol=5,
        shadow=True,
        handleheight=5)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize(notation_size)
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize(notation_size)

    ax.grid(b=True, linestyle='-.', linewidth=0.5)
    plt.show()

    fig.savefig('SCESN_complexity.png')


if __name__ == '__main__':
    main()