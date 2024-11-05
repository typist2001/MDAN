from basicsr.archs.MDAN_abl_bfn_3_arch import MDAN_abl_bfn_3
from basicsr.archs.MDAN_abl_bfn_5_arch import MDAN_abl_bfn_5
from basicsr.archs.MDAN_abl_esa2_arch import MDAN_abl_esa2
from thop import profile
from thop import clever_format
import torch
from torchstat import stat
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
from basicsr.archs.BSRN_arch import BSRN
from basicsr.archs.BSRN_bd2_gdfn_arch import BSRN_bd2_gdfn
from basicsr.archs.BSRN_bd_arch import BSRN_bd
from basicsr.archs.BSRN_bd_gdfn_arch import BSRN_bd_gdfn
from basicsr.archs.BSRN_gdca_arch import BSRN_gdca

from basicsr.archs.BSRN_gdca2_arch import BSRN_gdca2
from basicsr.archs.MAN_arch import MAN
from basicsr.archs.MAN_gdfn_arch import MAN_gdfn
from basicsr.archs.MAN_gdfn2_arch import MAN_gdfn2
from basicsr.archs.MAN_scpa_arch import MAN_scpa
from basicsr.archs.BSRN_lka_arch import BSRN_lka
from basicsr.archs.BSRN_lka_rlfb_arch import BSRN_lka_rlfb
from basicsr.archs.BSRN_lka_rlfb2_arch import BSRN_lka_rlfb2
from basicsr.archs.BSRN_lka_rlfb2_noca_arch import BSRN_lka_rlfb2_noca
from basicsr.archs.BSRN_lka_rlfb2_d5_arch import BSRN_lka_rlfb2_d5
from basicsr.archs.BSRN_lka_rlfb2_b_arch import BSRN_lka_rlfb2_b
from basicsr.archs.BSRN_lka_rlfb3_arch import BSRN_lka_rlfb3

# 一般的输出尺寸
from basicsr.archs.MDAN_abl_conv_arch import MDAN_abl_conv
from basicsr.archs.MDAN_abl_esa2_cca_arch import MDAN_abl_esa2_cca
from basicsr.archs.MDAN_abl_esa_arch import MDAN_abl_esa
from basicsr.archs.MDAN_abl_esa_cca_arch import MDAN_abl_esa_cca
from basicsr.archs.MDAN_abl_fn_arch import MDAN_abl_fn
from basicsr.archs.MDAN_abl_no_bfn_arch import MDAN_abl_no_bfn
from basicsr.archs.MDAN_abl_no_lrm_arch import MDAN_abl_no_lrm
from basicsr.archs.MDAN_abl_no_mdab_arch import MDAN_abl_no_mdab
from basicsr.archs.MDAN_arch import MDAN
from basicsr.archs.MDAN_block_arch import MDAN_block
from basicsr.archs.MDAN_comp_order_3_arch import MDAN_comp_order_3
from basicsr.archs.MDAN_comp_order_4_arch import MDAN_comp_order_4
from basicsr.archs.MDAN_test_arch import MDAN_test
from basicsr.archs.RFDN_arch import RFDN

input = (3, 48, 48)


def meminfo(model, scale=2):
    # input = (output[0], output[1] // scale, output[2] // scale)
    summary(model, input_size=input, device='cuda')


if __name__ == '__main__':
    # 指定缩放因子
    scale = 4

    # 指定模型
    # model = MAN(scale=scale, n_resblocks=24, n_resgroups=1, n_feats=60)
    # model = BSRN(upscale=scale)
    # model = MDAN(upscale=scale, num_feat=51)
    model = BSRN(upscale=scale)
    # model = RFDN(upscale=scale)
    model = model.to("cuda")

    # 计算复杂度方法1
    meminfo(model, scale)
    # 计算复杂度方法2
    # statCalculate(model,scale)

    # 计算图生成
    # writer = SummaryWriter("logs")
    # input = (1, output[0], output[1] // scale, output[2] // scale)
    # input = torch.ones(input)
    # writer.add_graph(model, input)
    # writer.close()
