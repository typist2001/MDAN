from basicsr.archs.DAT_arch import DAT
from basicsr.archs.DAT_merge_arch import DAT_merge
from basicsr.archs.DAT_merge_ea2_arch import DAT_merge_ea2
from basicsr.archs.DAT_merge_ea3_arch import DAT_merge_ea3
from basicsr.archs.DAT_merge_ea_arch import DAT_merge_ea
from basicsr.archs.MDAN_abl_bfn_3_arch import MDAN_abl_bfn_3
from basicsr.archs.MDAN_abl_bfn_5_arch import MDAN_abl_bfn_5
from basicsr.archs.MDAN_abl_esa2_arch import MDAN_abl_esa2
from basicsr.archs.SCESN_abl_no_sceb_arch import SCESN_abl_no_sceb
from basicsr.archs.SCESN_arch import SCESN
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.esrt_arch import ESRT
from basicsr.archs.scet_arch import SCET
from basicsr.archs.srformer_arch import SRFormer
from thop import profile
from thop import clever_format
import torch
from torchstat import stat
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

output = (3, 1280, 720)


# 计算方法1
def statCalculate(model, scale=2):
    input = (output[0], output[1] // scale, output[2] // scale)
    stat(model, input)


# 计算方法2
def thopCalculate(model, scale=2):
    input = (output[0], output[1] // scale, output[2] // scale)
    x = torch.randn(1, *input)
    flops, params = profile(model, inputs=(x,))
    macs, params = clever_format([flops, params], "%.3f")
    print(f"{params}, {macs}")


def getSRFormer(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'window_size': 16,
        'img_range': 1.,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2,
        'upsampler': 'pixelshuffledirect',
        'resi_connection': '1conv'
    }
    return SRFormer(**now_args)


def getDAT(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [18],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 32],
        'upsampler': 'pixelshuffledirect'
    }
    return DAT(**now_args)

def getDAT_merge(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [12],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 32],
        'upsampler': 'pixelshuffledirect'
    }
    return DAT_merge(**now_args)
def getDAT_merge_ea(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [12],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 32],
        'upsampler': 'pixelshuffledirect'
    }
    return DAT_merge_ea(**now_args)
def getDAT_merge_ea2(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [12],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 32],
        'upsampler': 'pixelshuffledirect'
    }
    return DAT_merge_ea2(**now_args)
def getDAT_merge_ea3(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [12],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 32],
        'upsampler': 'pixelshuffledirect'
    }
    return DAT_merge_ea3(**now_args)
def get_SCESN(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [8],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 8],
        'upsampler': 'pixelshuffledirect'
    }
    return SCESN(**now_args)
def get_SCESN_abl_no_sceb(scale):
    now_args = {
        'upscale': scale,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'depth': [12],
        'embed_dim': 60,
        'num_heads': [6],
        'expansion_factor': 2,
        'resi_connection': '3conv',
        'split_size': [8, 32],
        'upsampler': 'pixelshuffledirect'
    }
    return SCESN_abl_no_sceb(**now_args)
if __name__ == '__main__':
    # 指定缩放因子
    scale = 2

    # 指定模型
    # model = MAN(scale=scale, n_resblocks=24, n_resgroups=1, n_feats=60)
    # model = BSRN(upscale=scale)
    # model = get_SCESN_abl_no_sceb(scale)
    model = get_SCESN(scale)
    # model = EDSR(upscale=scale)
    # model = SMSR(upscale=scale)

    # 计算复杂度方法1
    thopCalculate(model, scale)
    # 计算复杂度方法2
    # statCalculate(model,scale)

    # 计算图生成
    # writer = SummaryWriter("logs")
    # input = (1, output[0], output[1] // scale, output[2] // scale)
    # input = torch.ones(input)
    # writer.add_graph(model, input)
    # writer.close()
