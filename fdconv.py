# fdconv.py
# Frequency Domain Convolution (FDConv) - minimal, clean, PyTorch-only
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class StarReLU(nn.Module):
    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        scale_learnable: bool = True,
        bias_learnable: bool = True,
        inplace: bool = False,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.ones(1) * scale_value, requires_grad=scale_learnable)
        self.bias = nn.Parameter(torch.ones(1) * bias_value, requires_grad=bias_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x) ** 2 + self.bias


def get_fft2freq(d1: int, d2: int, use_rfft: bool = False):
    freq_h = torch.fft.fftfreq(d1)
    freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)  # (d1, d2|d2//2+1, 2)
    dist = torch.norm(freq_hw, dim=-1)
    _, indices = torch.sort(dist.reshape(-1))
    if use_rfft:
        d2_eff = d2 // 2 + 1
        sorted_coords = torch.stack([indices // d2_eff, indices % d2_eff], dim=-1)
    else:
        sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    return sorted_coords.permute(1, 0), freq_hw


class KernelSpatialModulation_Global(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 4,
        min_channel: int = 16,
        temp: float = 1.0,
        kernel_temp: Optional[float] = None,
        att_multi: float = 1.0,
        ksm_only_kernel_att: bool = False,
        spatial_freq_decompose: bool = False,
        act_type: str = 'sigmoid',
        stride: int = 1,
        att_grid: int = 1,
    ):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = int(kernel_size)
        self.kernel_num = int(kernel_num)
        self.temperature = max(1e-6, float(temp))
        self.kernel_temp = max(1e-6, float(kernel_temp if kernel_temp is not None else self.temperature))
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.att_multi = float(att_multi)
        self.spatial_freq_decompose = spatial_freq_decompose

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        # Channel att
        if ksm_only_kernel_att:
            self.func_channel = self._skip
        else:
            out_c = in_planes * 2 if (spatial_freq_decompose and self.kernel_size > 1) else in_planes
            self.channel_fc = nn.Conv2d(attention_channel, out_c, 1, bias=True)
            self.func_channel = self._get_channel_attention

        # Filter att
        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:
            self.func_filter = self._skip
        else:
            out_f = out_planes * 2 if spatial_freq_decompose else out_planes
            self.filter_fc = nn.Conv2d(attention_channel, out_f, 1, stride=stride, bias=True)
            self.func_filter = self._get_filter_attention

        # Spatial att
        if self.kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self._skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, self.kernel_size * self.kernel_size, 1, bias=True)
            self.func_spatial = self._get_spatial_attention

        # Kernel att
        if self.kernel_num == 1:
            self.func_kernel = self._skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, self.kernel_num, 1, bias=True)
            self.func_kernel = self._get_kernel_attention

        self._init_weights()

    def _init_weights(self): # 🔥 সিনট্যাক্স ঠিক করা হলো
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for opt in ['spatial_fc', 'kernel_fc', 'channel_fc', 'filter_fc']:
            m = getattr(self, opt, None)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=1e-6)

    @staticmethod
    def _skip(_x):
        return 1.0

    def _get_channel_attention(self, x: torch.Tensor):
        if self.act_type == 'sigmoid':
            out = torch.sigmoid(self.channel_fc(x) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            out = 1 + torch.tanh(self.channel_fc(x) / self.temperature)
        else:
            raise NotImplementedError
        B = x.size(0)
        return out.view(B, 1, 1, -1, 1, 1)

    def _get_filter_attention(self, x: torch.Tensor):
        if self.act_type == 'sigmoid':
            out = torch.sigmoid(self.filter_fc(x) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            out = 1 + torch.tanh(self.filter_fc(x) / self.temperature)
        else:
            raise NotImplementedError
        B = x.size(0)
        return out.view(B, 1, -1, 1, 1, 1)

    def _get_spatial_attention(self, x: torch.Tensor):
        out = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            out = torch.sigmoid(out / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            out = 1 + torch.tanh(out / self.temperature)
        else:
            raise NotImplementedError
        return out

    def _get_kernel_attention(self, x: torch.Tensor):
        out = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(out / self.kernel_temp, dim=1)

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False):
        if use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x: torch.Tensor):
        avg_x = self.relu(self.bn(self.fc(self.avgpool(x))))
        return (
            self.func_channel(avg_x),
            self.func_filter(avg_x),
            self.func_spatial(avg_x),
            self.func_kernel(avg_x),
        )


class KernelSpatialModulation_Local(nn.Module):
    def __init__(self, channel: Optional[int] = None, kernel_num: int = 1, out_n: int = 1, k_size: int = 3, use_global: bool = False):
        super().__init__()
        self.kn = int(kernel_num)
        self.out_n = int(out_n)
        self.channel = channel
        self.use_global = use_global
        if channel is not None:
            k_size = int(round((math.log2(channel) / 2) + 0.5))
            k_size = (k_size // 2) * 2 + 1
        self.conv = nn.Conv1d(1, self.kn * self.out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        if self.use_global and channel is not None:
            self.complex_weight = nn.Parameter(torch.randn(1, (channel // 2) + 1, 2, dtype=torch.float32) * 1e-6)
            self.norm = nn.LayerNorm(channel)

    def forward(self, x: torch.Tensor, x_std: Optional[torch.Tensor] = None):
        x = x.squeeze(-1).transpose(-1, -2)  # (B,C,1,1)->(B,1,C)
        b, _, c = x.shape
        if getattr(self, 'complex_weight', None) is not None:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
            x = self.norm(x)
        att_logit = self.conv(x)  # (B, kn*out_n, C)
        att_logit = att_logit.reshape(b, self.kn, self.out_n, c).permute(0, 1, 3, 2)  # (B, kn, C, out_n)
        return att_logit


class FrequencyBandModulation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        k_list: List[int] = (2,),
        lowfreq_att: bool = False,
        act: str = 'sigmoid',
        spatial_group: int = 1,
        spatial_kernel: int = 3,
        init: str = 'zero',
        max_size: Tuple[int, int] = (64, 64),
        **kwargs,
    ):
        super().__init__()
        self.k_list = list(k_list)
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.act = act
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group
        self.freq_weight_conv_list = nn.ModuleList()
        n = len(self.k_list) + (1 if lowfreq_att else 0)
        for _ in range(n):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.spatial_group,
                stride=1,
                kernel_size=spatial_kernel,
                groups=self.spatial_group,
                padding=spatial_kernel // 2,
                bias=True,
            )
            if init == 'zero':
                nn.init.normal_(conv.weight, std=1e-6)
                if conv.bias is not None:
                    conv.bias.data.zero_()
            self.freq_weight_conv_list.append(conv)
        self.register_buffer('cached_masks', self._precompute_masks(max_size, self.k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        max_h, max_w = max_size
        _, freq_hw = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        freq_indices = freq_hw.abs().max(dim=-1).values  # (max_h, max_w//2+1)
        masks = []
        for freq in k_list:
            mask = (freq_indices < (0.5 / float(freq) + 1e-8))
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)  # (num_masks,1,max_h,max_w//2+1)

    def _sp_act(self, t: torch.Tensor):
        if self.act == 'sigmoid':
            return t.sigmoid() * 2.0
        if self.act == 'tanh':
            return 1.0 + t.tanh()
        if self.act == 'softmax':
            return F.softmax(t, dim=1)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, att_feat: Optional[torch.Tensor] = None):
        if att_feat is None:
            att_feat = x
        b, c, h, w = x.shape
        x = x.to(torch.float32)
        pre_x = x.clone()
        x_fft = torch.fft.rfft2(x, norm='ortho')
        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')
        x_list = []
        for idx, _freq in enumerate(self.k_list):
            mask = current_masks[idx]
            low_fft = x_fft * mask
            low_part = torch.fft.irfft2(low_fft, s=(h, w), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part
            fw = self.freq_weight_conv_list[idx](att_feat)
            fw = self._sp_act(fw)
            tmp = fw.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        if self.lowfreq_att:
            fw = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            fw = self._sp_act(fw)
            tmp = fw.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
        return sum(x_list)


class FDConv(nn.Conv2d):
    """
    Drop-in replacement for nn.Conv2d. For practical stability:
    - Build base spatial weights from learnable DFT weights (optional).
    - Apply global/local/spatial attentions to form batch-conditioned weights.
    """
    def __init__(
        self,
        *args,
        reduction: float = 0.0625,
        kernel_num: Optional[int] = 4,
        use_fdconv_if_c_gt: int = 16,
        use_fdconv_if_k_in: List[int] = (1, 3),
        use_fbm_if_k_in: List[int] = (3,),
        kernel_temp: Optional[float] = None,
        temp: Optional[float] = None,
        att_multi: float = 1.0,
        param_ratio: int = 1,
        param_reduction: float = 1.0,
        ksm_only_kernel_att: bool = False,
        use_ksm_local: bool = True,
        ksm_local_act: str = 'sigmoid',
        ksm_global_act: str = 'sigmoid',
        spatial_freq_decompose: bool = False,
        convert_param: bool = True,
        fbm_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = int(use_fdconv_if_c_gt)
        self.use_fdconv_if_k_in = list(use_fdconv_if_k_in)
        self.use_fbm_if_k_in = list(use_fbm_if_k_in)
        self.kernel_num = kernel_num if kernel_num is not None else max(1, self.out_channels // 2)
        self.param_ratio = int(param_ratio)
        self.param_reduction = float(param_reduction)
        self.use_ksm_local = bool(use_ksm_local)
        self.att_multi = float(att_multi)
        self.spatial_freq_decompose = bool(spatial_freq_decompose)
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act

        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        base_temp = math.sqrt(max(1, self.kernel_num * self.param_ratio))
        kernel_temp = base_temp if kernel_temp is None else kernel_temp
        temp = base_temp if temp is None else temp
        self.alpha = (min(self.out_channels, self.in_channels) // 2) * self.kernel_num * self.param_ratio / max(1e-6, self.param_reduction)

        # Enable only if shape conditions match
        if (min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt) or \
           (self.kernel_size not in self.use_fdconv_if_k_in):
            self._fd_enabled = False
            self.dft_weight = None
            return
        self._fd_enabled = True

        # Global KSM
        self.KSM_Global = KernelSpatialModulation_Global(
            self.in_channels, self.out_channels, int(self.kernel_size),
            groups=self.groups, temp=temp, kernel_temp=kernel_temp, reduction=reduction,
            kernel_num=self.kernel_num, ksm_only_kernel_att=ksm_only_kernel_att,
            att_multi=self.att_multi, spatial_freq_decompose=self.spatial_freq_decompose,
            stride=self.stride if isinstance(self.stride, int) else self.stride,
        )

        # Local KSM
        if self.use_ksm_local:
            kH = kW = int(self.kernel_size)
            self.KSM_Local = KernelSpatialModulation_Local(
                channel=self.in_channels, kernel_num=1, out_n=int(self.out_channels * kH * kW)
            )
        else:
            self.KSM_Local = None

        # Optional FBM on inputs
        if int(self.kernel_size) in self.use_fbm_if_k_in:
            if fbm_cfg is None:
                fbm_cfg = dict(k_list=[2, 4, 8], lowfreq_att=False, act='sigmoid', spatial_group=1, spatial_kernel=3, init='zero')
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)

        self._convert2dftweight(convert_param)

    def _convert2dftweight(self, convert_param: bool):
        if not self._fd_enabled:
            self.dft_weight = None
            return
        d1, d2 = self.out_channels, self.in_channels
        kH = kW = int(self.kernel_size)
        w = self.weight.permute(0, 2, 1, 3).reshape(d1 * kH, d2 * kW)  # (d1*kH, d2*kW)
        w_rfft = torch.fft.rfft2(w, dim=(0, 1))  # (d1*kH, d2*kW//2+1)
        w_rfft = torch.stack([w_rfft.real, w_rfft.imag], dim=-1)  # (..., 2)
        if convert_param:
            self.dft_weight = nn.Parameter(w_rfft[None, ...].repeat(self.param_ratio, 1, 1, 1), requires_grad=True)
            # remove spatial weight param so only dft_weight is trainable
            self.register_parameter('weight', None)
        else:
            self.dft_weight = None  # keep spatial weight as is

    def _weight_from_dft(self) -> torch.Tensor:
        d1, d2 = self.out_channels, self.in_channels
        kH = kW = int(self.kernel_size)
        w_rfft = self.dft_weight 
        w_complex = torch.view_as_complex(w_rfft)
        w_spatial = torch.fft.irfft2(w_complex, s=(d1 * kH, d2 * kW))
        w_spatial = w_spatial.reshape(d1, kH, d2, kW).permute(0, 2, 1, 3).contiguous()  # (OC,IC,kH,kW)
        return w_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fd_enabled:
            return super().forward(x)

        b, _, h, w = x.size()
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_att, filter_att, spatial_att, kernel_att = self.KSM_Global(global_x)

        if self.use_ksm_local and self.KSM_Local is not None:
            hr_logit = self.KSM_Local(global_x)  # (B,1,Cin,out_n)
            kH = kW = int(self.kernel_size)
            out_n = self.out_channels * kH * kW
            hr_logit = hr_logit.reshape(b, 1, self.in_channels, out_n).permute(0, 1, 3, 2)
            hr_logit = hr_logit.reshape(b, 1, self.out_channels, kH, kW, self.in_channels).permute(0, 1, 2, 5, 3, 4)
            if self.ksm_local_act == 'sigmoid':
                hr_att = torch.sigmoid(hr_logit) * self.att_multi
            else:
                hr_att = 1.0 + torch.tanh(hr_logit)
        else:
            hr_att = 1.0

        base_weight = self._weight_from_dft() if self.dft_weight is not None else self.weight  # (OC,IC,kH,kW)

        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        kH = kW = int(self.kernel_size)
        agg_w = base_weight.unsqueeze(0).unsqueeze(1)  # (1,1,OC,IC,kH,kW)
        if not isinstance(spatial_att, float):
            agg_w = agg_w * spatial_att
        if not isinstance(channel_att, float):
            agg_w = agg_w * channel_att
        if not isinstance(filter_att, float):
            agg_w = agg_w * filter_att
        if not isinstance(hr_att, float):
            agg_w = agg_w * hr_att
        agg_w = torch.sum(agg_w, dim=1)  # (B,OC,IC,kH,kW)

        agg_w = agg_w.view(-1, self.in_channels // self.groups, kH, kW)
        x_ = x.reshape(1, -1, h, w)
        out = F.conv2d(
            x_,
            weight=agg_w,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * b,
        )
        out = out.view(b, self.out_channels, out.size(-2), out.size(-1))
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.rand(2, 64, 32, 32)
    m = FDConv(in_channels=64, out_channels=64, kernel_size=3, padding=1, kernel_num=8, bias=True)
    y = m(x)
    print('FDConv OK:', tuple(y.shape))
