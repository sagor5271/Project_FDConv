# fdconv.py
# Frequency Domain Convolution (FDConv) - single-file implementation
# Dependencies: PyTorch >= 1.10

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =============================================================================
# Custom Activation Functions
# =============================================================================
class StarReLU(nn.Module):
    """StarReLU activation function: s * relu(x)^2 + b"""

    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        scale_learnable: bool = True,
        bias_learnable: bool = True,
        mode: Optional[str] = None,
        inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x) ** 2 + self.bias


# =============================================================================
# Utility
# =============================================================================
def get_fft2freq(d1: int, d2: int, use_rfft: bool = False):
    """
    Generate 2D frequency coordinates and sort them by distance from origin.
    Returns:
        sorted_coords (2, N): indices for (d1, d2//2+1 if rfft else d2), flattened in row-major
        freq_hw (d1, d2 or d2//2+1, 2): frequency grid
    """
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)  # (d1, d2|d2//2+1, 2)
    dist = torch.norm(freq_hw, dim=-1)  # (d1, d2|d2//2+1)
    _, indices = torch.sort(dist.reshape(-1))
    if use_rfft:
        d2_eff = d2 // 2 + 1
        sorted_coords = torch.stack([indices // d2_eff, indices % d2_eff], dim=-1)  # (N, 2)
    else:
        sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)  # (N, 2)
    return sorted_coords.permute(1, 0), freq_hw  # (2, N), (d1, d2|d2//2+1, 2)


# =============================================================================
# Global Kernel and Spatial Modulation
# =============================================================================
class KernelSpatialModulation_Global(nn.Module):
    """
    Global Kernel and Spatial Modulation Module
    Produces channel/filter/spatial/kernel attentions from pooled context.
    """

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
        kernel_att_init: Optional[str] = None,
        att_multi: float = 1.0,
        ksm_only_kernel_att: bool = False,
        att_grid: int = 1,
        stride: int = 1,
        spatial_freq_decompose: bool = False,
        act_type: str = 'sigmoid',
    ):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = max(1e-6, temp)
        self.kernel_temp = max(1e-6, kernel_temp if kernel_temp is not None else self.temperature)
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi
        self.att_grid = att_grid
        self.spatial_freq_decompose = spatial_freq_decompose

        # GAP for context
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Shared trunk
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        # Channel attention
        if ksm_only_kernel_att:
            self.func_channel = self._skip
        else:
            if spatial_freq_decompose:
                out_c = in_planes * 2 if kernel_size > 1 else in_planes
            else:
                out_c = in_planes
            self.channel_fc = nn.Conv2d(attention_channel, out_c, 1, bias=True)
            self.func_channel = self._get_channel_attention

        # Filter attention
        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:
            self.func_filter = self._skip
        else:
            out_f = out_planes * 2 if spatial_freq_decompose else out_planes
            self.filter_fc = nn.Conv2d(attention_channel, out_f, 1, stride=stride, bias=True)
            self.func_filter = self._get_filter_attention

        # Spatial attention
        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self._skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self._get_spatial_attention

        # Kernel attention
        if kernel_num == 1:
            self.func_kernel = self._skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self._get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if hasattr(self, 'spatial_fc'):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
        if hasattr(self, 'kernel_fc'):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
        if hasattr(self, 'channel_fc'):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)
        if hasattr(self, 'filter_fc'):
            nn.init.normal_(self.filter_fc.weight, std=1e-6)

    def update_temperature(self, temperature: float):
        self.temperature = max(1e-6, temperature)

    @staticmethod
    def _skip(_x):
        return 1.0

    def _get_channel_attention(self, x: torch.Tensor):
        # x is (B, C_att, 1, 1)
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
        out = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)  # (B, K,1,1,1,1)
        if self.act_type == 'softmax':
            out = F.softmax(out / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            out = torch.sigmoid(out / self.kernel_temp) * 2.0 / out.size(1)
        elif self.act_type == 'tanh':
            out = (1.0 + torch.tanh(out / self.kernel_temp)) / out.size(1)
        else:
            raise NotImplementedError
        return out

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False):
        # x expected (B, C, H, W)
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x: torch.Tensor):
        # Use global context
        avg_x = self.avgpool(x)
        avg_x = self.relu(self.bn(self.fc(avg_x)))
        return (
            self.func_channel(avg_x),
            self.func_filter(avg_x),
            self.func_spatial(avg_x),
            self.func_kernel(avg_x),
        )


# =============================================================================
# Local Kernel and Spatial Modulation
# =============================================================================
class KernelSpatialModulation_Local(nn.Module):
    """
    Local attention via 1D conv over channels (optionally frequency-enhanced).
    Returns logits shaped (B, kernel_num, C, out_n).
    """

    def __init__(self, channel: Optional[int] = None, kernel_num: int = 1, out_n: int = 1, k_size: int = 3, use_global: bool = False):
        super().__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        self.use_global = use_global
        if channel is not None:
            k_size = int(round((math.log2(channel) / 2) + 0.5))
            k_size = (k_size // 2) * 2 + 1  # make odd
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        if self.use_global:
            self.complex_weight = nn.Parameter(
                torch.randn(1, (self.channel // 2) + 1, 2, dtype=torch.float32) * 1e-6
            )
            self.norm = nn.LayerNorm(self.channel)

    def forward(self, x: torch.Tensor, x_std: Optional[torch.Tensor] = None):
        # x: (B, C, 1, 1)
        x = x.squeeze(-1).transpose(-1, -2)  # (B, C, 1) -> (B, 1, C)
        b, _, c = x.shape
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)  # (B,1,C//2+1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
            x = self.norm(x)
        att_logit = self.conv(x)  # (B, kn*out_n, C)
        att_logit = att_logit.reshape(b, self.kn, self.out_n, c)  # (B, kn, out_n, C)
        att_logit = att_logit.permute(0, 1, 3, 2)  # (B, kn, C, out_n)
        return att_logit


# =============================================================================
# Frequency Band Modulation
# =============================================================================
class FrequencyBandModulation(nn.Module):
    """Decompose features into bands via RFFT2 and apply per-band spatial attention."""

    def __init__(
        self,
        in_channels: int,
        k_list: List[int] = (2,),
        lowfreq_att: bool = False,
        fs_feat: str = 'feat',
        act: str = 'sigmoid',
        spatial: str = 'conv',
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
        self.fs_feat = fs_feat
        self.act = act
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group
        if spatial == 'conv':
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
        else:
            raise NotImplementedError

        # Precompute masks buffer at max_size
        self.register_buffer('cached_masks', self._precompute_masks(max_size, self.k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        max_h, max_w = max_size
        _, freq_hw = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)  # (max_h, max_w//2+1, 2)
        # radius metric (Chebyshev-like using max abs over (fx, fy))
        freq_indices = freq_hw.abs().max(dim=-1).values  # (max_h, max_w//2+1)
        masks = []
        for freq in k_list:
            # threshold band by radius
            mask = (freq_indices < (0.5 / float(freq) + 1e-8))
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)  # (num_masks,1,max_h,max_w//2+1)

    def _sp_act(self, freq_weight: torch.Tensor):
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2.0
        elif self.act == 'tanh':
            return 1.0 + freq_weight.tanh()
        elif self.act == 'softmax':
            # softmax over spatial group channel dimension
            return F.softmax(freq_weight, dim=1)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, att_feat: Optional[torch.Tensor] = None):
        # x: (B,C,H,W)
        if att_feat is None:
            att_feat = x
        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, c, h, w = x.shape

        # FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')  # (B,C,H,W//2+1)

        # Resize cached masks
        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        # Per-band processing
        for idx, _freq in enumerate(self.k_list):
            mask = current_masks[idx]  # (1,1,H,W//2+1), broadcastable
            low_part_fft = x_fft * mask  # masked FFT
            low_part = torch.fft.irfft2(low_part_fft, s=(h, w), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part

            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self._sp_act(freq_weight)

            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))

        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self._sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)

        return sum(x_list)


# =============================================================================
# Main FDConv Implementation
# =============================================================================
class FDConv(nn.Conv2d):
    """Frequency Domain Convolution Layer (drop-in for Conv2d when enabled)."""

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
        att_grid: int = 1,
        use_ksm_local: bool = True,
        ksm_local_act: str = 'sigmoid',
        ksm_global_act: str = 'sigmoid',
        spatial_freq_decompose: bool = False,
        convert_param: bool = True,
        linear_mode: bool = False,
        fbm_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = list(use_fdconv_if_k_in)
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = float(param_reduction)
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = list(use_fbm_if_k_in)
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        self.linear_mode = linear_mode

        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2

        # Reasonable temps
        base_temp = math.sqrt(max(1, self.kernel_num * self.param_ratio))
        kernel_temp = base_temp if kernel_temp is None else kernel_temp
        temp = base_temp if temp is None else temp

        # Scale alpha (kept for continuity)
        self.alpha = (min(self.out_channels, self.in_channels) // 2) * self.kernel_num * self.param_ratio / max(1e-6, self.param_reduction)

        # Early exit: fallback to standard Conv2d if not eligible
        if (min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt) or \
           (self.kernel_size not in self.use_fdconv_if_k_in):
            # nothing else to initialize
            self._fd_enabled = False
            return

        self._fd_enabled = True

        # Global KSM
        self.KSM_Global = KernelSpatialModulation_Global(
            self.in_channels, self.out_channels, self.kernel_size,
            groups=self.groups, temp=temp, kernel_temp=kernel_temp, reduction=reduction,
            kernel_num=self.kernel_num * self.param_ratio, kernel_att_init=None,
            att_multi=att_multi, ksm_only_kernel_att=ksm_only_kernel_att,
            act_type=self.ksm_global_act, att_grid=att_grid, stride=self.stride,
            spatial_freq_decompose=spatial_freq_decompose
        )

        # FBM if needed (for input feature modulation)
        if self.kernel_size in self.use_fbm_if_k_in:
            if fbm_cfg is None:
                fbm_cfg = dict(
                    k_list=[2, 4, 8],
                    lowfreq_att=False,
                    fs_feat='feat',
                    act='sigmoid',
                    spatial='conv',
                    spatial_group=1,
                    spatial_kernel=3,
                    init='zero',
                )
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)

        # Local KSM
        if self.use_ksm_local:
            kH = kW = self.kernel_size
            self.KSM_Local = KernelSpatialModulation_Local(
                channel=self.in_channels, kernel_num=1, out_n=int(self.out_channels * kH * kW)
            )
        else:
            self.KSM_Local = None

        # Convert conv weights to frequency parameterization if requested
        self._convert2dftweight(convert_param)

    def _convert2dftweight(self, convert_param: bool):
        if not getattr(self, '_fd_enabled', False):
            return
        d1, d2 = self.out_channels, self.in_channels
        kH = kW = self.kernel_size

        # Prepare current conv weight (OC, IC, kH, kW) -> (OC*kH, IC*kW)
        w = self.weight.permute(0, 2, 1, 3).reshape(d1 * kH, d2 * kW)
        w_rfft = torch.fft.rfft2(w, dim=(0, 1))  # (d1*kH, d2*kW//2+1)
        w_rfft = torch.stack([w_rfft.real, w_rfft.imag], dim=-1)  # (..., 2)

        if convert_param:
            # Learnable frequency weights
            self.dft_weight = nn.Parameter(w_rfft[None, ...].repeat(self.param_ratio, 1, 1, 1), requires_grad=True)
            # Remove spatial weight param to avoid using it
            self.register_parameter('weight', None)
        else:
            # keep spatial weight as is
            self.dft_weight = None

    def _weight_from_dft(self) -> torch.Tensor:
        # Reconstruct spatial weights from DFT weights
        d1, d2 = self.out_channels, self.in_channels
        kH = kW = self.kernel_size
        # dft_weight: (R, d1*kH, d2*kW//2+1, 2) -> take the first ratio slice
        
        # 🐛 SYNTAX ERROR FIXED
        w_rfft = self.dft_weight[0] 
        
        w_complex = torch.view_as_complex(w_rfft)
        w_spatial = torch.fft.irfft2(w_complex, s=(d1 * kH, d2 * kW))
        w_spatial = w_spatial.reshape(d1, kH, d2, kW).permute(0, 2, 1, 3).contiguous()  # (OC,IC,kH,kW)
        return w_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to standard conv if not enabled by conditions
        if not getattr(self, '_fd_enabled', False):
            return super().forward(x)

        b, in_planes, h, w = x.size()

        # Global attentions
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        # Shapes:
        # channel_attention: (B,1,1,Cin( or 2Cin),1,1)
        # filter_attention:  (B,1,Cout,1,1,1)
        # spatial_attention: (B,1,1,1,kH,kW)
        # kernel_attention:  (B,K,1,1,1,1) (unused in this simplified weight path)

        # Local attentions
        if self.use_ksm_local and self.KSM_Local is not None:
            hr_att_logit = self.KSM_Local(global_x)  # (B,1,Cin,out_n)
            kH = kW = self.kernel_size
            out_n = self.out_channels * kH * kW
            hr_att_logit = hr_att_logit.reshape(b, 1, self.in_channels, out_n)
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2)  # (B,1,out_n,Cin)
            hr_att_logit = hr_att_logit.reshape(b, 1, self.out_channels, kH, kW, self.in_channels)
            hr_att_logit = hr_att_logit.permute(0, 1, 2, 5, 3, 4)  # (B,1,OC,IC,kH,kW)
            if self.ksm_local_act == 'sigmoid':
                hr_att = torch.sigmoid(hr_att_logit) * self.att_multi
            else:
                hr_att = 1.0 + torch.tanh(hr_att_logit)
        else:
            hr_att = 1.0

        # Base spatial weights
        if self.dft_weight is not None:
            base_weight = self._weight_from_dft()  # (OC,IC,kH,kW)
        else:
            base_weight = self.weight  # (OC,IC,kH,kW)

        # Optional FBM on inputs
        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        # Aggregate all attentions into weight
        kH = kW = self.kernel_size
        aggregate_weight = base_weight.unsqueeze(0).unsqueeze(1)  # (1,1,OC,IC,kH,kW)
        if not isinstance(spatial_attention, float):
            aggregate_weight = aggregate_weight * spatial_attention  # (B,1,OC,IC,kH,kW)
        if not isinstance(channel_attention, float):
            aggregate_weight = aggregate_weight * channel_attention  # (B,1,OC,IC,kH,kW)
        if not isinstance(filter_attention, float):
            aggregate_weight = aggregate_weight * filter_attention  # (B,1,OC,IC,kH,kW)
        if not isinstance(hr_att, float):
            aggregate_weight = aggregate_weight * hr_att  # (B,1,OC,IC,kH,kW)

        aggregate_weight = torch.sum(aggregate_weight, dim=1)  # (B,OC,IC,kH,kW)

        # Batched grouped convolution
        aggregate_weight = aggregate_weight.view(-1, self.in_channels // self.groups, kH, kW)
        x_ = x.reshape(1, -1, h, w)
        out = F.conv2d(
            x_,
            weight=aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * b,
        )
        out = out.view(b, self.out_channels, out.size(-2), out.size(-1))

        # Apply filter attention post-conv if we skipped above (kept for compatibility)
        # (here already applied in weight path)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out



