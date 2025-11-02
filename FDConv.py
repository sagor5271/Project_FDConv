import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from torch.utils.checkpoint import checkpoint

from torch import Tensor
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
import time

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
class KernelSpatialModulation_Global(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, 
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=1.0, # 🔥 সংশোধিত: att_multi=1.0 (স্থিতিশীল)
                 ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.temperature = temp
        self.kernel_temp = kernel_temp
        
        self.ksm_only_kernel_att = ksm_only_kernel_att

        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU() 

        self.spatial_freq_decompose = spatial_freq_decompose

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1, bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att: # depth-wise convolution
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or self.ksm_only_kernel_att: # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 🔥 Kernel_fc bias zero initialization for stability
        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
            if self.kernel_fc.bias is not None:
                nn.init.constant_(self.kernel_fc.bias, 0)
        
        # 🔥 Channel_fc bias zero initialization
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)
            if self.channel_fc.bias is not None:
                nn.init.constant_(self.channel_fc.bias, 0)
        
        # 🔥 Filter_fc bias zero initialization
        if hasattr(self, 'filter_fc') and isinstance(self.filter_fc, nn.Conv2d):
            nn.init.normal_(self.filter_fc.weight, std=1e-6)
            if self.filter_fc.bias is not None:
                nn.init.constant_(self.filter_fc.bias, 0)
        
        # 🔥 Spatial_fc bias zero initialization
        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
            if self.spatial_fc.bias is not None:
                nn.init.constant_(self.spatial_fc.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        # att_multi=1.0 is used for better stability with sigmoid/tanh
        if self.act_type =='sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi 
        elif self.act_type =='tanh':
            channel_attention = 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) 
        else:
            raise NotImplementedError
        return channel_attention

    def get_filter_attention(self, x):
        # att_multi=1.0 is used for better stability with sigmoid/tanh
        if self.act_type =='sigmoid':
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi 
        elif self.act_type =='tanh':
            filter_attention = 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) 
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) 
        if self.act_type =='sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type =='tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        
        # 🔥 CRITICAL FIX: Always use Softmax for kernel weights to ensure they sum to 1.
        # This is essential for proper weight aggregation in dynamic convolution.
        if self.kernel_temp is None:
            kernel_temp = 1.0
        else:
            kernel_temp = self.kernel_temp
            
        kernel_attention = F.softmax(kernel_attention / kernel_temp, dim=1)
        
        # Removed the unstable sigmoid/tanh options for kernel attention:
        # if self.act_type =='softmax': ...
        # elif self.act_type =='sigmoid': ... (removed)
        # elif self.act_type =='tanh': ... (removed)
        
        return kernel_attention
    
    def forward(self, x, use_checkpoint=False):
        # Assuming 'checkpoint' is available in your training environment
        if use_checkpoint and 'checkpoint' in globals(): 
             return checkpoint(self._forward, x)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)
        
class KernelSpatialModulation_Local(nn.Module):
    """
    Constructs a Local Channel Modulation/Attention module, similar to ECA, 
    but designed to output multiple logits (kernel_num * out_n) for dynamic convolution.
    """
    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        
        # 1. k_size Calculation (Adaptive Kernel Size, ECA style)
        if channel is not None: 
            # Recalculate k_size to ensure it's an odd number for symmetrical padding
            # Original formula: k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
            # Simple k_size based on log(C) used in ECA:
            b = 1 # The base factor used in some ECA implementations
            t = int(abs(math.log2(channel) + b) / 2.) 
            k_size = t if t % 2 else t + 1 # k_size must be odd
        
        self.k_size = k_size
        
        # 2. 1D Convolution for Channel Attention (ECA's core)
        # Input: (B, 1, C), Output: (B, kn * out_n, C)
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=self.k_size, 
                              padding=(self.k_size - 1) // 2, bias=False) 
        
        # 🔥 CRITICAL FIX: Initialize weight to 0 for stability (starts as identity mapping)
        nn.init.constant_(self.conv.weight, 0.0) 
        
        self.use_global = use_global
        
        # 3. Frequency Domain Modulation (if use_global=True)
        if self.use_global:
            # Complex weight initialized close to zero for minimal initial frequency modulation
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1 , 2, dtype=torch.float32) * 1e-6)
            self.norm = nn.LayerNorm(self.channel)
        
    def forward(self, x, x_std=None):
        # 1. Apply Global Average Pooling implicitly on H and W (x should be B, C, 1, 1 or B, C)
        # We assume input 'x' is already a globally pooled vector of shape (B, C, 1, 1).
        
        # Squeeze the spatial dimensions (1, 1) and transpose (B, C) -> (B, 1, C)
        # x.size(-2) and x.size(-1) in original code suggest input is B, C, H, W
        if x.dim() == 4:
            x = x.mean(dim=[2, 3], keepdim=False) # Global Average Pool 
        
        # Reshape for Conv1D: (B, C) -> (B, 1, C)
        x = x.unsqueeze(1) 
        b, _, c = x.shape
        
        # 2. Frequency Domain Modulation (if enabled)
        if self.use_global:
            # Type casting to float is often required for FFT
            x_rfft = torch.fft.rfft(x.float(), dim=-1) 
            
            # Complex multiplication is essentially: 
            # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
            
            # Simplified (as in original code, which is complex multiplication if complex_weight is real/imag)
            # The original code's approach is slightly simplified/non-standard complex mult.
            x_real = x_rfft.real * self.complex_weight[..., 0] 
            x_imag = x_rfft.imag * self.complex_weight[..., 1] 
            
            # Inverse FFT and skip connection (Residual connection in frequency domain)
            x_modulated = torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
            x = x + x_modulated.unsqueeze(1) # B, 1, C
        
        # 3. Normalization
        if self.use_global:
            # LayerNorm is applied to the C dimension: (B, 1, C) -> (B, 1, C)
            x = self.norm(x)
            
        # 4. 1D Convolution (Local Channel Interaction)
        # Input: (B, 1, C) -> Output: (B, kn * out_n, C)
        att_logit = self.conv(x)
        
        # 5. Reshape Output
        # (B, kn * out_n, C) -> (B, kn, out_n, C) -> (B, kn, C, out_n)
        att_logit = att_logit.reshape(b, self.kn, self.out_n, c) 
        att_logit = att_logit.permute(0, 1, 3, 2) # B, kn, C, out_n
        
        # 🔥 CRITICAL FIX: The output should usually be normalized or activated for use.
        # Since this module only produces *logits*, we won't add activation here, 
        # but the subsequent module (the dynamic conv layer) should apply Softmax or Sigmoid.
        
        return att_logit
def get_fft2freq(d1, d2, use_rfft=False):
    # Frequency components for rows and columns
    freq_h = torch.fft.fftfreq(d1)  # Frequency for the rows (d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)  # Frequency for the columns (d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    
    # 🔥 CRITICAL FIX 1: Specify indexing='ij' for PyTorch meshgrid
    # Ensures that freq_h maps to dimension 0 (rows) and freq_w maps to dimension 1 (columns)
    freq_hw_grids = torch.meshgrid(freq_h, freq_w, indexing='ij') 
    
    # Meshgrid to create a 2D grid of frequency coordinates
    freq_hw = torch.stack(freq_hw_grids, dim=-1) # Shape: (d1, d2) or (d1, d2//2 + 1), 2
    
    # Calculate the distance from the origin (0, 0) in the frequency space
    dist = torch.norm(freq_hw, dim=-1)
    # 
    
    # Sort the distances and get the indices
    sorted_dist, indices = torch.sort(dist.view(-1)) # Flatten the distance tensor for sorting
    
    # 🔥 CRITICAL FIX 2: Correctly determine the width (d2_used) for index conversion
    # d2 must be the *current* width of the 2D tensor (dist).
    d2_used = d2 // 2 + 1 if use_rfft else d2
    
    # Get the corresponding coordinates for the sorted distances
    # indices // d2_used gives the row index, indices % d2_used gives the column index
    sorted_coords = torch.stack([indices // d2_used, indices % d2_used], dim=-1) 
    
    # Output shape: (2, N) where N is the total number of frequency bins
    return sorted_coords.permute(1, 0), freq_hw
class FDConv(nn.Conv2d):
    def __init__(self, *args, 
                 reduction=0.0625, kernel_num=4, use_fdconv_if_c_gt=16, 
                 use_fdconv_if_k_in=[1, 3], use_fdconv_if_stride_in=[1], 
                 use_fbm_if_k_in=[3], use_fbm_for_stride=False,
                 kernel_temp=1.0, temp=None, att_multi=2.0, 
                 param_ratio=1, param_reduction=1.0, ksm_only_kernel_att=False, 
                 att_grid=1, use_ksm_local=True, ksm_local_act='sigmoid', 
                 ksm_global_act='sigmoid', spatial_freq_decompose=False, 
                 convert_param=True, linear_mode=False, fbm_cfg={}, **kwargs):
        super().__init__(*args, **kwargs)
        # Parameter assignments...
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act

        # Kernel num & Kernel temp setting
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
        
        # Condition for using FDConv
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt \
            or self.kernel_size[0] not in self.use_fdconv_if_k_in \
            or self.stride[0] not in self.use_fdconv_if_stride_in:
                return
        
        print('*** kernel_num:', self.kernel_num)
        
        # 🔥 FIX 1: Alpha factor for stability (Original calculation was prone to explosion)
        self.alpha = 1.0 # Use 1.0 for stable scaling/normalization
        
        # KSM_Global initialization
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0], groups=self.groups, 
                                                         temp=temp, kernel_temp=kernel_temp, reduction=reduction, 
                                                         kernel_num=self.kernel_num * self.param_ratio, # kn * R is the total kernel attention size
                                                         kernel_att_init=None, att_multi=att_multi, 
                                                         ksm_only_kernel_att=ksm_only_kernel_att, act_type=self.ksm_global_act, 
                                                         att_grid=att_grid, stride=self.stride, spatial_freq_decompose=spatial_freq_decompose)

        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
            
        if self.use_ksm_local:
            # 🔥 FIX 2: KSM_Local out_n must match the total number of elements it modulates
            total_local_att_size = int(self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.param_ratio / self.groups)
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=total_local_att_size)
        
        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True) 
        
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))

        # Normalization factor for trainable weight initialization
        norm_factor = 1.0
        
        if self.param_reduction < 1:
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :num_to_keep] 
            
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            
            weight_rfft = weight_rfft.reshape(-1, 2)[None, ].repeat(self.param_ratio, 1, 1) / norm_factor
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / norm_factor
        
        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            if hasattr(self, 'weight'):
                 del self.weight
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.0)
        
        # 🔥 FIX 3: Correct logic for reshaping indices for registration
        indices = []
        total_freq_bins = freq_indices.size(1)
        for i in range(self.param_ratio):
            indices_chunk = freq_indices[:, i * (total_freq_bins // self.param_ratio) : (i + 1) * (total_freq_bins // self.param_ratio)]
            
            if indices_chunk.size(1) > 0 and (indices_chunk.size(1) % self.kernel_num) == 0:
                indices.append(indices_chunk.reshape(2, self.kernel_num, -1))
            elif indices_chunk.size(1) > 0:
                valid_len = indices_chunk.size(1) - (indices_chunk.size(1) % self.kernel_num)
                indices.append(indices_chunk[:, :valid_len].reshape(2, self.kernel_num, -1))
            else:
                 # Handle the case where a chunk has zero bins
                 indices.append(torch.empty(2, self.kernel_num, 0, dtype=torch.long))
                 
        # Pad indices to ensure consistent size before stack (Necessary for torch.stack)
        max_len = max([i.size(-1) for i in indices]) if indices else 0
        padded_indices = []
        for i in indices:
            if i.size(-1) < max_len:
                pad_size = max_len - i.size(-1)
                i_padded = F.pad(i, (0, pad_size), mode='constant', value=0)
                padded_indices.append(i_padded)
            else:
                padded_indices.append(i)
                
        self.register_buffer('indices', torch.stack(padded_indices, dim=0), persistent=False)

    def get_FDW(self, ):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()
        # Use stable norm_factor (1.0) for consistency
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / 1.0 
        return weight_rfft
        
    def forward(self, x):
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return super().forward(x)
        
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        
        if self.use_ksm_local:
            hr_att_logit = self.KSM_Local(global_x) # B, 1, out_n
            # Reshape hr_att to match (B, R, Cout, Cin, K1, K2)
            hr_att_logit = hr_att_logit.reshape(x.size(0), self.param_ratio, self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            hr_att_logit = hr_att_logit.permute(0, 1, 2, 3, 4, 5) 
            
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1
            
        b = x.size(0)
        batch_size, in_planes, height, width = x.size()
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        
        # DFT_map shape: (B, H_fft, W_fft/2 + 1, 2)
        DFT_map = torch.zeros((b, d1 * k1, d2 * k2 // 2 + 1, 2), device=x.device)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1) # B, R, kn, 1
        
        dft_weight = self.dft_weight if hasattr(self, 'dft_weight') else self.get_FDW()

        for i in range(self.param_ratio):
            indices = self.indices[i] # (2, kernel_num, N_freq_bins_per_kernel)
            
            if indices.size(-1) == 0:
                continue
                
            N_bins = indices.size(-1)
            
            if self.param_reduction < 1:
                # Reduced case: dft_weight[i] is (N_reduced_total, 2)
                w = dft_weight[i].reshape(self.kernel_num, N_bins, 2)[None] # (1, kn, N_bins, 2)
            else:
                # Full case: Select based on indices
                total_w_bins = d1 * k1 * (d2 * k2 // 2 + 1)
                flat_indices = indices[0] * (d2 * k2 // 2 + 1) + indices[1] # (kn, N_bins)
                
                w_flat = dft_weight[i].reshape(-1, 2)
                
                # Apply alpha (which is 1.0 in stable version)
                w = w_flat[flat_indices].reshape(self.kernel_num, N_bins, 2)[None] * self.alpha # (1, kn, N_bins, 2)
                
            att_factor = kernel_attention[:, i].unsqueeze(-2) # (b, kn, 1, 1)

            # Modulate and accumulate
            modulated_w = torch.stack([
                w[..., 0] * att_factor[..., 0], 
                w[..., 1] * att_factor[..., 0]
            ], dim=-1).reshape(b, -1, 2)
            
            # Scatter/Index into DFT_map
            DFT_map[:, indices[0, :, :], indices[1, :, :]] += modulated_w


        # IRFFT2 - Use explicit size 's' for safety and reshape
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), s=(d1*k1, d2*k2), dim=(1, 2))
        adaptive_weights = adaptive_weights.reshape(batch_size, 1, self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5) # (B, R, Cout, Cin, K1, K2)
        
        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        # Final dynamic grouped convolution
        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (in_planes + self.out_channels) * height * width:
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1) # Sum over param_ratio (R)
            
            # Grouped Conv Preparation: (B*Cout, Cin/G, K1, K2)
            aggregate_weight = aggregate_weight.view([-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width) # (1, B*Cin, H, W)
            
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            
            output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            # filter_attention is already applied in aggregate_weight
        else:
            # Alternative path (kept)
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            
            if not isinstance(channel_attention, float): 
                x = x * channel_attention.view(b, -1, 1, 1)
                
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            
            output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            if not isinstance(filter_attention, float):
                 output = output * filter_attention.view(b, -1, 1, 1) # Apply filter attention separately

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    # Profile module logic simplified for safety/completeness
    def profile_module(self, input: Tensor, *args, **kwargs):
        b_sz, c, h, w = input.shape
        seq_len = h * w
        
        # FFT iFFT operation is dominant
        m_ff = 5 * b_sz * seq_len * int(math.log2(seq_len)) * c 
        
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        macs = 0 
        
        return input, params, macs + m_ff

if __name__ == '__main__':
    # Test with param_ratio=2, reduction=0.8
    x = torch.rand(4, 128, 64, 64) * 1
    m = FDConv(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, bias=True, param_ratio=2, param_reduction=0.8)
    
    print("FDConv Module:")
    print(m)
    
    y = m(x)
    print("\nOutput Shape:", y.shape)
    
    expected_h = (64 + 2*1 - 3) // 1 + 1 
    print(f"Expected Output Size: (4, 64, {expected_h}, {expected_h})")
