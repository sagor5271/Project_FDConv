"""
Frequency Domain Convolution (FDConv) Implementation

This module implements FDConv, an advanced neural network convolution layer that performs 
convolution operations in the frequency domain using Fast Fourier Transform (FFT).

Original repository: https://github.com/Linwei-Chen/FDConv.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint


# =============================================================================
# Custom Activation Functions
# =============================================================================

class StarReLU(nn.Module):
    """
    StarReLU activation function: s * relu(x)^2 + b
    
    A learnable activation function that applies a quadratic transformation
    to ReLU outputs with learnable scale and bias parameters.
    
    Args:
        scale_value (float): Initial scale parameter value
        bias_value (float): Initial bias parameter value
        scale_learnable (bool): Whether scale parameter is trainable
        bias_learnable (bool): Whether bias parameter is trainable
        mode: Unused parameter (kept for compatibility)
        inplace (bool): Whether to perform operation in-place
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


# =============================================================================
# Global Kernel and Spatial Modulation
# =============================================================================
    
class KernelSpatialModulation_Global(nn.Module):
    """
    Global Kernel and Spatial Modulation Module
    
    This module generates attention weights for channels, filters, spatial positions,
    and kernel selection using global average pooling and learnable transformations.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels  
        kernel_size (int): Size of the convolution kernel
        groups (int): Number of groups for grouped convolution
        reduction (float): Channel reduction ratio for attention computation
        kernel_num (int): Number of kernels to choose from
        min_channel (int): Minimum number of attention channels
        temp (float): Temperature for attention softmax/sigmoid
        kernel_temp (float): Temperature for kernel attention
        kernel_att_init (str): Initialization method for kernel attention
        att_multi (float): Attention multiplier
        ksm_only_kernel_att (bool): Whether to use only kernel attention
        att_grid (int): Attention grid size
        stride (int): Convolution stride
        spatial_freq_decompose (bool): Whether to use spatial frequency decomposition
        act_type (str): Type of activation function ('sigmoid', 'tanh', 'softmax')
    """
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, 
                 kernel_num=4, min_channel=16, temp=1.0, kernel_temp=None, 
                 kernel_att_init='dyconv_as_extra', att_multi=2.0, ksm_only_kernel_att=False, 
                 att_grid=1, stride=1, spatial_freq_decompose=False, act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        
        # Basic configuration
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi
        self.att_grid = att_grid
        self.spatial_freq_decompose = spatial_freq_decompose

        # Global average pooling for context extraction
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Feature extraction layers
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        # Channel attention branch
        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1, bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention

        # Filter attention branch
        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        # Spatial attention branch
        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # Kernel attention branch
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all layers in the module."""
        # Standard initialization for conv and batch norm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Specialized initialization for attention layers
        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
            
        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
            
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)
            

    def update_temperature(self, temperature):
        """Update the temperature parameter for attention computation."""
        self.temperature = temperature

    @staticmethod
    def skip(_):
        """Skip function that returns 1.0 (no attention applied)."""
        return 1.0

    def get_channel_attention(self, x):
        """Generate channel attention weights."""
        if self.act_type == 'sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            channel_attention = 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return channel_attention

    def get_filter_attention(self, x):
        """Generate filter attention weights."""
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        """Generate spatial attention weights."""
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) 
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        """Generate kernel selection attention weights."""
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
        return kernel_attention
    
    def forward(self, x, use_checkpoint=False):
        """
        Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input tensor
            use_checkpoint: Whether to use gradient checkpointing for memory efficiency
            
        Returns:
            Tuple of (channel_attention, filter_attention, spatial_attention, kernel_attention)
        """
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        """Internal forward pass that computes all attention weights."""
        # Extract global context using adaptive average pooling
        avg_x = self.relu(self.bn(self.fc(x)))
        
        # Generate all attention components
        return (self.func_channel(avg_x), 
                self.func_filter(avg_x), 
                self.func_spatial(avg_x), 
                self.func_kernel(avg_x))


# =============================================================================
# Local Kernel and Spatial Modulation  
# =============================================================================


class KernelSpatialModulation_Local(nn.Module):
    """
    Local Kernel and Spatial Modulation Module
    
    This module generates local attention weights using 1D convolutions along the channel dimension.
    Optionally supports frequency domain processing for enhanced feature extraction.
    
    Args:
        channel (int): Number of input channels (if None, k_size is used directly)
        kernel_num (int): Number of kernels to generate attention for
        out_n (int): Output dimension multiplier
        k_size (int): Kernel size for 1D convolution
        use_global (bool): Whether to use global frequency domain processing
    """
    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        
        # Basic configuration
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        self.use_global = use_global
        
        # Adaptive kernel size based on channel dimension
        if channel is not None: 
            k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
            
        # 1D convolution for local attention computation
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, 
                             padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        
        # Optional global frequency domain processing
        if self.use_global:
            self.complex_weight = nn.Parameter(
                torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-6)
                
        # Normalization layer
        self.norm = nn.LayerNorm(self.channel)

    def forward(self, x, x_std=None):
        """
        Forward pass for local attention computation.
        
        Args:
            x: Input tensor with shape (B, C, 1, 1) - global pooled features
            x_std: Optional standard deviation features (unused in current implementation)
            
        Returns:
            Attention logits with shape (B, kernel_num, C, out_n)
        """
        # Reshape for 1D convolution: (B, C, 1) -> (B, 1, C)
        x = x.squeeze(-1).transpose(-1, -2)
        b, _, c = x.shape
        
        # Optional frequency domain enhancement
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
            
        # Normalize features
        x = self.norm(x)
        
        # Generate attention logits via 1D convolution
        att_logit = self.conv(x)
        
        # Reshape to final attention format: (B, kernel_num, C, out_n)
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c)
        att_logit = att_logit.permute(0, 1, 3, 2)
        
        return att_logit


# =============================================================================
# Frequency Band Modulation
# =============================================================================

class FrequencyBandModulation(nn.Module):
    """
    Frequency Band Modulation Module
    
    This module decomposes input features into different frequency bands using FFT
    and applies separate attention to each band. It uses pre-computed frequency masks
    for efficiency and supports configurable frequency decomposition.
    
    Args:
        in_channels (int): Number of input channels
        k_list (list): List of frequency cutoff values for band decomposition
        lowfreq_att (bool): Whether to apply attention to low frequency components
        fs_feat (str): Feature selection method (currently 'feat')
        act (str): Activation function type ('sigmoid', 'tanh', 'softmax')
        spatial (str): Type of spatial attention ('conv')
        spatial_group (int): Number of groups for spatial convolution
        spatial_kernel (int): Kernel size for spatial attention
        init (str): Weight initialization method ('zero')
        max_size (tuple): Maximum size for pre-computed masks (H, W)
    """
    def __init__(self, in_channels, k_list=[2], lowfreq_att=False, fs_feat='feat',
                 act='sigmoid', spatial='conv', spatial_group=1, spatial_kernel=3,
                 init='zero', max_size=(64, 64), **kwargs):
        super().__init__()
        
        # Configuration parameters
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.fs_feat = fs_feat
        self.act = act
        
        # Adjust spatial group size if needed
        if spatial_group > 64: 
            spatial_group = in_channels
        self.spatial_group = spatial_group

        # Build attention convolution layers for each frequency band
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:
                _n += 1
                
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.spatial_group,
                    stride=1,
                    kernel_size=spatial_kernel,
                    groups=self.spatial_group,
                    padding=spatial_kernel // 2,
                    bias=True
                )
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)
                    if freq_weight_conv.bias is not None:
                        freq_weight_conv.bias.data.zero_()
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
            
        # Pre-compute and cache frequency masks for efficiency
        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list))

    def _precompute_masks(self, max_size, k_list):
        """
        Pre-compute frequency masks at initialization for efficiency.
        
        Args:
            max_size (tuple): Maximum size (H, W) for mask computation
            k_list (list): List of frequency cutoff values
            
        Returns:
            torch.Tensor: Pre-computed masks with shape (num_masks, 1, max_h, max_w//2+1)
        """
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        
        # Get maximum frequency for each position
        freq_indices = freq_indices.abs().max(dim=-1, keepdims=False)[0]
        
        # Create masks for each frequency band
        masks = []
        for freq in k_list:
            mask = freq_indices < 0.5 / freq + 1e-8
            masks.append(mask)
        
        # Stack masks and add channel dimension for broadcasting
        return torch.stack(masks, dim=0).unsqueeze(1)

    def sp_act(self, freq_weight):
        """Apply spatial activation function to frequency weights."""
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError

    def forward(self, x, att_feat=None):
        """
        Forward pass for frequency band modulation.
        
        Args:
            x: Input tensor with shape (B, C, H, W)
            att_feat: Optional attention feature tensor (defaults to x)
            
        Returns:
            torch.Tensor: Modulated features with same shape as input
        """
        if att_feat is None:
            att_feat = x
            
        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, _, h, w = x.shape
        
        # Transform to frequency domain
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # Resize cached masks to current feature map frequency domain size
        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        # Process each frequency band
        for idx, freq in enumerate(self.k_list):
            # Apply frequency mask to extract current band
            mask = current_masks[idx]
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')
            
            # High frequency part is the residual
            high_part = pre_x - low_part
            pre_x = low_part
            
            # Apply spatial attention to high frequency components
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)
            
            # Modulate high frequency components with attention weights
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
            
        # Handle low frequency components
        if self.lowfreq_att:
            # Apply attention to low frequency residual
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self.sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            # Keep low frequency components unchanged
            x_list.append(pre_x)
            
        return sum(x_list)


# =============================================================================
# Utility Functions
# =============================================================================

def get_fft2freq(d1, d2, use_rfft=False):
    """
    Generate 2D frequency coordinates and sort them by distance from origin.
    
    This function creates a 2D frequency grid and sorts frequency components
    by their distance from the DC component (0, 0).
    
    Args:
        d1 (int): Size of first dimension
        d2 (int): Size of second dimension  
        use_rfft (bool): Whether to use real FFT (half spectrum)
        
    Returns:
        tuple: (sorted_coords, freq_hw)
            - sorted_coords: Coordinates sorted by frequency distance
            - freq_hw: 2D frequency grid
    """
    # Generate frequency components for each dimension
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    
    # Create 2D frequency grid
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    
    # Calculate distance from origin in frequency space
    dist = torch.norm(freq_hw, dim=-1)
    
    # Sort distances and get corresponding indices
    sorted_dist, indices = torch.sort(dist.view(-1))
    
    # Convert flat indices back to 2D coordinates
    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    
    # Optional visualization (disabled by default)
    if False:
        plt.imshow(dist.cpu().numpy(), cmap='gray', origin='lower')
        plt.colorbar()
        plt.title('Frequency Domain Distance')
        plt.show()
        
    return sorted_coords.permute(1, 0), freq_hw
from torch import Tensor
class FDConv(nn.Conv2d):
    def __init__(self, 
                 *args, 
                 reduction=0.0625, 
                 kernel_num=4,
                 use_fdconv_if_c_gt=16, #if channel greater or equal to 16, e.g., 64, 128, 256, 512
                 use_fdconv_if_k_in=[1, 3], #if kernel_size in the list
                 use_fdconv_if_stride_in=[1], #if stride in the list
                 use_fbm_if_k_in=[3], #if kernel_size in the list
                 use_fbm_for_stride=False,
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg={
                     'k_list':[2, 4, 8],
                     'lowfreq_att':False,
                     'fs_feat':'feat',
                     'act':'sigmoid',
                     'spatial':'conv',
                     'spatial_group':1,
                     'spatial_kernel':3,
                     'init':'zero',
                     'global_selection':False,
                 },
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in

        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        ### Kernel num & Kernel temp setting
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt \
            or self.kernel_size[0] not in self.use_fdconv_if_k_in:
                return
        print('*** kernel_num:', self.kernel_num)
        self.alpha = min(self.out_channels, self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0], groups=self.groups, 
                                                 temp=temp,
                                                 kernel_temp=kernel_temp,
                                                 reduction=reduction, kernel_num=self.kernel_num * self.param_ratio, 
                                                 kernel_att_init=None, att_multi=att_multi, ksm_only_kernel_att=ksm_only_kernel_att, 
                                                 act_type=self.ksm_global_act,
                                                 att_grid=att_grid, stride=self.stride, spatial_freq_decompose=spatial_freq_decompose)
        
        # print(use_fbm_for_stride, self.stride[0] > 1)
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
            # self.channel_comp = ChannelPool(reduction=16)
            
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(self.out_channels * self.kernel_size[0] * self.kernel_size[1]) )
        
        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)
            
    def convert2dftweight(self, convert_param):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        # এই ফাংশনটি বাইরে সংজ্ঞায়িত করা
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True) # 2, d1 * k1 * (d2 * k2 // 2 + 1)
        # freq_indices = freq_indices.reshape(2, self.kernel_num, -1)
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)) # d1 * k1, d2 * k2 // 2 + 1
        if self.param_reduction < 1:
            # freq_indices = freq_indices[:, torch.randperm(freq_indices.size(1), generator=torch.Generator().manual_seed(freq_indices.size(1)))] # 2, indices
            # freq_indices = freq_indices[:, :int(freq_indices.size(1) * self.param_reduction)] # 2, indices
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :num_to_keep] # 保留前 k 个最低频的索引
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None, ].repeat(self.param_ratio, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2) #param_ratio, d1, d2, k*k, 2
        
        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
        indices = []
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num, -1)) # paramratio, 2, kernel_num, d1 * k1 * (d2 * k2 // 2 + 1) // kernel_num
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self, ):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous() # d1 * k1, d2 * k2 // 2 + 1
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2) #param_ratio, d1, d2, k*k, 2
        return weight_rfft
        
    def forward(self, x):
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return super().forward(x)
        global_x = F.adaptive_avg_pool2d(x, 1)
        # এখানে KSM_Global, KernelSpatialModulation_Local, FBM এর সংজ্ঞা অনুপস্থিত, যা ধরে নেওয়া হলো অন্য ফাইল থেকে ইমপোর্ট করা হয়েছে।
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        if self.use_ksm_local:
            # global_x_std = torch.std(x, dim=(-1, -2), keepdim=True)
            hr_att_logit = self.KSM_Local(global_x) # b, kn, cin, cout * ratio, k1*k2, 
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])
            # hr_att_logit = hr_att_logit + self.hr_cin_bias[None, None, :, None, None, None] + self.hr_cout_bias[None, None, None, :, None, None] + self.hr_spatial_bias[None, None, None, None, :, :]
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
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
        DFT_map = torch.zeros((b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2), device=x.device)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()
            # print('get_FDW')

        # _t0 = time.perf_counter()
        for i in range(self.param_ratio):
            # print(i)
            # print(DFT_map.device)
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha # 1, kernel_num, -1, 2
                # print(w.shape)
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
                pass
        # print(time.perf_counter() - _t0)
        
        # --- 👇 সংশোধিত লাইন (ERROR FIX) 👇 ---
        
        target_rows = self.out_channels * self.kernel_size[0]
        target_cols = self.in_channels * self.kernel_size[1]
        
        adaptive_weights_flat = torch.fft.irfft2(
            torch.view_as_complex(DFT_map), 
            s=(target_rows, target_cols),  # <--- ফিক্স
            dim=(1, 2)
        )
        
        adaptive_weights = adaptive_weights_flat.reshape(
            batch_size, 
            1, 
            self.out_channels, 
            self.kernel_size[0], 
            self.in_channels, 
            self.kernel_size[1]
        ).permute(0, 1, 2, 4, 3, 5)

        # --- 👆 সংশোধিত লাইন শেষ 👆 ---
        
        # print(spatial_attention, channel_attention, filter_attention)
        if hasattr(self, 'FBM'):
            x = self.FBM(x)
            # x = self.FBM(x, self.channel_comp(x))

        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (in_planes + self.out_channels) * height * width:
            # print(channel_attention.shape, filter_attention.shape, hr_att.shape)
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            # aggregate_weight = spatial_attention * channel_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            # print(aggregate_weight.abs().max())
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float): 
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1)) # * filter_attention.reshape(b, -1, 1, 1)
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float): 
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            # if isinstance(filter_attention, torch.FloatTensor): 
            if isinstance(filter_attention, float): 
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1)) * filter_attention.view(b, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    def profile_module(
                self, input: Tensor, *args, **kwargs
            ):
            # TODO: to edit it
            b_sz, c, h, w = input.shape
            seq_len = h * w

            # FFT iFFT
            p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
            # others
            # params = macs = sum([p.numel() for p in self.parameters()])
            params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
            # // 2 min n become half after fft
            macs = macs * b_sz * seq_len

            # return input, params, macs
            return input, params, macs + m_ff

if __name__ == '__main__':
    # ... (মেইন ব্লক অপরিবর্তিত)
    x = torch.rand(4, 128, 64, 64) * 1
    # m = ODPEConv2d(in_channels=128, out_channels=128, kernel_num=8, kernel_size=3, padding=1, mirror_weight=False, weight_residual=False, use_rfft=True)
    # m = ODPEAdaptConv2d(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, mirror_weight=False, weight_residual=False, use_rfft=True, bias=True, param_ratio=4, omni_only_kernel_att=False, use_hr_att=False, att_grid=1, stride=2, spatial_freq_decompose=False)
    m = FDConv(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, bias=True)
    # m2 = DFT_Att(n=128)
    print(m)
    # m.convert2dftweight()
    y = m(x)
    print(y.shape)
    pass
