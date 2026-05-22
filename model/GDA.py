import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalDetailPath(nn.Module):
    
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        
        # Corresponds to: DWConv -> Conv 1x1
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        # Corresponds to: Conv 1x1
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class GatedDualPathAttention(nn.Module):
    
    def __init__(self, dim=36):
        super(GatedDualPathAttention, self).__init__()
        
        # Initial 1x1 conv to split features into two streams (X_F^1, X_F^2)
        self.split_conv = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        
        # === LDP (Local Detail Path) ===
        # This branch processes X_F^1
        self.ldp = LocalDetailPath(dim, 2)

        # === GGP (Global Gated Path) ===
        # This branch processes X_F^2
        self.ggp_dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim) # For spatial summary
        self.ggp_gate_conv = nn.Conv2d(dim, dim, 1, 1, 0)           # Corresponds to Conv1x1 in Eq 1
        
        self.gelu = nn.GELU()
        self.down_scale = 8
        self.min_feat_size = 3

        # Learnable parameters alpha and beta from Eq 1
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        
        # === Fusion ===
        # Final 1x1 conv to fuse LDP and GGP outputs
        self.fusion_conv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, f_fused):
        """
        f_fused corresponds to the enhanced feature map 'F' in the paper.
        """
        _, _, h, w = f_fused.shape
        
        # Split input 'F' into two parallel streams: x_f1 (for LDP) and x_f2 (for GGP)
        # Corresponds to X_F^1 and X_F^2
        x_f1, x_f2 = self.split_conv(f_fused).chunk(2, dim=1)
        
        # --- LDP Branch ---
        # Process x_f1 through the Local Detail Path
        # x_ldp corresponds to X_LDP = MLP(X_F^1)
        x_ldp = self.ldp(x_f1)
        
        # --- GGP Branch ---
        # Process x_f2 through the Global Gated Path
        
        # Calculate global descriptors:
        # 1. Spatial summary (x_s)
        down_scale = max(1, min(self.down_scale, h//self.min_feat_size, w//self.min_feat_size))
        spatial_summary = self.ggp_dw_conv(F.adaptive_max_pool2d(x_f2, (h // down_scale, w // down_scale)))
        
        # 2. Channel-wise variance (x_v)
        channel_variance = torch.var(x_f2, dim=(-2, -1), keepdim=True)
        
        # Calculate dynamic gating signal (G_gate) based on Eq 1
        gate_signal = self.ggp_gate_conv(spatial_summary * self.alpha + channel_variance * self.beta)
        dynamic_gate = F.interpolate(self.gelu(gate_signal), size=(h, w), mode='nearest')
        
        # Apply gate to the x_f2 stream
        # x_ggp corresponds to X_GGP = x * G_gate
        x_ggp = x_f2 * dynamic_gate

        # --- Final Fusion ---
        # Fuse the two paths (LDP + GGP) and apply the final 1x1 convolution
        # Corresponds to F_enhanced = Conv(X_LDP + X_GGP)
        f_enhanced = self.fusion_conv(x_ldp + x_ggp)
        
        return f_enhanced