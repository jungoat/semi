import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    시간 t를 위한 Sinusoidal Positional Embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    """
    Residual Block (논문의 Figure 2 오른쪽)
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.class_emb_proj = nn.Linear(class_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb, class_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        time_cond = self.time_emb_proj(F.silu(time_emb))
        class_cond = self.class_emb_proj(F.silu(class_emb))
        h = h + time_cond[:, :, None, None] + class_cond[:, :, None, None]

        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.residual_conv(x)

class Attention(nn.Module):
    """
    Self-Attention Block
    """
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)
        
        return x + self.out(out)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, has_attn=False):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_emb_dim, class_emb_dim)
        self.attn = Attention(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, time_emb, class_emb):
        x = self.res(x, time_emb, class_emb)
        x = self.attn(x)
        x = self.downsample(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, has_attn=False):
        super().__init__()
        # [버그 수정] Upsample 레이어의 입력 채널을 in_channels로, 출력 채널을 out_channels로 수정
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        # ResBlock은 skip connection과 합쳐진 후의 채널을 입력으로 받음 (out_channels * 2)
        self.res = ResBlock(out_channels * 2, out_channels, time_emb_dim, class_emb_dim)
        self.attn = Attention(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip_x, time_emb, class_emb):
        # [버그 수정] Upsample을 먼저 수행하여 x의 크기를 skip_x와 맞춤
        x = self.upsample(x)
        # 그 다음 skip connection과 합침
        x = torch.cat([x, skip_x], dim=1)
        x = self.res(x, time_emb, class_emb)
        x = self.attn(x)
        return x

class FlawMatchUnet(nn.Module):
    """
    논문 아키텍처를 직접 구현한 U-Net
    """
    def __init__(self, in_channels=1, out_channels=1, num_classes=12, cin=32):
        super().__init__()
        
        time_emb_dim = cin * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cin),
            nn.Linear(cin, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, cin)
        self.class_mlp = nn.Sequential(
            nn.Linear(cin, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, cin, 3, padding=1)
        
        channels = [cin, cin*2, cin*4, cin*8]
        
        self.down1 = DownBlock(channels[0], channels[1], time_emb_dim, time_emb_dim)
        self.down2 = DownBlock(channels[1], channels[2], time_emb_dim, time_emb_dim)
        self.down3 = DownBlock(channels[2], channels[3], time_emb_dim, time_emb_dim, has_attn=True)

        self.mid_res1 = ResBlock(channels[3], channels[3], time_emb_dim, time_emb_dim)
        self.mid_attn = Attention(channels[3])
        self.mid_res2 = ResBlock(channels[3], channels[3], time_emb_dim, time_emb_dim)

        # [버그 수정] UpBlock의 in_channels와 out_channels를 논리적으로 맞게 수정
        self.up1 = UpBlock(channels[3], channels[2], time_emb_dim, time_emb_dim, has_attn=True)
        self.up2 = UpBlock(channels[2], channels[1], time_emb_dim, time_emb_dim)
        self.up3 = UpBlock(channels[1], channels[0], time_emb_dim, time_emb_dim)
        
        self.final_res = ResBlock(channels[0] * 2, channels[0], time_emb_dim, time_emb_dim)
        self.final_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x, time, class_labels):
        t_emb = self.time_mlp(time)
        c_emb = self.class_mlp(self.class_emb(class_labels))

        x1 = self.init_conv(x)
        x2 = self.down1(x1, t_emb, c_emb)
        x3 = self.down2(x2, t_emb, c_emb)
        x4 = self.down3(x3, t_emb, c_emb)

        m = self.mid_res1(x4, t_emb, c_emb)
        m = self.mid_attn(m)
        m = self.mid_res2(m, t_emb, c_emb)

        # [버그 수정] UpBlock의 forward 순서가 바뀌었으므로, upsample이 자동으로 적용됨
        u = self.up1(m, x3, t_emb, c_emb)
        u = self.up2(u, x2, t_emb, c_emb)
        u = self.up3(u, x1, t_emb, c_emb)
        
        u = torch.cat([u, x1], dim=1)
        u = self.final_res(u, t_emb, c_emb)
        return self.final_conv(u)
