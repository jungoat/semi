# models/flow_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------- Positional Encoding -------- #
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb  # (B, dim)

# -------- Time Embedding -------- #
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.pos_enc = PositionalEncoding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        t = self.pos_enc(t)
        return self.mlp(t)  # (B, emb_dim)

# -------- Class Embedding -------- #
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, c):
        return self.linear(self.embedding(c))

# -------- Self Attention -------- #
class SelfAttention2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)

        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)                   # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)

# -------- ResBlock -------- #
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, use_attn=False):
        super().__init__()
        groups_in = math.gcd(in_ch, 8)
        groups_out = math.gcd(out_ch, 8)

        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.use_attn = use_attn
        if use_attn:
            self.attn = SelfAttention2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        if self.use_attn:
            h = self.attn(h)
        return h + self.skip(x)

# -------- Flow Generator (Multi-level Summation) -------- #
class FlowGenerator(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, emb_dim=128, num_classes=12):
        super().__init__()
        self.time_emb = TimeEmbedding(emb_dim)
        self.class_emb = ClassEmbedding(num_classes, emb_dim)

        # 레벨별 embedding projection
        self.enc1_emb = nn.Linear(emb_dim, base_ch)         # 128 -> 32
        self.enc2_emb = nn.Linear(emb_dim, base_ch * 2)     # 128 -> 64
        self.enc3_emb = nn.Linear(emb_dim, base_ch * 4)     # 128 -> 128
        self.bot_emb  = nn.Linear(emb_dim, base_ch * 4)     # 128 -> 128
        self.dec3_emb = nn.Linear(emb_dim, base_ch * 2)     # 128 -> 64
        self.dec2_emb = nn.Linear(emb_dim, base_ch)         # 128 -> 32
        self.dec1_emb = nn.Linear(emb_dim, base_ch)         # 128 -> 32

        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, base_ch)
        self.enc2 = ResBlock(base_ch, base_ch*2, base_ch*2)
        self.enc3 = ResBlock(base_ch*2, base_ch*4, base_ch*4, use_attn=True)

        # Bottleneck
        self.bot = ResBlock(base_ch*4, base_ch*4, base_ch*4, use_attn=True)

        # Decoder
        self.dec3 = ResBlock(base_ch*4 + base_ch*2, base_ch*2, base_ch*2)
        self.dec2 = ResBlock(base_ch*2 + base_ch, base_ch, base_ch)
        self.dec1 = ResBlock(base_ch + in_ch, base_ch, base_ch)

        self.final = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t, c):
        # Global embedding
        t_emb = self.time_emb(t)
        c_emb = self.class_emb(c) if c is not None else 0
        emb = t_emb + c_emb

        # Encoder
        x1 = self.enc1(x, self.enc1_emb(emb))
        x2 = self.enc2(F.avg_pool2d(x1, 2), self.enc2_emb(emb))
        x3 = self.enc3(F.avg_pool2d(x2, 2), self.enc3_emb(emb))

        # Bottleneck
        b = self.bot(x3, self.bot_emb(emb))

        # Decoder
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2), x2], dim=1), self.dec3_emb(emb))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), x1], dim=1), self.dec2_emb(emb))
        d1 = self.dec1(torch.cat([d2, x], dim=1), self.dec1_emb(emb))

        return self.final(d1)
