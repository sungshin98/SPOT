import torch
import torch.nn as nn
from einops import rearrange

class TimeSformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=400,
                 num_frames=8, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_frames = num_frames
        self.patch_dim = 3 * patch_size * patch_size
        self.dim = dim

        # 패치 임베딩을 위한 Linear projection
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # Positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        total_patches = self.num_patches_per_frame * self.num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, total_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 출력 헤드
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # 패치 크기 저장
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, video):  # [B, F, C, H, W]
        B, F, C, H, W = video.shape
        P = self.patch_size

        # 영상 → 패치 분할 & 펼치기: [B, F, C, H, W] → [B, F*P*P, patch_dim]
        x = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=P, p2=P)
        x = self.patch_embedding(x)  # [B, F*patches, D]

        # cls token 추가
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)

        # Positional Encoding
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # CLS token → 회귀 출력 (num_classes가 T개인 회귀일 경우)
        x = self.norm(x)
        out = self.head(x[:, 0])  # [B, num_classes]
        return out
