import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

# 🔹 Cross-Attention Module (CAFGL)
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):  # query: [B, Q, D], context: [B, K, D]
        B, Q, D = query.shape
        _, K, _ = context.shape

        # 🔥 반드시 정수형 변환
        B = int(B)
        Q = int(Q)
        K = int(K)
        D = int(D)
        d = D // self.heads

        Q_proj = self.query_proj(query).view(B, Q, self.heads, d).transpose(1, 2)
        K_proj = self.key_proj(context).view(B, K, self.heads, d).transpose(1, 2)
        V_proj = self.value_proj(context).view(B, K, self.heads, d).transpose(1, 2)

        attn_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V_proj)
        out = out.transpose(1, 2).contiguous().view(B, Q, D)  # 🔥 여기서 Tensor면 에러 발생
        return self.out_proj(out)

# 🔹 CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, frames):  # [B, F, C, H, W]
        B, F, C, H, W = frames.shape
        features = []

        for t in range(F):
            x = frames[:, t]  # [B, C, H, W]
            x = self.proj(self.pool(self.cnn(x)).squeeze(-1).squeeze(-1))  # [B, D]
            features.append(x)

        return torch.stack(features, dim=1)  # [B, F, D]

# 🔹 SPOT + CAFGL Model
class SPOT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1,
                 num_frames=8, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size.'

        self.num_frames = num_frames
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # CNN Branch
        self.cnn_branch = CNNFeatureExtractor(embed_dim=dim)

        # Cross-Attention (CAFGL)
        self.cross_attention = CrossAttention(dim=dim, heads=heads, dropout=dropout)

        # Positional Encoding
        total_tokens = self.num_frames * self.num_patches_per_frame + self.num_frames + 1  # Patch + CNN + CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, video):  # [B, F, C, H, W]
        B, F, C, H, W = video.shape
        P = self.patch_size

        # Patch Tokens
        patches = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=P, p2=P)
        patch_tokens = self.patch_embedding(patches)  # [B, F*N, D]

        # CNN Tokens
        cnn_tokens = self.cnn_branch(video)           # [B, F, D]

        # ✅ CAFGL: Cross-Attention between CNN & Patch Tokens
        fused_cnn_tokens = self.cross_attention(cnn_tokens, patch_tokens)  # [B, F, D]

        # Combine Tokens (Patch + Fused CNN) + CLS
        x = torch.cat([patch_tokens, fused_cnn_tokens], dim=1)  # [B, F*N + F, D]
        cls_token = self.cls_token.expand(B, -1, -1)            # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)                    # [B, 1 + F*N + F, D]

        # Positional Encoding + Dropout
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        # Transformer Encoding
        x = self.transformer(x)
        x = self.norm(x)

        # CLS Token Output
        return self.head(x[:, 0])  # [B, num_classes]
