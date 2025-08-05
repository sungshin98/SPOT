import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
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

        B, Q, K, D = int(B), int(Q), int(K), int(D)
        d = D // self.heads

        Q_proj = self.query_proj(query).view(B, Q, self.heads, d).transpose(1, 2)
        K_proj = self.key_proj(context).view(B, K, self.heads, d).transpose(1, 2)
        V_proj = self.value_proj(context).view(B, K, self.heads, d).transpose(1, 2)

        attn_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V_proj)
        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        return self.out_proj(out)


# 🔹 CNN Feature Extractor (selectable backbone)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, backbone: str = "resnet18"):
        super().__init__()

        if backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            out_dim = 512
            self.cnn = nn.Sequential(*list(model.children())[:-2])

        elif backbone == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
            out_dim = 512
            self.cnn = nn.Sequential(*list(model.children())[:-2])

        elif backbone == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            out_dim = 2048
            self.cnn = nn.Sequential(*list(model.children())[:-2])

        elif backbone == "mobilenet_v3_small":
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            out_dim = 576
            self.cnn = model.features

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(out_dim, embed_dim)

    def forward(self, frames):  # [B, F, C, H, W]
        B, F, C, H, W = frames.shape
        features = []
        for t in range(F):
            x = frames[:, t]  # [B, C, H, W]
            # 기존 스타일에 맞게 squeeze(-1).squeeze(-1) 사용
            x = self.proj(self.pool(self.cnn(x)).squeeze(-1).squeeze(-1))
            features.append(x)

        return torch.stack(features, dim=1)  # [B, F, D]


# 🔹 SPOT + CAFGL Model
class SPOT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1,
                 num_frames=8, dim=768, depth=12, heads=12, mlp_dim=3072,
                 dropout=0.1, cnn_backbone: str = "resnet18"):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size.'

        self.num_frames = num_frames
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # CNN Branch (backbone 선택 가능)
        self.cnn_branch = CNNFeatureExtractor(embed_dim=dim, backbone=cnn_backbone)

        # Cross-Attention
        self.cross_attention = CrossAttention(dim=dim, heads=heads, dropout=dropout)

        # Positional Encoding
        total_tokens = self.num_frames * self.num_patches_per_frame + self.num_frames + 1
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

        # Patch tokens
        patches = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=P, p2=P)
        patch_tokens = self.patch_embedding(patches)

        # CNN tokens
        cnn_tokens = self.cnn_branch(video)

        # Cross Attention
        fused_cnn_tokens = self.cross_attention(cnn_tokens, patch_tokens)

        # Combine tokens
        x = torch.cat([patch_tokens, fused_cnn_tokens], dim=1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        return self.head(x[:, 0])
