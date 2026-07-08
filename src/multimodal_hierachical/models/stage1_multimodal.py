import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Fuses Text and Audio features using Multi-head Cross-Attention.
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.3):
        super().__init__()
        # batch_first=True requires input tensor shape to be (Batch, Seq_Len, Embed_Dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_emb, audio_emb):
        # Input static features have shape (Batch, 768). 
        # Need to unsqueeze to (Batch, Seq_Len=1, 768) to fit the Attention function.
        text_emb = text_emb.unsqueeze(1)
        audio_emb = audio_emb.unsqueeze(1)
        
        # Text controls (Query), Audio provides auxiliary context (Key, Value)
        attn_output, _ = self.multihead_attn(query=text_emb, key=audio_emb, value=audio_emb)
        
        # Residual Connection + LayerNorm
        fused_vector = self.layer_norm(text_emb + self.dropout(attn_output))
        
        # Squeeze back to (Batch, 768)
        return fused_vector.squeeze(1)


class Stage1SentimentClassifier(nn.Module):
    """
    Complete network architecture for Phase 3 (Stage 1).
    """
    def __init__(self, embed_dim=768, num_classes=3, dropout=0.3):
        super().__init__()
        self.fusion_module = CrossAttentionFusion(embed_dim=embed_dim, num_heads=8, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_emb, text_emb):
        # 1. Extract multimodal fused vector
        fused_vector = self.fusion_module(text_emb=text_emb, audio_emb=audio_emb)
        
        # 2. Classify into 3 Sentiments
        logits = self.classifier(fused_vector)
        return logits