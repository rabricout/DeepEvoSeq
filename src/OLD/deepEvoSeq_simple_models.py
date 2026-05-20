import torch
import torch.nn as nn



class DeepEvoSeqASimplePosition(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_heads: int = 8,
        attn_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.attn_dim = attn_dim

        # Project input to attention space
        self.proj_in = nn.Linear(input_dim, attn_dim)

        # Multi‑head self‑attention over residues
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(attn_dim)

        # Optional feed‑forward layer
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.GELU(),
            nn.Linear(128, attn_dim),
        )
        self.norm2 = nn.LayerNorm(attn_dim)

        # Final classifier head per residue
        self.classifier = nn.Linear(attn_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of shape [B, L, D] (batch, residues, aa_idx)

        Returns:
            logits: [B, L, nb_amino_acids]
        """
        B, L, D = x.shape

        # Project input
        h = self.proj_in(x)  # [B, L, attn_dim]

        # Self‑attention (residue ↔ residue)
        attn_out, _ = self.attn(
            query=h,
            key=h,
            value=h,
        )
        h = h + attn_out
        h = self.dropout(h)
        h = self.norm1(h)

        # FFN block
        ff_out = self.ffn(h)
        h = h + ff_out
        h = self.dropout(h)
        h = self.norm2(h)

        # Final per‑residue logits
        logits = self.classifier(h)  # [B, L, nb_amino_acids]

        return logits
    


class DeepEvoSeqSimpleNature(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nb_amino_acids: int,
        attn_heads: int = 8,
        attn_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.nb_amino_acids = nb_amino_acids
        self.attn_dim = attn_dim

        # Project input to attention space
        self.proj_in = nn.Linear(input_dim, attn_dim)

        # Multi‑head self‑attention over residues
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(attn_dim)

        # Optional feed‑forward layer
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.norm2 = nn.LayerNorm(256)

        # Final classifier head per residue
        self.classifier = nn.Linear(256, nb_amino_acids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of shape [B, L, 21] (batch, residues, aa_idx)

        Returns:
            logits: [B, L, nb_amino_acids]
        """
        B, L, D = x.shape

        # Project input
        h = self.proj_in(x)  # [B, L, attn_dim]

        # Self‑attention (residue ↔ residue)
        attn_out, _ = self.attn(
            query=h,
            key=h,
            value=h,
        )
        h = h + attn_out
        h = self.dropout(h)
        h = self.norm1(h)

        # FFN block
        ff_out = self.ffn(h)
        h = h + ff_out
        h = self.dropout(h)
        h = self.norm2(h)

        # Final per‑residue logits
        logits = self.classifier(h)  # [B, L, nb_amino_acids]

        return logits
    
