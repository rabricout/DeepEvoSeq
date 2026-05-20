import torch.nn.functional as F
import torch.nn as nn
import torch


class DeepEvoSeqGeneric(nn.Module):
    def __init__(
        self,
        nb_species: int,
        nb_amino_acids: int,
        is_simple:bool=False,
        attention:bool=False,
        internal_dim: int = 64,
        dropout: float = 0.25,
        attn_heads: int = 4,
        esm_embed_dim: int = 320,
    ):
        super().__init__()
        self.use_attention = attention
        self.is_simple = is_simple
        self.nb_amino_acids = nb_amino_acids
        self.internal_dim = internal_dim
        self.specs = {'internal_dim': internal_dim, 'dropout': dropout, 'is_simple': is_simple, 'attention': attention}
        if attention:
            self.specs['attn_heads'] = attn_heads

        if is_simple:
            # Project input to attention space
            self.proj_in = nn.Linear(nb_amino_acids, 32)
        else:
            # Project ESM embedding to attention space
            self.proj_in = nn.Linear(esm_embed_dim, 32)
        # Compact data from species
        self.proj_align = nn.Linear(32*nb_species, internal_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(internal_dim)
        
        if attention:
            attn_dim = internal_dim
            # Multi‑head self‑attention over residues
            self.attn = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(attn_dim, internal_dim),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(internal_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, internal_dim),
            )
        self.norm2 = nn.LayerNorm(internal_dim)

        # Final regression value head per residue
        self.before_classifier = nn.Linear(internal_dim + nb_species*nb_species, internal_dim + nb_species*nb_species)
        self.classifier = nn.Linear(internal_dim + nb_species*nb_species, 1)

    def give_specs_dict(self):
        return self.specs
    
    def forward(self, x: torch.Tensor, x_rates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of shape [B, L, 21] (batch, residues, aa_idx)

        Returns:
            logits: [B, L, nb_amino_acids]
        """

        if self.is_simple:
            # Reshape data as one-hot encoding
            x = F.one_hot(x, num_classes=self.nb_amino_acids).float()    

        # Project ESM embeddings
        h = self.proj_in(x)  # [B, S, L, 32]
        B, S, L, D = h.shape
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(B, L, S*D)

        h = self.proj_align(h)    # [B, L, internal_dim]

        if self.use_attention:
            # Self‑attention (residue ↔ residue)
            attn_out, _ = self.attn(
                query=h,
                key=h,
                value=h,
            )
            h = h + attn_out
            h = self.dropout(h)
            h = self.norm1(h)

        else:
            # FFN block
            ff_out = self.ffn(h)
            h = h + ff_out
            h = self.dropout(h)
            h = self.norm2(h)

        # Final per‑residue logits
        h = h.mean(dim=1)
        h = torch.cat((h, x_rates), dim=1)
        h = self.before_classifier(h)
        logits = self.classifier(h)  # [B, L, nb_amino_acids]

        return logits[:,0]
