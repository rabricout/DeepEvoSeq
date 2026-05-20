import torch.nn.functional as F
import torch.nn as nn
import torch



# class DeepEvoSeqFCPosition(nn.Module):
#     def __init__(
#         self,
#         nb_species: int, 
#         esm_embed_dim: int,
#         internal_dim: int = 64,
#         dropout: float = 0.25,
#     ):
#         super().__init__()
#         self.attn_dim = internal_dim
#         self.specs = {'attn_heads': internal_dim, 'attn_dim': internal_dim, 'dropout': dropout}

#         # Project input to attention space
#         self.proj_in = nn.Linear(esm_embed_dim, 32)
#         # Compact data from species
#         self.proj_align = nn.Linear(32*nb_species, internal_dim)

#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(internal_dim)

#         # Optional feed‑forward layer
#         self.ffn = nn.Sequential(
#             nn.Linear(internal_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 128),
#             nn.GELU(),
#             nn.Linear(128, internal_dim),
#         )
#         self.norm2 = nn.LayerNorm(internal_dim)

#         # Final classifier head per residue
#         self.classifier = nn.Linear(internal_dim, 2)

#     def give_specs_dict(self):
#         return self.specs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: ESM2 embeddings of shape [B, L, D] (batch, residues, embed_dim)

#         Returns:
#             logits: [B, L, nb_amino_acids]
#         """
#         B, L, D = x.shape

#         # Project ESM embeddings
#         h = self.proj_in(x)  # [B, L, attn_dim]
#         B, S, L, D = h.shape
#         h = h.permute(0, 2, 1, 3)
#         h = h.reshape(B, L, S*D)

#         h = self.proj_align(h)    # [B, L, internal_dim]
#         h = self.dropout(h)
#         h = self.norm1(h)

#         # FFN block
#         ff_out = self.ffn(h)
#         h = h + ff_out
#         h = self.dropout(h)
#         h = self.norm2(h)

#         # Final per‑residue logits
#         logits = self.classifier(h)  # [B, L, nb_amino_acids]

#         return logits



# class DeepEvoSeqAttnHeadPosition(nn.Module):
#     def __init__(
#         self,
#         nb_species: int, 
#         esm_embed_dim: int,
#         attn_heads: int = 4,
#         attn_dim: int = 64,
#         dropout: float = 0.25,
#     ):
#         super().__init__()
#         self.attn_dim = attn_dim
#         self.specs = {'attn_heads': attn_heads, 'attn_dim': attn_dim, 'dropout': dropout}

#         # Project input to attention space
#         self.proj_in = nn.Linear(esm_embed_dim, 32)
#         # Compact data from species
#         self.proj_align = nn.Linear(32*nb_species, attn_dim)

#         # Multi‑head self‑attention over residues
#         self.attn = nn.MultiheadAttention(
#             embed_dim=attn_dim,
#             num_heads=attn_heads,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(attn_dim)

#         # Optional feed‑forward layer
#         self.ffn = nn.Sequential(
#             nn.Linear(attn_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, attn_dim),
#         )
#         self.norm2 = nn.LayerNorm(attn_dim)

#         # Final classifier head per residue
#         self.classifier = nn.Linear(attn_dim, 2)

#     def give_specs_dict(self):
#         return self.specs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: ESM2 embeddings of shape [B, L, D] (batch, residues, embed_dim)

#         Returns:
#             logits: [B, L, nb_amino_acids]
#         """

#         # Project ESM embeddings
#         h = self.proj_in(x)  # [B, L, attn_dim]
#         B, S, L, D = h.shape
#         h = h.permute(0, 2, 1, 3)
#         h = h.reshape(B, L, S*D)

#         h = self.proj_align(h)    # [B, L, internal_dim]

#         # Self‑attention (residue ↔ residue)
#         attn_out, _ = self.attn(
#             query=h,
#             key=h,
#             value=h,
#         )
#         h = h + attn_out
#         h = self.dropout(h)
#         h = self.norm1(h)

#         # FFN block
#         ff_out = self.ffn(h)
#         h = h + ff_out
#         h = self.dropout(h)
#         h = self.norm2(h)

#         # Final per‑residue logits
#         logits = self.classifier(h)  # [B, L, nb_amino_acids]

#         return logits





# class DeepEvoSeqSimpleFCPosition(nn.Module):
#     def __init__(
#         self,
#         nb_species: int,
#         nb_amino_acids: int,
#         internal_dim: int = 64,
#         dropout: float = 0.25,
#     ):
#         super().__init__()
#         self.nb_amino_acids = nb_amino_acids
#         self.attn_dim = internal_dim
#         self.specs = {'attn_heads': internal_dim, 'attn_dim': internal_dim, 'dropout': dropout}

#         # Project input to internal space
#         self.proj_in = nn.Linear(nb_amino_acids, 32)
#         # Compact data from species
#         self.proj_align = nn.Linear(32*nb_species, internal_dim)

#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(internal_dim)

#         # Optional feed‑forward layer
#         self.ffn = nn.Sequential(
#             nn.Linear(internal_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 128),
#             nn.GELU(),
#             nn.Linear(128, internal_dim),
#         )
#         self.norm2 = nn.LayerNorm(internal_dim)

#         # Final classifier head per residue
#         self.classifier = nn.Linear(internal_dim, 2)

#     def give_specs_dict(self):
#         return self.specs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: ESM2 embeddings of shape [B, S, L, D] (batch, species, residues, embed_dim)

#         Returns:
#             logits: [B, L, 2]
#         """

#         # Initial raw embedding as one-hot
#         x = F.one_hot(x, num_classes=self.nb_amino_acids).float()

#         # Project one-hot embeddings
#         h = self.proj_in(x)  # [B, S, L, 32]
#         B, S, L, D = h.shape
#         h = h.permute(0, 2, 1, 3)
#         h = h.reshape(B, L, S*D)  # [B, L, S*32]

#         h = self.proj_align(h)    # [B, L, internal_dim]
#         h = self.dropout(h)
#         h = self.norm1(h)

#         # FFN block
#         ff_out = self.ffn(h)
#         h = h + ff_out
#         h = self.dropout(h)
#         h = self.norm2(h)

#         # Final per‑residue logits
#         logits = self.classifier(h)  # [B, L, 2]

#         return logits



# class DeepEvoSeqSimpleAttnHeadPosition(nn.Module):
#     def __init__(
#         self,
#         nb_species: int,
#         nb_amino_acids: int,
#         attn_heads: int = 4,
#         attn_dim: int = 64,
#         dropout: float = 0.25,
#     ):
#         super().__init__()
#         self.nb_amino_acids = nb_amino_acids
#         self.attn_dim = attn_dim
#         self.specs = {'attn_heads': attn_heads, 'attn_dim': attn_dim, 'dropout': dropout}

#         # Project input to attention space
#         self.proj_in = nn.Linear(nb_amino_acids, 32)
#         # Compact data from species
#         self.proj_align = nn.Linear(32*nb_species, attn_dim)

#         # Multi‑head self‑attention over residues
#         self.attn = nn.MultiheadAttention(
#             embed_dim=attn_dim,
#             num_heads=attn_heads,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(attn_dim)

#         # Optional feed‑forward layer
#         self.ffn = nn.Sequential(
#             nn.Linear(attn_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, attn_dim),
#         )
#         self.norm2 = nn.LayerNorm(attn_dim)

#         # Final classifier head per residue
#         self.classifier = nn.Linear(attn_dim, 2)

#     def give_specs_dict(self):
#         return self.specs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: ESM2 embeddings of shape [B, S, L, D] (batch, species, residues, embed_dim)

#         Returns:
#             logits: [B, L, 2]
#         """

#         # Reshape data as one-hot encoding
#         x = F.one_hot(x, num_classes=self.nb_amino_acids).float()

#         # Project one_hot embeddings
#         h = self.proj_in(x)  # [B, S, L, 32]
#         B, S, L, D = h.shape
#         h = h.permute(0, 2, 1, 3)
#         h = h.reshape(B, L, S*D)
#         h = self.proj_align(h)    # [B, L, internal_dim]

#         # Self‑attention (residue ↔ residue)
#         attn_out, _ = self.attn(
#             query=h,
#             key=h,
#             value=h,
#         )
#         h = h + attn_out
#         h = self.dropout(h)
#         h = self.norm1(h)

#         # FFN block
#         ff_out = self.ffn(h)
#         h = h + ff_out
#         h = self.dropout(h)
#         h = self.norm2(h)

#         # Final per‑residue logits
#         logits = self.classifier(h)  # [B, L, 2]

#         return logits



# class DeepEvoSeqFCPosition(nn.Module):
#     def __init__(
#         self,
#         nb_species: int,
#         esm_embed_dim: int,
#         internal_dim: int = 64,
#         dropout: float = 0.25,
#     ):
#         super().__init__()
#         self.attn_dim = internal_dim
#         self.specs = {'internal_dim': internal_dim, 'dropout': dropout}

#         # Project input to internal space
#         self.proj_in = nn.Linear(esm_embed_dim, 32)
#         # Compact data from species
#         self.proj_align = nn.Linear(32*nb_species, internal_dim)

#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(internal_dim)

#         # Optional feed‑forward layer
#         self.ffn = nn.Sequential(
#             nn.Linear(internal_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 128),
#             nn.GELU(),
#             nn.Linear(128, internal_dim),
#         )
#         self.norm2 = nn.LayerNorm(internal_dim)

#         # Final classifier head per residue
#         self.classifier = nn.Linear(internal_dim, 2)

#     def give_specs_dict(self):
#         return self.specs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: ESM2 embeddings of shape [B, S, L, D] (batch, species, residues, embed_dim)

#         Returns:
#             logits: [B, L, 2]
#         """

#         # Project esm embeddings
#         h = self.proj_in(x)  # [B, S, L, 32]
#         B, S, L, D = h.shape
#         h = h.permute(0, 2, 1, 3)
#         h = h.reshape(B, L, S*D)  # [B, L, S*32]

#         h = self.proj_align(h)    # [B, L, internal_dim]
#         h = self.dropout(h)
#         h = self.norm1(h)

#         # FFN block
#         ff_out = self.ffn(h)
#         h = h + ff_out
#         h = self.dropout(h)
#         h = self.norm2(h)

#         # Final per‑residue logits
#         logits = self.classifier(h)  # [B, L, 2]

#         return logits



# class DeepEvoSeqAttnHeadPosition(nn.Module):
#     def __init__(
#         self,
#         nb_species: int,
#         esm_embed_dim: int,
#         attn_heads: int = 4,
#         attn_dim: int = 64,
#         dropout: float = 0.25,
#     ):
#         super().__init__()
#         self.attn_dim = attn_dim
#         self.specs = {'attn_heads': attn_heads, 'attn_dim': attn_dim, 'dropout': dropout}

#         # Project input to attention space
#         self.proj_in = nn.Linear(esm_embed_dim, 32)
#         # Compact data from species
#         self.proj_align = nn.Linear(32*nb_species, attn_dim)

#         # Multi‑head self‑attention over residues
#         self.attn = nn.MultiheadAttention(
#             embed_dim=attn_dim,
#             num_heads=attn_heads,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(attn_dim)

#         # Optional feed‑forward layer
#         self.ffn = nn.Sequential(
#             nn.Linear(attn_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, attn_dim),
#         )
#         self.norm2 = nn.LayerNorm(attn_dim)

#         # Final classifier head per residue
#         self.classifier = nn.Linear(attn_dim, 2)

#     def give_specs_dict(self):
#         return self.specs
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: ESM2 embeddings of shape [B, S, L, D] (batch, species, residues, embed_dim)

#         Returns:
#             logits: [B, L, 2]
#         """

#         # Project ESM embeddings
#         h = self.proj_in(x)  # [B, L, attn_dim]
#         B, S, L, D = h.shape
#         h = h.permute(0, 2, 1, 3)
#         h = h.reshape(B, L, S*D)

#         h = self.proj_align(h)    # [B, L, internal_dim]

#         # Self‑attention (residue ↔ residue)
#         attn_out, _ = self.attn(
#             query=h,
#             key=h,
#             value=h,
#         )
#         h = h + attn_out
#         h = self.dropout(h)
#         h = self.norm1(h)

#         # FFN block
#         ff_out = self.ffn(h)
#         h = h + ff_out
#         h = self.dropout(h)
#         h = self.norm2(h)

#         # Final per‑residue logits
#         logits = self.classifier(h)  # [B, L, 2]

#         return logits




class DeepEvoSeqPositionGeneric(nn.Module):
    def __init__(
        self,
        nb_species: int,
        nb_amino_acids: int,
        is_simple:bool=False,
        attention:bool=False,
        internal_dim: int = 64,
        attn_heads: int = 4,
        dropout: float = 0.25,
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

        self.proj_one_hot = nn.Linear(nb_amino_acids, 32)
        if is_simple:
            self.proj_in = nn.Linear(32*nb_species, internal_dim)
        else:
            self.proj_embedding = nn.Linear(esm_embed_dim, 32)
            self.proj_in = nn.Linear(32*nb_species*2, internal_dim)

        # if is_simple:
        #     # Project input to attention space
        #     self.proj_in = nn.Linear(nb_amino_acids, 32)
        # else:
        #     # Project ESM embedding to attention space
        #     self.proj_in = nn.Linear(esm_embed_dim, 32)
        # # Compact data from species
        # self.proj_align = nn.Linear(32*nb_species, internal_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(internal_dim)

        if attention:
            self.norm1 = nn.LayerNorm(internal_dim)
            self.norm1b = nn.LayerNorm(internal_dim)
            attn_dim = internal_dim
            # Multi‑head self‑attention over residues
            self.attn_1 = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_2 = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(internal_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, internal_dim),
            )
        self.norm2 = nn.LayerNorm(internal_dim)

        # Final classifier head per residue
        self.classifier = nn.Linear(internal_dim, 2)

    def give_specs_dict(self):
        return self.specs

    def forward(self, x_raw: torch.Tensor, x_embedded: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: ESM2 embeddings of shape [B, S, L, D] (batch, species, residues, embed_dim)

        Returns:
            logits: [B, L, 2]
        """

        # Reshape data as one-hot encoding
        x_raw = F.one_hot(x_raw, num_classes=self.nb_amino_acids).float()
        x_raw = self.proj_one_hot(x_raw)
        if not self.is_simple:
            x_embed = self.proj_embedding(x_embedded)
            x = torch.cat((x_raw, x_embed), dim=-1) 
        else:
            x = x_raw

        # if self.is_simple:
        #     # Reshape data as one-hot encoding
        #     x = F.one_hot(x, num_classes=self.nb_amino_acids).float()

        # Project ESM embeddings
        # h = self.proj_in(x)  # [B, L, attn_dim]
        h = x
        B, S, L, D = h.shape
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(B, L, S*D)

        h = self.proj_in(h)    # [B, L, internal_dim]

        if self.use_attention:
            # Self‑attention (residue ↔ residue)
            attn_out, _ = self.attn_1(
                query=h,
                key=h,
                value=h,
            )
            h = h + attn_out
            h = self.dropout(h)
            h = self.norm1(h)
            attn_out, _ = self.attn_2(
                query=h,
                key=h,
                value=h,
            )
            h = h + attn_out
            h = self.dropout(h)
            h = self.norm1b(h)
        else:
            # FFN block
            ff_out = self.ffn(h)
            h = h + ff_out
            h = self.dropout(h)
            h = self.norm2(h)

        # Final per‑residue logits
        logits = self.classifier(h)  # [B, L, 2]

        return logits
