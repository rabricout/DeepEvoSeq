import numpy as np

# Load .npy file
data = np.load('A1_SINGLE_REPR/10000.fasta.output/A1_closest_aplodontia_ancestor_single_repr_rank_001_alphafold2_ptm_model_1_seed_000.npy')

# Check shape and type
print(f"Shape: {data.shape}")
print(f"Type: {data.dtype}")
print(data)