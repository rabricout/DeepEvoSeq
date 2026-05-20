import torch
import esm

# 1. Load a pretrained ESM model and alphabet
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout

# 2. Prepare sequences: list of (sequence_id, description, sequence)
data = [
    ("seq1", "MKTFFVAGVILLLATFTATALLLATFTATA"),  # example sequence
    ("seq2", "GAFVIVSSAVLGAGKSALTILLLATFTATA")
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 3. Run model to get representations
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[6], return_contacts=False)
token_representations = results["representations"][6]  # layer 6 for this model

print(token_representations.size())
input()

# 4. Pool per-residue to per-sequence embeddings (excluding BOS/EOS tokens)
sequence_embeddings = []
for i, (label, seq, seq_str) in enumerate(zip(batch_labels, batch_tokens, batch_strs)):
    # tokens for this sequence: tokens[i, 1:len(seq_str)+1]
    embedding = token_representations[i, 1:len(seq_str)+1].mean(0)
    sequence_embeddings.append(embedding)   # tensor of shape [hidden_dim]

print(sequence_embeddings)
print(len(sequence_embeddings), len(sequence_embeddings[0]))