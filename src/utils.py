from torch.nn.utils.rnn import pad_sequence
import torch

MAX_LENGTH=1024

# def collate_fn_aa(batch, max_len=MAX_LENGTH):
#     # xs = [x[:max_len] for x in [item[0] for item in batch]]  # crop first
#     # ys = [y[:max_len] for y in [item[1] for item in batch]]
#     xs = [item[0] for item in batch]  # list of 1D tensors, different lengths
#     xs_t = []
#     for name, seq in xs:
#         xs_t.append((name, seq[:max_len]))
#     ys = [item[1][:max_len] for item in batch]

#     #xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max, D]
#     ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)  # [B, L_max]

#     return xs_t, ys_padded


def collate_fn_aa(batch, max_len=MAX_LENGTH):
    # xs = [x[:max_len] for x in [item[0] for item in batch]]  # crop first
    # ys = [y[:max_len] for y in [item[1] for item in batch]]
    # xs = [item[0][0][:max_len] for item in batch]  # list of 1D tensors, different lengths
    xs = [(item[0][0][:,:max_len]).transpose(0,1) for item in batch]
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max]
    xs_padded = xs_padded.transpose(1,2)
    xs_embed = [item[0][1] for item in batch]  # list of 1D tensors, different lengths
    xs_embed_t = []
    for entry in xs_embed:
        xs_embed_t.append([(name, seq[:max_len]) for name, seq in entry])
    xs_a1 = [item[0][2][:max_len] for item in batch]
    ys = [item[1][0][:max_len] for item in batch]
    ps = [item[1][1][:max_len] for item in batch]

    #xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max, D]
    xs_a1_padded = pad_sequence(xs_a1, batch_first=True, padding_value=0.0)  # [B, L_max]
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)  # [B, L_max]
    ps_padded = pad_sequence(ps, batch_first=True, padding_value=False)  # [B, L_max]

    return ((xs_padded, xs_embed_t), xs_a1_padded), (ys_padded, ps_padded)



def collate_fn_aa_simple(batch, max_len=MAX_LENGTH):
    # xs = [x[:max_len] for x in [item[0] for item in batch]]  # crop first
    # ys = [y[:max_len] for y in [item[1] for item in batch]]
    xs = [(item[0][0][:,:max_len]).transpose(0,1) for item in batch]
    xs_a1 = [item[0][1][:max_len] for item in batch]
    ys = [item[1][0][:max_len] for item in batch]
    ps = [item[1][1][:max_len] for item in batch]

    #xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max, D]
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max]
    xs_padded = xs_padded.transpose(1,2)
    # print(xs_a1)
    xs_a1_padded = pad_sequence(xs_a1, batch_first=True, padding_value=0.0)  # [B, L_max]
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)  # [B, L_max]
    ps_padded = pad_sequence(ps, batch_first=True, padding_value=False)  # [B, L_max]

    return (xs_padded, xs_a1_padded), (ys_padded, ps_padded)



def collate_fn_aa_position_simple(batch, max_len=MAX_LENGTH):
    xs = [(item[0][:,:max_len]).transpose(0,1) for item in batch]
    ys = [item[1][:max_len] for item in batch]

    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max, D]
    xs_padded = xs_padded.transpose(1,2)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)  # [B, L_max]

    return xs_padded, ys_padded



def collate_fn_aa_position(batch, max_len=MAX_LENGTH):
    xs = [(item[0][0][:,:max_len]).transpose(0,1) for item in batch]
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max]
    xs_padded = xs_padded.transpose(1,2)
    xs_embed = [item[0][1] for item in batch]  # list of 1D tensors, different lengths
    xs_embed_t = []
    for entry in xs_embed:
        xs_embed_t.append([(name, seq[:max_len]) for name, seq in entry])

    # xs = [item[0] for item in batch]  # list of 1D tensors, different lengths
    # xs_t = []
    # for entry in xs:
    #     xs_t.append([(name, seq[:max_len]) for name, seq in entry])
    ys = [item[1][:max_len] for item in batch]
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)  # [B, L_max]
    
    return (xs_padded, xs_embed_t), ys_padded



def collate_fn_aa_subst_rate_simple(batch, max_len=MAX_LENGTH):
    xs = [(item[0][0][:,:max_len]).transpose(0,1) for item in batch]
    x_rates = torch.tensor([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch])

    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)  # [B, L_max, D]
    xs_padded = xs_padded.transpose(1,2)

    return (xs_padded, x_rates), ys



def collate_fn_aa_subst_rate(batch, max_len=MAX_LENGTH):
    xs = [item[0] for item in batch]  # list of 1D tensors, different lengths
    xs_t = []
    for entry in xs:
        xs_t.append([(name, seq[:max_len]) for name, seq in entry])
    x_rates = torch.tensor([item[0][1] for item in batch])
    ys = torch.tensor([item[1] for item in batch])

    return (xs_t, x_rates), ys