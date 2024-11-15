import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
import gc

def np_softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def sim_matrix_inference(text_embeds_diff_per_video_id, vid_embeds_pooled_per_video_id, pooling_type):

        text_embeds_diff_per_video_id = text_embeds_diff_per_video_id / text_embeds_diff_per_video_id.norm(dim=-1, keepdim=True)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1,
                                                                                                          keepdim=True)

        num_txts, num_vids, max_text_per_vid, embed_dim = text_embeds_diff_per_video_id.shape

        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1, 2, 3, 0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.reshape(num_vids * max_text_per_vid, embed_dim,
                                                                                num_vids)
        text_embeds_diff_per_video_id = text_embeds_diff_per_video_id.permute(0, 2, 1, 3)
        text_embeds_diff_per_video_id = text_embeds_diff_per_video_id.reshape(num_vids * max_text_per_vid, num_txts, embed_dim)


        sims = torch.bmm(text_embeds_diff_per_video_id,
                         vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, num_txts, num_vids)
        sims_diag = torch.stack([sims[i, :, :, i] for i in range(sims.shape[0])],
                                dim=-1)
        sims_diag = sims_diag.permute(1, 0, 2)
        return sims_diag

def vid_embeds_pooled_per_video_id_diff( vid_embeds_pooled , all_vid_ids):
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                                                                             vid_embeds_pooled_per_video_id[i].keys(),
                                                                             vid_embeds_pooled.shape[-1])
        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)    
        return vid_embeds_pooled_per_video_id
def generate_embeds_per_video_id_diff(text_embeds_diff_allpairs, vid_embeds_pooled, all_vid_ids,
                                            pooling_type):        
    text_embeds_per_video_id = text_embeds_per_video_id_diff( text_embeds_diff_allpairs , all_vid_ids)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id_diff(vid_embeds_pooled , all_vid_ids)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1,0,2)
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input

def text_embeds_per_video_id_diff(text_embeds_diff_allpairs , all_vid_ids):
        text_embeds_per_video_id = []
        for i in range(text_embeds_diff_allpairs.shape[0]):
            text_embeds_per_video_id.append({})
            for idx, t_id in enumerate(all_vid_ids):
                if t_id in text_embeds_per_video_id[i]:
                    text_embeds_per_video_id[i][t_id].append(text_embeds_diff_allpairs[i, idx, :])
                else:
                    text_embeds_per_video_id[i][t_id] = [text_embeds_diff_allpairs[i, idx, :]]

        for i in range(len(text_embeds_per_video_id)):
            for t_id in text_embeds_per_video_id[i]:
                text_embeds_per_video_id[i][t_id] = torch.stack(text_embeds_per_video_id[i][t_id])

            text_embeds_per_video_id[i] = pad_and_stack_dict_to_tensor(text_embeds_per_video_id[i],
                                                                       text_embeds_per_video_id[i].keys(),
                                                                       text_embeds_diff_allpairs.shape[-1])

        text_embeds_per_video_id = torch.stack(text_embeds_per_video_id)
        return text_embeds_per_video_id