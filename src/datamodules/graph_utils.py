import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn import metrics

def get_train_val_test_masks(node_ids):
    idxs = [0, 1, 2]
    idx_masks = np.array(random.choices(idxs, weights = (0.7, 0.15, 0.15), k=len(node_ids)))
    train_val_test_masks = np.zeros((idx_masks.size, idx_masks.max() + 1))
    train_val_test_masks[np.arange(idx_masks.size), idx_masks] = 1
    train_val_test_masks = np.array(train_val_test_masks, dtype=bool)
    
    return train_val_test_masks


def nodes_table(modal, modal_dict, modal_type_map):
    node_ids = list(modal_dict.keys())
    print(f'modal {modal} count: {len(node_ids)}')
    nodes = pd.DataFrame({'node_id': list(node_ids)})

    if modal == 'images':
        train_val_test_masks = get_train_val_test_masks(node_ids)    
    else:
        train_val_test_masks = np.ones((len(node_ids), 3))
    
    nodes[['train_mask', 'val_mask', 'test_mask']] = train_val_test_masks
    tqdm.pandas(desc=f'formatting {modal} node embeddings')
    nodes['ntype'] = [modal_type_map[modal]]*len(nodes)
    nodes['feat'] = nodes['node_id'].progress_apply(lambda x: ', '.join([str(y) for y in modal_dict[x].tolist()]))

    return nodes


def edges_table(node_links, src_col, dest_col):
    edges = node_links[[src_col, dest_col]]
    dest_is_list = (edges.applymap(type) == list)[dest_col].iloc[0]
    if dest_is_list == True:
        edges = edges.explode(dest_col)
                     
    edges = edges.dropna(subset=dest_col)\
                 .rename(columns={src_col: 'src_id', dest_col: 'dst_id'})\
                 .reset_index(drop=True)
    
    return edges


def get_cosine_sim(vecs, all_vecs, vec_idxs):
    '''
    Given a batch of vectors bxd and a matrix of vectors mxd for comparison, 
    give the cosine similarities between the batch vecs and every vector in all_vecs, which
    is calculated by dot(vecs, all_vecs) / (norm(vecs)*norm(all_vecs)) for vec2s in all_vecs

    Returns bxm-sized array of cosine similarities, where idxs corresponding to input vecs is set to 0
    '''
    cosine_sims = metrics.pairwise.cosine_similarity(vecs, all_vecs)
    cosine_sims[np.arange(vec_idxs.size), vec_idxs] = 0

    return cosine_sims

