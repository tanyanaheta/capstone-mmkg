import os
import sys
import random
import math
import json
import yaml
import joblib
from ast import literal_eval
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pydantic
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import dgl
import graph_utils


def load_zillow_nodes(cfg : DictConfig, scenes = False):
    '''
    Build modal_dicts from zillow data with structure:
    {
        modality_1 (e.g. images): {
            modality1_node_id_1: [modality1_node_id_1_feature_embeddings]
            modality1_node_id_2: [modality1_node_id_2_feature_embeddings]
            ...
            },
        modality_2: {
            modality2_node_id_1: [modality2_node_id_1_feature_embeddings]
            ...
        }
    }
    '''
    
    image_embed_dict = joblib.load(cfg.data.zillow.image_embeds)
    keyword_embed_dict = {}
    scene_embed_dict = {}

    keyword_embed_dictlist = joblib.load(cfg.data.zillow.keyword_embeds)
    for dict in tqdm(keyword_embed_dictlist, desc='converting list to dict'):
        dict_key = list(dict.keys())[0]
        keyword_embed_dict[dict_key] = dict[dict_key]

    node_dicts = {
        'images': image_embed_dict,
        'keywords': keyword_embed_dict
    }

    if scenes == True:
        scene_embed_dictlist = joblib.load(cfg.data.zillow.scene_embeds)
        for dict in tqdm(scene_embed_dictlist, desc='converting list to dict'):
            dict_key = list(dict.keys())[0]
            scene_embed_dict[dict_key] = dict[dict_key]
        
        node_dicts['scenes'] = scene_embed_dict
    
    return node_dicts


def load_mscoco_nodes(cfg : DictConfig):
    '''
    Same type of output as load_zillow_data but for MSCOCO data.
    '''

    image_embeds = torch.load(cfg.data.mscoco.image_embeds).numpy()
    keyword_embeds = torch.load(cfg.data.mscoco.keyword_embeds).numpy()

    image_ids = np.arange(len(image_embeds))
    keyword_ids = np.arange(len(image_ids), len(image_ids)+len(keyword_embeds))

    node_dicts = {
        'images': dict(zip(image_ids, image_embeds)),
        'keywords': dict(zip(keyword_ids, keyword_embeds))
    }

    return node_dicts


def get_all_graph_edges(cfg : DictConfig, org='coco', scenes=False):
    if org == 'coco':
        node_links = pd.read_csv(cfg.data.mscoco.connections)
        scenes = False
        src_id, dst_id = ('img_id', 'tag_ids')
        
    elif org == 'zillow':
        node_links = pd.read_csv(cfg.data.zillow.connections)
        src_id, dst_id = ('url_hash', 'image_keyword_hash') # url_hash = img_id, # image_keyword_hash = list of keyword_ids

    else:
        raise ValueError(f'Expected org input of "coco" or "zillow", got {org}')
    
    node_links[dst_id] = node_links[dst_id].apply(lambda x: literal_eval(x)) # convert dst_ids (loaded from csv in string format) to list format
    edges = graph_utils.edges_table(node_links, src_id, dst_id)

    if scenes == True:
        dst_id = 'scene_hash'
        edges = pd.concat([edges, graph_utils.edges_table(node_links, src_id, dst_id)])
    
    edges = edges.drop_duplicates().reset_index(drop=True)
    
    return edges


def get_all_graph_nodes(node_dicts):
    '''
    Use nodes_table() method from graph_utils to build a table of node IDs from
    all modalities, plus train / val / test masks and node features (embeddings)
    '''

    nodes_table_modals = pd.DataFrame()
    
    for modal in node_dicts:
        nodes_table_modals = pd.concat([nodes_table_modals, graph_utils.nodes_table(modal, node_dicts[modal])])
    
    nodes_table_modals = nodes_table_modals.drop_duplicates(subset='node_id').reset_index(drop=True)

    return nodes_table_modals


def get_modal_embeds(modal, node_dicts):
    '''
    Inputs:
        - modal (str): modality key to access modality's embeddings from modal_dicts
        - modal_dicts (dict): dictionary that is structured as follows:
            modal_dict = {modal: {modality_hash_1: [hash_1_embedding],
                                  modality_hash_2: [hash_2_embedding],
                                  ...}}

    Outputs:
    For a given modality (image or keyword), return: 
        - modal_embeds (np array): all embeddings corresponding for the given modality
        - hash_ids (np array): np array where ith entry is the hash_id of the ith row in modal_embeds
    

    '''
    node_dict = node_dicts[modal]
    embed_dim = len(node_dict[list(node_dict.keys())[0]])
    modal_embeds = np.empty((len(node_dict.keys()), embed_dim))
    hash_ids = np.array(['']*len(node_dict.keys()), dtype=object)

    for i, key in enumerate(node_dict.keys()):
        modal_embeds[i] = node_dict[key]
        hash_ids[i] = key

    return modal_embeds, hash_ids


def get_new_edges(modal_embeds, hash_ids, sim_threshold, batch_size=500):
    new_edges = pd.DataFrame(columns=['src_id', 'dst_id'])

    for i in tqdm(range(0, len(modal_embeds), batch_size), desc='getting new similarity-based edges'):
        edges_to_add = pd.DataFrame()
        end = min(i+batch_size, len(modal_embeds))

        embeds_batch = modal_embeds[i:end]
        cosine_sims_matrix = graph_utils.get_cosine_sim(embeds_batch, modal_embeds, np.array(range(i,end)))
        
        image_matches = []
        for cosine_sims in cosine_sims_matrix:
            hash_ids_relevant = hash_ids[(cosine_sims>sim_threshold)]
            image_matches.append(hash_ids_relevant.tolist())
        
        edges_to_add['src_id'] = hash_ids[i:end]
        edges_to_add['dst_id'] = image_matches
        
        new_edges = pd.concat([new_edges, edges_to_add])
    
    new_edges = new_edges.explode('dst_id').dropna().drop_duplicates().reset_index(drop=True)

    return new_edges


def main_wrapper(org='coco', new_edge_mode=None, sim_threshold=None, new_edges_batch_size=500):
    @hydra.main(version_base=None, config_path='../../conf', config_name='config')
    def graph_builder(cfg):
        if org == 'coco':
            node_dicts = load_mscoco_nodes(cfg)
            graph_location = cfg.graph.mscoco.graph_dir
            graph_name = cfg.graph.mscoco.dataset_name
            edges_filename = cfg.graph.mscoco.edges
            nodes_filename = cfg.graph.mscoco.nodes
        
        elif org == 'zillow':
            node_dicts = load_zillow_nodes(cfg)
            graph_location = cfg.graph.zillow.graph_dir
            graph_name = cfg.graph.zillow.dataset_name
            edges_filename = cfg.graph.zillow.edges
            nodes_filename = cfg.graph.zillow.nodes
        
        else:
            raise ValueError(f'Expected org input of "coco" or "zillow", got {org}')
        
        nodes_table = get_all_graph_nodes(node_dicts)
        edges_table = get_all_graph_edges(cfg, org=org)

        if new_edge_mode == 'images' or new_edge_mode == 'keywords':
            if sim_threshold == None:
                raise ValueError('If new_edge_mode is provided, sim_threshold must be a float between 0 and 1')
            node_embeds, hash_ids = get_modal_embeds(new_edge_mode, node_dicts)
            new_edges = get_new_edges(node_embeds, hash_ids, sim_threshold, new_edges_batch_size)
            print(f'Added {len(new_edges)} new {new_edge_mode} links')
            
            edges_table = pd.concat([edges_table, new_edges]).drop_duplicates().reset_index(drop=True)

        elif new_edge_mode != None:
            raise ValueError('Invalid new_edge_mode input, expected "images" or "keywords" or None')
        
        # Store graph csv files

        if not os.path.exists(graph_location):
            os.mkdir(graph_location)

        edges_table.to_csv(graph_location + edges_filename, index=False)
        nodes_table.to_csv(graph_location + nodes_filename, index=False)

        g_metadata = {
            'dataset_name': graph_name,
            'edge_data': [{'file_name': edges_filename}],
            'node_data': [{'file_name': nodes_filename}]
        }

        with open(graph_location + 'meta.yaml', 'w') as file:
            yaml.dump(g_metadata, file)
        
        graph_dataset = dgl.data.CSVDataset(graph_location)
        
        print('Finished building graph:')
        print(graph_dataset[0])
    
    graph_builder()


if __name__ == "__main__":
    main_wrapper(org='coco', new_edge_mode='images', sim_threshold=0.95, new_edges_batch_size=500)
