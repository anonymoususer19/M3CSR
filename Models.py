import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

class Transformer(nn.Module):
    def __init__(self, col, nh = 4, action_item_size = 64, att_emb_size = 16):
        super(Transformer, self).__init__()
        self.nh = nh
        self.att_emb_size = att_emb_size
        self.Q = nn.Parameter(torch.empty(col, att_emb_size * nh))
        self.K = nn.Parameter(torch.empty(action_item_size, att_emb_size * nh))
        self.V = nn.Parameter(torch.empty(action_item_size, att_emb_size * nh))
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(self, query_input, action_list_input):
        querys = torch.tensordot(query_input, self.Q, dims=([-1], [0]))
        keys = torch.tensordot(action_list_input, self.K, dims=([-1], [0]))
        values = torch.tensordot(action_list_input, self.V, dims=([-1], [0]))

        querys = torch.stack(torch.chunk(querys, self.nh, dim=2)) # [-2, 1]
        keys = torch.stack(torch.chunk(keys, self.nh, dim=2))
        values = torch.stack(torch.chunk(values, self.nh, dim=2))

        inner_product = torch.matmul(querys, keys.transpose(-2, -1)) / 8.0
        normalized_att_scores = torch.nn.functional.softmax(inner_product, dim=-1)
        result = torch.matmul(normalized_att_scores, values)
        result = result.permute(1, 2, 0, 3)

        mha_result = result.reshape((query_input.shape[0], self.nh * self.att_emb_size))

        return mha_result

class M3CSR(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, dataset, image_feats, text_feats, audio_feats, cluster_num, vv_num):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.cluster_num = cluster_num
        self.vv_num = vv_num
        self.embedding_dim = embedding_dim
        self.dataset = dataset

        self.user_image_dense = nn.Linear(image_feats.shape[1], args.embed_size)
        self.photo_image_dense = nn.Linear(image_feats.shape[1], args.embed_size)
        self.photo_image_top_dense = nn.Linear(args.embed_size * 2, args.embed_size)
        self.user_text_dense = nn.Linear(text_feats.shape[1], args.embed_size)
        self.photo_text_dense = nn.Linear(text_feats.shape[1], args.embed_size)
        self.photo_text_top_dense = nn.Linear(args.embed_size * 2, args.embed_size)
        nn.init.xavier_uniform_(self.user_image_dense.weight)
        nn.init.xavier_uniform_(self.photo_image_dense.weight)
        nn.init.xavier_uniform_(self.user_text_dense.weight)
        nn.init.xavier_uniform_(self.photo_text_dense.weight)
        nn.init.xavier_uniform_(self.photo_image_top_dense.weight)
        nn.init.xavier_uniform_(self.photo_text_top_dense.weight)
        if self.dataset == 'tiktok':
            self.user_audio_dense = nn.Linear(audio_feats.shape[1], args.embed_size)
            self.photo_audio_dense = nn.Linear(audio_feats.shape[1], args.embed_size)
            self.photo_audio_top_dense = nn.Linear(args.embed_size * 2, args.embed_size)
            nn.init.xavier_uniform_(self.user_audio_dense.weight)
            nn.init.xavier_uniform_(self.photo_audio_dense.weight)
            nn.init.xavier_uniform_(self.photo_audio_top_dense.weight)

        self.user_id_embedding = nn.Embedding(n_users + 1, self.embedding_dim)
        self.photo_id_embedding = nn.Embedding(n_items + 1, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.photo_id_embedding.weight)
        
        self.image_cluster_id_embedding = nn.Embedding(self.cluster_num + 1, self.embedding_dim)
        self.text_cluster_id_embedding = nn.Embedding(self.cluster_num + 1, self.embedding_dim)
        nn.init.xavier_uniform_(self.image_cluster_id_embedding.weight)
        nn.init.xavier_uniform_(self.text_cluster_id_embedding.weight)

        if self.dataset == 'tiktok':
            self.audio_cluster_id_embedding = nn.Embedding(self.cluster_num + 1, self.embedding_dim)
            nn.init.xavier_uniform_(self.audio_cluster_id_embedding.weight)
        
        self.vv_id_embedding = nn.Embedding(self.vv_num, self.embedding_dim)
        nn.init.xavier_uniform_(self.vv_id_embedding.weight)

        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
        if self.dataset == 'tiktok':
            self.audio_feats = torch.tensor(audio_feats).float().cuda()
            self.audio_embedding = nn.Embedding.from_pretrained(torch.Tensor(audio_feats), freeze=False)

        self.user_id_dense = nn.Linear(args.embed_size * 3, args.embed_size)
        self.photo_id_dense = nn.Linear(args.embed_size * 3, args.embed_size)
        self.image_alpha_dense = nn.Linear(args.embed_size * 2, args.embed_size)
        self.text_alpha_dense = nn.Linear(args.embed_size * 2, args.embed_size)

        self.pop_dense = nn.Linear(args.embed_size, 1)

        nn.init.xavier_uniform_(self.user_id_dense.weight)
        nn.init.xavier_uniform_(self.photo_id_dense.weight)
        nn.init.xavier_uniform_(self.image_alpha_dense.weight)
        nn.init.xavier_uniform_(self.text_alpha_dense.weight)
        nn.init.xavier_uniform_(self.pop_dense.weight)

        if self.dataset == 'tiktok':
            self.audio_alpha_dense = nn.Linear(args.embed_size * 2, args.embed_size)
            nn.init.xavier_uniform_(self.audio_alpha_dense.weight)
        
        self.user_one_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.user_two_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.photo_one_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.photo_two_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.user_image_transformer = Transformer(col = 64, nh = 4, action_item_size = 128, att_emb_size = 16)
        self.user_text_transformer = Transformer(col = 64, nh = 4, action_item_size = 128, att_emb_size = 16)

        if self.dataset == 'tiktok':
            self.user_audio_transformer = Transformer(col = 64, nh = 4, action_item_size = 128, att_emb_size = 16)
 

    # user_one_hop 每个用户交互的u2i，做好填充 [[10个], [10个], ...]
    # user_two_hop 每个用户交互的u2i2u，做好填充 [[10个], [10个], ...]
    # photo_one_hop 每个视频被交互的i2u，做好填充 [[10个], [10个], ...]
    # photo_two_hop 每个视频被交互的i2u2i，做好填充 [[10个], [10个], ...]
    # user_cluster_ids 每个用户交互的u2i的聚类id，做好填充 [[10个], [10个], ...]
    # photo_cluster_id 每个视频的聚类id [id, id, ...]
    # photo_vv 每个视频的vv分层id [id, id, ...]
    def forward(self, user_one_hop, user_two_hop, photo_one_hop, photo_two_hop, user_cluster_ids, photo_cluster_id, photo_vv):
        # user tower
        ## user id
        user_one_hop_embeddings = self.photo_id_embedding(user_one_hop) # [user_num, 6, 64]
        user_two_hop_embeddings = self.user_id_embedding(user_two_hop) # [user_num, 6, 64]

        user_query = torch.reshape(self.user_id_embedding.weight.detach(), (-1, 1, 64))

        user_one_hop_emb = self.user_one_hop_transformer(user_query, user_one_hop_embeddings) # [user_num, 64]
        user_two_hop_emb = self.user_two_hop_transformer(user_query, user_two_hop_embeddings) # [user_num, 64]

        user_id_top = torch.cat((self.user_id_embedding.weight, user_one_hop_emb, user_two_hop_emb), dim = -1) # [user_num, 192]

        user_id_top = self.user_id_dense(user_id_top) # [user_num, 64]
        user_id_top_normalized = F.normalize(user_id_top, p = 2, dim = -1)

        ## user image
        user_image_emb = self.image_embedding(user_one_hop) 
        user_image_cluster_ids = self.image_cluster_id_embedding(user_cluster_ids) # [user_num, 10, 64]
        user_image_mapping = self.user_image_dense(user_image_emb) # [user_num, 10, 64]
        user_image = torch.cat((user_image_mapping, user_image_cluster_ids), dim = -1) # [user_num, 10, 128]
    
        user_image_interest = self.user_image_transformer(user_query, user_image) # [user_num, 64]

        user_image_interest_normalized = F.normalize(user_image_interest, p = 2, dim = -1)

        ## user text
        user_text_emb = self.text_embedding(user_one_hop)
        user_text_cluster_ids = self.text_cluster_id_embedding(user_cluster_ids)
        user_text_mapping = self.user_text_dense(user_text_emb)
        user_text = torch.cat((user_text_mapping, user_text_cluster_ids), dim = -1)
    
        user_text_interest = self.user_text_transformer(user_query, user_text)

        user_text_interest_normalized = F.normalize(user_text_interest, p = 2, dim = -1)

        ## user audio
        if self.dataset == 'tiktok':
            user_audio_emb = self.audio_embedding(user_one_hop)
            user_audio_cluster_ids = self.audio_cluster_id_embedding(user_cluster_ids)
            user_audio_mapping = self.user_audio_dense(user_audio_emb)
            user_audio = torch.cat((user_audio_mapping, user_audio_cluster_ids), dim = -1)
        
            user_audio_interest = self.user_audio_transformer(user_query, user_audio)

            user_audio_interest_normalized = F.normalize(user_audio_interest, p = 2, dim = -1)

        ## user modal interest weight
        rou = 0.07

        alpha_image = torch.sigmoid(self.image_alpha_dense(torch.cat((user_id_top_normalized.detach(), user_image_interest_normalized.detach()), dim = -1)))
        alpha_text = torch.sigmoid(self.text_alpha_dense(torch.cat((user_id_top_normalized.detach(), user_text_interest_normalized.detach()), dim = -1)))

        if self.dataset == 'tiktok':
            alpha_audio = torch.sigmoid(self.audio_alpha_dense(torch.cat((user_id_top_normalized.detach(), user_audio_interest_normalized.detach()), dim = -1)))

            sum_alpha = torch.exp(alpha_image / rou) + torch.exp(alpha_text / rou) + torch.exp(alpha_audio / rou)

            alpha_image = torch.exp(alpha_image / rou) / sum_alpha
            alpha_text = torch.exp(alpha_text / rou) / sum_alpha
            alpha_audio = torch.exp(alpha_audio / rou) / sum_alpha

            ## user multi-modal interest weighted
            user_image_interest_weighted = alpha_image * user_image_interest_normalized
            user_text_interest_weighted = alpha_text * user_text_interest_normalized
            user_audio_interest_weighted = alpha_audio * user_audio_interest_normalized

            user_mmu_interest = torch.cat((user_image_interest_weighted, user_text_interest_weighted, user_audio_interest_weighted), dim = -1)

        else:
            sum_alpha = torch.exp(alpha_image / rou) + torch.exp(alpha_text / rou)

            alpha_image = torch.exp(alpha_image / rou) / sum_alpha
            alpha_text = torch.exp(alpha_text / rou) / sum_alpha

            ## user multi-modal interest weighted
            user_image_interest_weighted = alpha_image * user_image_interest_normalized
            user_text_interest_weighted = alpha_text * user_text_interest_normalized

            user_mmu_interest = torch.cat((user_image_interest_weighted, user_text_interest_weighted), dim = -1)

        user_tower = torch.cat((user_id_top_normalized, user_mmu_interest), dim = -1)


        # photo tower
        ## photo id
        photo_one_hop_embeddings = self.user_id_embedding(photo_one_hop) # [photo_num, 10, 64]
        photo_two_hop_embeddings = self.photo_id_embedding(photo_two_hop) # [photo_num, 10, 64]

        photo_query = torch.reshape(self.photo_id_embedding.weight.detach(), (-1, 1, 64))

        photo_one_hop_emb = self.photo_one_hop_transformer(photo_query, photo_one_hop_embeddings) # [photo_num, 64]
        photo_two_hop_emb = self.photo_two_hop_transformer(photo_query, photo_two_hop_embeddings) # [photo_num, 64]

        photo_id_top = torch.cat((self.photo_id_embedding.weight, photo_one_hop_emb, photo_two_hop_emb), dim = -1) # [photo_num, 192]

        photo_id_top = self.photo_id_dense(photo_id_top) # [photo_num, 64]
        photo_id_top_normalized = F.normalize(photo_id_top, p = 2, dim = -1)

        ## photo image
        photo_image_emb = self.image_embedding.weight # [photo_num, 64]
        photo_image_cluster_ids = self.image_cluster_id_embedding(photo_cluster_id) # [photo_num, 64]

        photo_image_mapping = self.photo_image_dense(photo_image_emb) # [photo_num, 64]
        photo_image = torch.cat((photo_image_mapping, photo_image_cluster_ids), dim = -1) # [photo_num, 128]

        photo_image = self.photo_image_top_dense(photo_image) # [photo_num, 64]
        photo_image_normalized = F.normalize(photo_image, dim = -1)

        ## photo image
        photo_text_emb = self.text_embedding.weight
        photo_text_cluster_ids = self.text_cluster_id_embedding(photo_cluster_id)

        photo_text_mapping = self.photo_text_dense(photo_text_emb)
        photo_text = torch.cat((photo_text_mapping, photo_text_cluster_ids), dim = -1)
        
        photo_text = self.photo_text_top_dense(photo_text)
        photo_text_normalized = F.normalize(photo_text, dim = -1)

        if self.dataset == 'tiktok':
            ## photo audio
            photo_audio_emb = self.audio_embedding.weight
            photo_audio_cluster_ids = self.audio_cluster_id_embedding(photo_cluster_id)

            photo_audio_mapping = self.photo_audio_dense(photo_audio_emb)
            photo_audio = torch.cat((photo_audio_mapping, photo_audio_cluster_ids), dim = -1)
            
            photo_audio = self.photo_audio_top_dense(photo_audio)
            photo_audio_normalized = F.normalize(photo_audio, dim = -1)

            photo_mmu = torch.cat((photo_image_normalized, photo_text_normalized, photo_audio_normalized), dim = -1)
        else:
            photo_mmu = torch.cat((photo_image_normalized, photo_text_normalized), dim = -1)
        
        # gate for behavior and mmu
        b_m_weight = torch.sigmoid(self.pop_dense(self.vv_id_embedding(photo_vv)))

        photo_tower = torch.cat((b_m_weight * photo_id_top_normalized, (1 - b_m_weight) * photo_mmu), dim = -1)

        if self.dataset == 'tiktok':
            return user_tower, photo_tower, user_image_interest_weighted, user_text_interest_weighted, user_audio_interest_weighted
        else:
            return user_tower, photo_tower, user_image_interest_weighted, user_text_interest_weighted, None
