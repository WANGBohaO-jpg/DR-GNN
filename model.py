import copy
import math
import os
import pdb
from matplotlib import pyplot as plt

from tqdm import tqdm
import world
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from world import cprint


class BasicModel(nn.Module):
    def __init__(self, config: dict, dataset):
        super(BasicModel, self).__init__()
        self.config = config
        self.dataset = dataset

    def getUsersRating(self, users):
        raise NotImplementedError

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset):
        super(LightGCN, self).__init__(config, dataset)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config["user_emb"]))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config["item_emb"]))
            print("use pretarined data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        self.graphdeg = self.dataset.graphdeg
        print(f"lgn is already to go(dropout:{self.config['enable_dropout']})")

    def computer(self, epoch=None, batch_i=None):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if (self.config["enable_DRO"] and self.training) or (
            self.config["aug_on"] and self.training and epoch >= self.config["aug_warm_up"]
        ):
            L_user, L_item = self.computeL(F.normalize(all_emb, p=2, dim=1), epoch=epoch)
            L = torch.cat([L_user, L_item])

            if self.config["aug_on"] and self.training and epoch >= self.config["aug_warm_up"]:
                temp_graph = self.dataset.aug_Graph
            else:
                temp_graph = self.Graph
        else:
            temp_graph = self.Graph
            L = torch.ones_like(self.Graph.values()).float().cuda()

        g_droped = (
            torch.sparse_coo_tensor(temp_graph.indices(), temp_graph.values() * L, temp_graph.size()).coalesce().cuda()
        )

        if self.config["enable_dropout"] and self.training:
            print("droping")
            g_droped = self.__dropout_x(g_droped, self.keep_prob)

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if self.config["norm_emb"]:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)

        return users, items

    def computeL(self, all_emb, epoch, batch_i=None):
        ENABLE_AUG_EDGE = self.config["aug_on"] and (epoch >= self.config["aug_warm_up"])

        if ENABLE_AUG_EDGE:
            if not self.config["full_batch"] and (
                (epoch % self.config["aug_gap"] == 0 and batch_i == 0) or not hasattr(self.dataset, "aug_res_edges")
            ):
                self.dataset.aug_edges(self.embedding_user, self.embedding_item, ratio=self.config["aug_ratio"])
            elif self.config["full_batch"] and (
                epoch % self.config["aug_gap"] == 0 or not hasattr(self.dataset, "aug_res_edges")
            ):
                self.dataset.aug_edges(self.embedding_user, self.embedding_item, ratio=self.config["aug_ratio"])
            edges, aug_edge_coe, aug_edge_coe2 = (
                self.dataset.aug_res_edges.cuda(non_blocking=True),
                self.dataset.aug_edge_coe.cuda(non_blocking=True),
                self.dataset.aug_edge_coe2.cuda(non_blocking=True),
            )
            deg1 = self.graphdeg
        else:
            edges = self.dataset.edges()
            deg1 = self.graphdeg

        U = edges[0]
        I_orig = edges[1]
        I = edges[1] + self.num_users

        if self.config["enable_DRO"]:
            f0 = -(all_emb[U] * all_emb[I]).sum(dim=1) / torch.sqrt(deg1[U] * deg1[I])
            f0 = f0.div(self.config["alpha"])
            f0 = torch.exp(f0).detach()

            if ENABLE_AUG_EDGE:
                f0 = f0 * aug_edge_coe
                f_aug_temp = f0 * aug_edge_coe2

            f_user, f_item = f0, f0
        else:
            if ENABLE_AUG_EDGE:
                f0 = torch.ones_like(U, dtype=torch.float32).cuda()
                f0 = f0 * aug_edge_coe
                f_aug_temp = f0 * aug_edge_coe2
                f_user, f_item = f0, f0
            else:
                f0 = torch.ones_like(U, dtype=torch.float32).cuda()
                f_user, f_item = f0, f0

        def cal_Ef(target, f):
            if target == "user":
                num = self.num_users
                idx = U
                deg = deg1[: self.num_users]
            elif target == "item":
                idx = I_orig
                deg = deg1[self.num_users :]
                num = self.num_items
            Ef = torch.zeros(num).cuda()
            Ef.scatter_add_(dim=0, index=idx, src=f)
            Ef = Ef / deg
            Ef = torch.where(Ef == 0, torch.tensor(1.0), Ef)
            return Ef

        if ENABLE_AUG_EDGE:
            Ef_user = cal_Ef(target="user", f=f_aug_temp)
            L_user = f_user / Ef_user[U]
        else:
            Ef_user = cal_Ef(target="user", f=f_user)
            L_user = f_user / Ef_user[U]
        L_item = L_user

        def resort_L(L, X, Y, target):
            if target == "user":
                col_len = self.num_items
            elif target == "item":
                col_len = self.num_users
            one_dim_indices = X * col_len + Y
            sorted_indices = torch.argsort(one_dim_indices)
            sorted_L = L[sorted_indices]
            return sorted_L

        L_user, L_item = resort_L(L_user, U, I_orig, "user"), resort_L(L_item, I_orig, U, "item")
        return L_user, L_item

    def getUsersRating(self, users, items_tensor=None):
        if items_tensor is None:
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            items_emb = all_items
            rating = self.f(torch.matmul(users_emb, items_emb.t()))
            return rating
        else:
            all_users, all_items = self.computer()
            rating_list = []
            for i, items in enumerate(items_tensor):
                user = users[i].long()
                items = torch.tensor(items, dtype=torch.long).cuda()

                users_emb = all_users[user].unsqueeze(0)
                items_emb = all_items[items]

                rating_list.append(self.f(torch.matmul(users_emb, items_emb.t())))
            return rating_list

    def getEmbedding(self, users, pos_items, neg_users, neg_items_list, epoch=None, batch_i=None):
        neg_items = neg_items_list.view(-1)

        all_users, all_items = self.computer(epoch, batch_i)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_users_emb = all_users[neg_users]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_users_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bprloss(self, users, pos, neg, epoch: int = None):
        neg_num = neg.shape[1]
        neg_users = users.view(-1, 1).repeat(1, neg_num).view(-1)

        (users_emb, pos_emb, neg_users_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg_users.long(), neg.long(), epoch
        )
        reg_loss = (
            (1 / 2)
            * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2) / neg_num)
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(neg_users_emb, neg_emb).sum(dim=1)

        loss = torch.mean(torch.nn.functional.softplus((neg_scores - pos_scores).div(self.config["tau"])))

        return loss, reg_loss

    def softmaxloss(self, users, pos, neg, epoch: int = None, batch_i: int = None):
        temp = self.config["ssm_temp"]
        neg_num = neg.shape[1]
        neg_users = users.view(-1, 1).repeat(1, neg_num).view(-1)

        (users_emb, pos_emb, neg_users_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg_users.long(), neg.long(), epoch, batch_i
        )
        reg_loss = (
            (1 / 2)
            * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2) / neg_num)
            / float(len(users))
        )

        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(neg_users_emb, neg_emb).sum(dim=1)
        y_pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.view(-1, neg_num)], dim=1)

        pos_logits = torch.exp(y_pred[:, 0] / temp)
        neg_logits = torch.exp(y_pred[:, 1:] / temp)
        neg_logits = torch.mean(neg_logits, dim=-1)

        loss = -torch.log(pos_logits / neg_logits).mean()

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        mul = torch.sum(inner_pro, dim=1)
        return mul


class MF(BasicModel):
    def __init__(self, config: dict, dataset):
        super(MF, self).__init__(config, dataset)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        cprint("use NORMAL distribution initilizer")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        self.graphdeg = self.dataset.graphdeg
        print(f"mf is already to go")

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        users, items = torch.split(all_emb, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bprloss(self, users, pos, neg, epoch: int = None):
        neg = neg.squeeze()

        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus((neg_scores - pos_scores).div(self.config["tau"])))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        mul = torch.sum(inner_pro, dim=1)
        return mul
