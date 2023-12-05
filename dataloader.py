from copy import deepcopy
import copy
import os
from os.path import join
import pdb
import random
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from tqdm import tqdm
import world
from world import cprint
from time import time
import cppimport


class Loader(Dataset):
    """
    Dataset type for pytorch
    Incldue graph information
    """

    def __init__(self, config=world.config, path="./data/gowalla"):
        # train or test
        cprint(f"loading [{path}]")
        self.n_user = 0
        self.m_item = 0
        self.config = config

        self.read_data(path)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.trainDataSize + self.testDataSize} interactions in total")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

        self.train_df = self._sample_pos_neg()

        self.getSparseGraph()

        self.interaction_tensor = self.create_interaction_tensor().cuda()

    def create_interaction_tensor(self):
        interaction_tensor = torch.zeros(self.n_user, self.m_item, dtype=torch.bool)
        for row in self.train_df.itertuples():
            user, pos_set = row.userId, row.pos_items
            interaction_tensor[user, list(pos_set)] = 1
        return interaction_tensor

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def read_data(self, path):
        train_file = path + "/train.txt"
        test_file = path + "/test.txt"
        self.path = path

        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split()  # l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    if len(items) == 0:
                        continue
                    uid = int(l[0])
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.traindataSize += len(items)
        self.m_item = max(trainItem)
        self.n_user = max(trainUser)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split()  # l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    if len(items) == 0:
                        continue
                    uid = int(l[0])
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    # add
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        print("Num_item: ", self.m_item)
        print("Nume_user: ", self.n_user)
        # self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainItem = np.array(trainItem)
        self.trainUser = np.array(trainUser)
        # self.testUniqueUsers = np.array(testUniqueUsers)
        self.testItem = np.array(testItem)
        self.testUser = np.array(testUser)

        # new add
        self.trainUser_tensor = torch.LongTensor(self.trainUser)
        self.trainItem_tensor = torch.LongTensor(self.trainItem)

    def _sample_pos_neg(self):
        train_df = pd.DataFrame({"userId": self.trainUser, "itemId": self.trainItem})
        train_df = train_df.groupby("userId")["itemId"].apply(set).reset_index().rename(columns={"itemId": "pos_items"})
        return train_df[["userId", "pos_items"]]

    def choose_items(self, exclude_set, num, method="method1"):
        if method == "method1":
            sample_neg = cppimport.imp("sample1")
            choose_set = list(sample_neg.choose_items(self.m_item - 1, num, exclude_set))
        elif method == "method2":
            sample_neg = cppimport.imp("sample2")
            choose_set = list(sample_neg.choose_items(self.m_item - 1, num, exclude_set))
        elif method == "method3":
            choose_set = random.choices(range(0, self.m_item), k=num)
            for i in range(len(choose_set)):
                while choose_set[i] in exclude_set:
                    choose_set[i] = random.randint(0, self.m_item - 1)
        elif method == "method4":
            choose_set = set(range(0, self.m_item)) - set(exclude_set)
            choose_set = random.choices(tuple(choose_set), k=num)
        return choose_set

    def get_train_neg_items(self, num_negatives=4):
        users, pos_items, neg_items = [], [], []
        for row in self.train_df.itertuples():
            user, pos_set = row.userId, row.pos_items
            len_pos_set = len(pos_set)
            if self.m_item < 5000:
                sample_method = "method4"
            else:
                sample_method = "method3"
            random_neg_set = self.choose_items(pos_set, num_negatives * len_pos_set, method=sample_method)
            neg_items += random_neg_set
            pos_items += list(pos_set)
            users += [int(user)] * len(pos_set)
        users, pos_items, neg_items = torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)

        return users, pos_items, neg_items

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def edges(self):
        return torch.stack(
            (torch.tensor(self.trainUser, device="cuda"), torch.tensor(self.trainItem, device="cuda")), dim=0
        )

    def aug_edges(self, user_emb, item_emb, ratio=0.1):
        print("=====Aug_edges begin=====")
        start = time()

        aug_train_User = list(self.trainUser)
        aug_train_Item = list(self.trainItem)
        edge_coe = [1.0 - self.config["aug_coefficient"]] * len(aug_train_Item)
        edge_coe2 = copy.deepcopy(edge_coe)

        deg1 = self.graphdeg
        user_emb = copy.deepcopy(user_emb.weight)
        item_emb = copy.deepcopy(item_emb.weight)
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        user_emb = user_emb / deg1[: user_emb.shape[0]].view(-1, 1)
        item_emb = item_emb / deg1[user_emb.shape[0] :].view(-1, 1)

        n = user_emb.size(0)
        sample_size = int(self.m_item * ratio)
        indices = torch.randperm(self.m_item)[:sample_size].cuda()
        sampled_item_embeddings = item_emb[indices]

        nearest_distances = torch.full((n,), float("inf"), device=user_emb.device)
        nearest_indices = torch.zeros(n, dtype=torch.long, device=user_emb.device)

        batch_size = 2048

        with torch.no_grad():
            for i in tqdm(range(0, n, batch_size)):
                user_batch = user_emb[i : i + batch_size].unsqueeze(1)  # Shape: (batch_size, 1, d)
                items = sampled_item_embeddings.unsqueeze(0)  # Shape: (1, sample_size, d)
                distances = (user_batch - items).norm(dim=2)  # Shape: (batch_size, sample_size)
                min_distances, min_indices = distances.min(dim=1)
                nearest_distances[i : i + batch_size] = torch.min(nearest_distances[i : i + batch_size], min_distances)
                nearest_indices[i : i + batch_size] = torch.where(
                    min_distances < nearest_distances[i : i + batch_size],
                    indices[min_indices],
                    nearest_indices[i : i + batch_size],
                )

        aug_train_Item.extend(nearest_indices.tolist())
        aug_train_User.extend(range(self.n_user))

        edge_coe.extend([self.config["aug_coefficient"]] * self.n_user)
        edge_coe2.extend(self.config["aug_coefficient"] * np.array(self.graphdeg_cpu[: self.n_user]))

        unique_pairs = set()
        unique_indices = []
        for idx, (user, item) in enumerate(zip(aug_train_User, aug_train_Item)):
            if (user, item) not in unique_pairs:
                unique_pairs.add((user, item))
                unique_indices.append(idx)

        aug_train_User = [aug_train_User[i] for i in unique_indices]
        aug_train_Item = [aug_train_Item[i] for i in unique_indices]
        edge_coe = [edge_coe[i] for i in unique_indices]
        edge_coe2 = [edge_coe2[i] for i in unique_indices]
        edge_coe = torch.tensor(edge_coe)
        edge_coe2 = torch.tensor(edge_coe2)

        res_edges = torch.stack(
            (torch.tensor(aug_train_User, device="cuda"), torch.tensor(aug_train_Item, device="cuda")), dim=0
        )
        self.getAugSparseGraph(res_edges)

        print("New add edges num: ", res_edges.shape[1] - len(self.trainUser))
        print("Aug_edges time: ", time() - start)
        print("======Aug_edges done!======")

        self.aug_res_edges = res_edges
        self.aug_edge_coe = edge_coe
        self.aug_edge_coe2 = edge_coe2

    def getAugSparseGraph(self, edges):
        # edges = self.edges()
        U = edges[0]
        I = edges[1]
        uI = I + self.n_users

        ind = torch.stack((torch.cat((U, uI)), torch.cat((uI, U))), dim=0)
        tempg = torch.sparse_coo_tensor(
            ind, torch.ones(ind.shape[1], device="cuda"), (self.n_users + self.m_items, self.n_users + self.m_items)
        ).coalesce()
        deg = torch.sparse.sum(tempg, dim=1).to_dense()
        deg = torch.where(deg == 0, torch.tensor(1.0), deg)
        muldeg = torch.pow(deg, -0.5)

        val = tempg.values()
        val = val * muldeg[ind[0]]
        val = val * muldeg[ind[1]]

        deg = deg.cuda()
        G = torch.sparse_coo_tensor(ind, val, (self.n_users + self.m_items, self.n_users + self.m_items))
        G = G.coalesce().cuda()

        self.aug_Graph, self.aug_graphdeg = G, deg

    def sample_edges(self, num):
        index = np.random.randint(0, self.traindataSize, num)
        S = np.zeros((num, 2))
        S[:, 0] = self.trainUser[index]
        S[:, 1] = self.trainItem[index]
        return S

    def getSparseGraph(self):
        print("loading adjacency matrix")
        edges = self.edges().cpu()
        U = edges[0]
        I = edges[1]
        uI = I + self.n_users

        ind = torch.stack((torch.cat((U, uI)), torch.cat((uI, U))), dim=0)
        tempg = torch.sparse_coo_tensor(
            ind, torch.ones(ind.shape[1]), (self.n_users + self.m_items, self.n_users + self.m_items)
        ).coalesce()
        deg = torch.sparse.sum(tempg, dim=1).to_dense()
        deg = torch.where(deg == 0, torch.tensor(1.0), deg)
        muldeg = torch.pow(deg, -0.5)

        val = tempg.values()
        val = val * muldeg[ind[0]]
        val = val * muldeg[ind[1]]

        deg = deg.cuda()
        G = torch.sparse_coo_tensor(ind, val, (self.n_users + self.m_items, self.n_users + self.m_items))
        G = G.coalesce().cuda()

        self.Graph, self.graphdeg = G, deg
        self.graphdeg_cpu = self.graphdeg.clone().cpu()
        return True

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
