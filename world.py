import os
from os.path import join
import torch
import multiprocessing

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DR-GNN")
    parser.add_argument("--trainbatch", type=int, default=2048, help="the batch size for bpr loss training procedure")
    parser.add_argument("--recdim", type=int, default=32, help="the embedding size of lightGCN")
    parser.add_argument("--layer", type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument("--lr", type=float, default=0.1, help="the learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="the weight decay for l2 normalizaton")
    parser.add_argument("--enable_dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--testbatch", type=int, default=2048, help="the batch size of users for testing")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gowalla",
        help="available datasets: [ gowalla, ml-1m, amazon-book, douban, douban622]",
    )
    parser.add_argument("--path", type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument("--topks", nargs="?", default="[20]", help="@k test list")
    parser.add_argument("--tensorboard", type=int, default=1, help="enable tensorboard")
    parser.add_argument("--comment", type=str, default="", help="comment of running")
    parser.add_argument("--load", type=int, default=0, help="whether load model weights")
    parser.add_argument("--epochs", type=int, default=1001, help="total number of epochs")
    parser.add_argument("--multicore", type=int, default=0, help="whether we use multiprocessing or not in test")
    parser.add_argument("--pretrain", type=int, default=0, help="whether we use pretrained weight or not")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--model", type=str, default="lgn")
    parser.add_argument("--loss", type=str, default="bpr", help="loss function, support [bpr, softmax]")

    parser.add_argument("--enable_DRO", type=int, default=0, help="whether using DRO weight during propagating")
    parser.add_argument("--norm_emb", type=int, default=0, help="whether normalize embeddings")
    parser.add_argument("--alpha", type=float, default=0.1, help="temperature alpha")
    parser.add_argument("--tau", type=float, default=1.0, help="hyperparameter in bprloss")

    parser.add_argument("--cuda", type=str, default="0", help="use which cuda")
    parser.add_argument("--ssm_temp", type=float, default=0.1)
    parser.add_argument("--num_negtive_items", type=int, default=1)
    parser.add_argument("--full_batch", action="store_true")
    parser.add_argument("--ood", type=str, default="popularity_shift")

    # aug_edges
    parser.add_argument("--aug_on", action="store_true")
    parser.add_argument("--aug_ratio", type=float, default=0.01)
    parser.add_argument("--aug_coefficient", type=float, default=0.1)
    parser.add_argument("--aug_warm_up", type=int, default=5)
    parser.add_argument("--aug_gap", type=int, default=5)

    return parser.parse_args()


args = parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ROOT_PATH = os.getcwd()
CODE_PATH = ROOT_PATH
# DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, "runs")
FILE_PATH = join(CODE_PATH, "checkpoints")
import sys

sys.path.append(join(CODE_PATH, "sources"))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {
    "train_batch": args.trainbatch,
    "n_layers": args.layer,
    "latent_dim_rec": args.recdim,
    "enable_dropout": args.enable_dropout,
    "keep_prob": args.keepprob,
    "test_u_batch_size": args.testbatch,
    "multicore": args.multicore,
    "loss": args.loss,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "pretrain": args.pretrain,
    "enable_DRO": args.enable_DRO,
    "alpha": args.alpha,
    "tau": args.tau,
    "norm_emb": args.norm_emb,
    "full_batch": args.full_batch,
    "num_negtive_items": args.num_negtive_items,
    "ood": args.ood,
    "cuda": args.cuda,
    "aug_on": args.aug_on,
    "aug_ratio": args.aug_ratio,
    "aug_coefficient": args.aug_coefficient,
    "aug_warm_up": args.aug_warm_up,
    "aug_gap": args.aug_gap,
    "ssm_temp": args.ssm_temp,
}

import os

CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model


TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

METHOD_CAT = None
if config["enable_DRO"]:
    METHOD_CAT = "DRO"
else:
    METHOD_CAT = "normal"


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
