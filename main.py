# let pandas shut up
from warnings import simplefilter
import os

simplefilter(action="ignore", category=FutureWarning)

import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import procedure
from os.path import join
from logger import CompleteLogger


os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

import dataloader
import model
from pprint import pprint

dataroot = "./OOD_data/" + world.config["ood"] + "/" + world.dataset
logroot = "./log/" + world.config["ood"] + "/" + world.dataset

dataset = dataloader.Loader(path=dataroot)

print("usernum: ", dataset.n_users)
print("itemnum: ", dataset.m_items)
print("edgenum: ", dataset.traindataSize + dataset.testDataSize)
print("===========config================")
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print(f"enable DRO: {world.config['enable_DRO']}")
print("===========end===================")

MODELS = {"lgn": model.LightGCN, "mf": model.MF}

Recmodel = MODELS[world.model_name](world.config, dataset).cuda()
bpr = utils.LossFunc(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu")))
        cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

if world.config["loss"] == "bpr":
    world.config["norm_emb"] = 0
elif world.config["loss"] == "softmax":
    world.config["norm_emb"] = 1

if world.config["aug_on"]:
    world.comment += "aug_edge"

# init tensorboard
if world.tensorboard:
    save_dir = join(
        logroot,
        time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.METHOD_CAT + "-" + world.comment + "-" + world.config["loss"],
    )
    i = 0
    while os.path.exists(save_dir):
        new_save_dir = save_dir + str(i)
        i += 1
        save_dir = new_save_dir
    w: SummaryWriter = SummaryWriter(save_dir)
    logger = CompleteLogger(root=save_dir)
else:
    w = None
    cprint("not enable tensorflowboard")

print("world.config")
print(world.config)
w.add_text("config", str(world.config), 0)
print("============================================")
try:
    best_recall, best_ndcg, best_hit, best_precision = 0, 0, 0, 0
    patience, early_stop = 0, 0

    for epoch in range(world.TRAIN_epochs):
        if epoch % 5 == 0:
            cprint("[TEST]")
            test_res = procedure.Test(dataset, Recmodel, epoch, w, world.config["multicore"])
            test_recall, test_ndcg, test_hit, test_precision = (
                test_res["recall"][0],
                test_res["ndcg"][0],
                test_res["hitratio"][0],
                test_res["precision"][0],
            )

            if test_ndcg > best_ndcg:
                patience = 0
                torch.save(Recmodel.state_dict(), os.path.join(save_dir, "best_model.pth"))
            else:
                patience += 1
                if patience >= 5:
                    early_stop = True

            best_recall, best_ndcg, best_hit, best_precision = (
                max(best_recall, test_recall),
                max(best_ndcg, test_ndcg),
                max(best_hit, test_hit),
                max(best_precision, test_precision),
            )
            print(test_res)
        if early_stop:
            break
        output_information = procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, world.config, w=w)
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}")

    print(
        "best_hit: {}, best_ndcg: {}, best_precision: {}, best_recall: {}".format(
            best_hit, best_ndcg, best_precision, best_recall
        )
    )
finally:
    if world.tensorboard:
        w.close()
