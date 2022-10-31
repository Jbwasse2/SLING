from __future__ import division
from __future__ import print_function

import time
import random
from line_profiler import LineProfiler
import datetime
import pudb
import argparse
import numpy as np
import tqdm
import os

import tensorflow as tf
import subprocess
from copy import deepcopy
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel, GCNConv, CGConv
from src.functions.feat_pred_fuc.batch_traj_loader import Loader
from src.utils.cfg import input_paths

from src.functions.feat_pred_fuc.deepgcn import XRN, TopoGCN
from src.functions.feat_pred_fuc.deepgcnNRNS import XRNNRNS

# Training settings
parser = argparse.ArgumentParser()
parser = input_paths(parser)
parser.add_argument("--run_name", type=str, default="CGConv")
parser.add_argument("--tag", type=str, default="tag")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--metric_loss", action="store_true", default=False)
parser.add_argument("--intra_loss", action="store_true", default=False)
parser.add_argument("--embed_loss", action="store_true", default=False)
parser.add_argument("--intra_loss_meta", action="store_true", default=False)
parser.add_argument("--intra_triangle", action="store_true", default=False)
parser.add_argument("--intra_symm", action="store_true", default=False)
parser.add_argument("--intra_identity", action="store_true", default=False)
parser.add_argument("--expand_dataset", action="store_true", default=False)
parser.add_argument("--no_mse_loss", action="store_true", default=False)
parser.add_argument("--triangle_gts", type=int, default=1)
parser.add_argument("--clean_data", action="store_true", default=False)
parser.add_argument("--distance_out", action="store_true", default=False)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--metric_weight", type=float, default=1.0)
parser.add_argument("--intra_weight", type=float, default=1.0)
parser.add_argument("--embed_weight", type=float, default=1.0)
parser.add_argument("--id_weight", type=float, default=1.0)
parser.add_argument("--tri_weight", type=float, default=1.0)
parser.add_argument("--symm_weight", type=float, default=1.0)
parser.add_argument("--hist_dist_percentage", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--early_stopping", type=int, default=15)
parser.add_argument("--prev_weights", type=str, default="")
parser.add_argument("--model", type=str, default="XRN")
args = parser.parse_args()
args.base_dir += f"{args.dataset}/"
args.data_splits += f"{args.dataset}/"
args.run_name += f"_{args.dataset}"
args.trajectory_data_dir = f"{args.base_dir}{args.trajectory_data_dir}"
args.clustered_graph_dir = f"{args.base_dir}{args.clustered_graph_dir}"
ltime = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace("-","_").replace(".","_")
save_path_global = "../../results/organized_logs/GD/" + args.run_name + "/" + args.tag + "/" + ltime + "/"
writer = SummaryWriter(save_path_global)
#Add git stuff to folder
os.system("git rev-parse HEAD > " + save_path_global + "gitversion")
os.system("git diff > " + save_path_global + "gitdiff")
os.system("unlink last_run")
os.system("ln -sf " + str(save_path_global) + " last_run")
f = open(save_path_global + "args", "w")
f.write(str(args))
f.close()
f = open(save_path_global + "command", "w")
f.write(str(subprocess.list2cmdline(sys.argv)))
f.close()
#I ADDED THIS
args.base_dir = "../../data/bar/foo/gibson/"
args.clustered_graph_dir = "../../data/bar/foo/gibson/clustered_graph/"
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)

def load_data():
    loader = Loader(args)
    if args.test:
        loader.build_dataset(split="valUnseen")
    else:
        loader.build_dataset(split="train")
        loader.build_dataset(split="valUnseen")
    return loader

def get_aux_metrics(train_iterator, model):
    model.eval_start()
    metrics = {
            "Meta_Metrics_symmetry/average" : [],
            "Meta_Metrics_symmetry/accuracy" : [],
            "Meta_Metrics_negative/average" : [],
            "Meta_Metrics_negative/inaccuracy" : [],
            "Meta_Metrics_identity/average" : [],
            "Meta_Metrics_identity/accuracy" : [],
            "Meta_Metrics_tri/average" : [],
            "Meta_Metrics_tri/inaccuracy" : [],
            "Metrics_topk/soft" : [],
            "Metrics_topk/hard" : [],
            "Metrics_distance/err" : [],
            "Metrics_distance/acc" : [],
            "Intra_Loss_EndOfEpoch/symmetric" : [],
            "Intra_Loss_EndOfEpoch/identity" : [],
            "Intra_Loss_EndOfEpoch/triangle" : [],
            "Intra_Meta_symm/accuracy" : [],
            "Intra_Meta_symm/distance" : [],
            "Intra_Meta_tri/accuracy" : [],
            "Intra_Meta_tri/distance" : [],
            "Intra_Meta_id/accuracy" : [],
            "Intra_Meta_id/distance" : [],
            }
    hist_guess = []
    s = 0
    t = 0
    for data_list in tqdm.tqdm(train_iterator):
        y_shapes = [dl.y.shape[0] for dl in data_list]
        batch_labels = (
            torch.cat([data.y for data in data_list])
            .to(torch.float32)
        )
        pred_feat_mask_by_graph = []
        pred_feat_mask = []
        pred_feat_mask_by_datalist = {}
        offset = 0
        for counter, (shape, dl) in enumerate(zip(y_shapes, data_list)):
            pred_feat_mask_by_graph.append([i + offset for i in dl.pred_feat.tolist()])
            pred_feat_mask.extend([i + offset for i in dl.pred_feat.tolist()])
            for feat in dl.pred_feat.tolist():
                pred_feat_mask_by_datalist[feat + offset] = (counter, feat)
            offset += shape
        data_list[0].pred_feat_mask = pred_feat_mask
        data_list[0].pred_feat_mask_by_datalist = pred_feat_mask_by_datalist
        data_list[0].pred_feat_mask_by_graph = pred_feat_mask_by_graph
        pred_dist, intra_pred = model.model(data_list)
        del data_list[0].pred_feat_mask
        del data_list[0].pred_feat_mask_by_datalist
        del data_list[0].pred_feat_mask_by_graph
        symmetric_loss, identity_loss, triangle_loss, symm_acc, symm_dist, id_acc, id_dist, tri_acc, tri_dist = model.get_intra_aux_metrics(intra_pred, pred_feat_mask_by_graph, pred_feat_mask_by_datalist, get_metrics=True)
        metrics["Intra_Loss_EndOfEpoch/symmetric"].append(symmetric_loss.item())
        metrics["Intra_Loss_EndOfEpoch/identity"].append(identity_loss.item())
        metrics["Intra_Meta_symm/accuracy"].append(symm_acc)
        metrics["Intra_Meta_symm/distance"].append(symm_dist)
        metrics["Intra_Meta_id/accuracy"].append(id_acc)
        metrics["Intra_Meta_id/distance"].append(id_dist)
        if triangle_loss == None:
            pass
        else:
            metrics["Intra_Loss_EndOfEpoch/triangle"].append(triangle_loss.item())
            metrics["Intra_Meta_tri/accuracy"].append(tri_acc)
            metrics["Intra_Meta_tri/distance"].append(tri_dist)
        for d in pred_dist:
            if random.random() < model.args.hist_dist_percentage:
                hist_guess.append(d.item())
#        symmetry_average, symmetry_accuracy, negative_average, negative_accuracy, identity_average, identity_accuracy= model.get_other_metric_auxiliaries(copy_data_list, pred_dist)
#        metrics["Meta_Metrics_symmetry/average"].append(symmetry_average)
#        metrics["Meta_Metrics_symmetry/accuracy"].append(symmetry_accuracy)
#        metrics["Meta_Metrics_negative/average"].append(negative_average)
#        metrics["Meta_Metrics_negative/inaccuracy"].append(negative_accuracy)
#        metrics["Meta_Metrics_identity/average"].append(identity_average)
#        metrics["Meta_Metrics_identity/accuracy"].append(identity_accuracy)
        predictions = pred_dist.cpu().detach().squeeze(1)[pred_feat_mask]
        labels = batch_labels.cpu().detach()[pred_feat_mask]
        #err = mean(abs(d-d*))
        #acc = % of predictiosn where error abs(d-d*)<0.1
        if model.args.distance_out:
            distance = 1.0
        else:
            distance = 0.1
        error, acc = model.calculate_error(predictions, labels, distance)
        metrics["Metrics_distance/err"].append(error)
        metrics["Metrics_distance/acc"].append(acc)
        batch_labels_distance = (
            torch.cat([data.ydist for data in data_list])
            .to(torch.float32)
        )
        for mask, dl in zip(pred_feat_mask_by_graph, data_list):
            metric_distances = dl.met_distances
            a = pred_dist.squeeze(1)[mask]
            b = batch_labels_distance[mask]
            pos = dl.pos[dl.pred_feat]
            soft, hard = model.top_k_acc(a, b, model.args.distance_out)
            metrics["Metrics_topk/soft"].append(soft)
            metrics["Metrics_topk/hard"].append(hard)
        t += len(data_list)
    print(s / t)
    return metrics, hist_guess



def evaluate(model, train_iterator):
    model.state = 'val'
    model.eval_start()
    eval_loss, eval_err, eval_acc, eval_acctopk_soft, eval_acctopk_hard, inf_counter, eval_metric_loss, eval_mse_loss, eval_intra_loss, eval_symmetric_loss, eval_identity_loss, eval_triangle_loss = [], [], [], [], [], [], [], [], [], [], [], []
    model.inf_counter = 0
    for data_list in tqdm.tqdm(train_iterator):
        loss, metric_loss, mse_loss, intra_loss, symmetric_loss, identity_loss, triangle_loss = model.data_epoch(data_list, model.args, isTrain=False)
        eval_metric_loss.append(metric_loss)
        eval_mse_loss.append(mse_loss)
        eval_loss.append(loss)
        eval_intra_loss.append(intra_loss)
        eval_symmetric_loss.append(symmetric_loss)
        eval_identity_loss.append(identity_loss)
        if triangle_loss != None:
            eval_triangle_loss.append(triangle_loss)
    mode = "Val"
    eval_metrics, hist_guess = get_aux_metrics(train_iterator, model)
    writer.add_histogram("Dist/Val", np.array(hist_guess), epoch)
    for key,val in eval_metrics.items():
        if len(val) > 0:
            if isinstance(val[0], torch.Tensor):
                mean = torch.mean(torch.Tensor(val)).item()
            else:
                mean = np.mean(val)
        else:
            mean = -1
        print(key, mean)
        writer.add_scalar(str(key) + mode, mean, epoch)
    writer.add_scalar("Loss/" + mode, np.mean(eval_loss), epoch)
    writer.add_scalar("Loss/" + mode + "/mse_loss", np.mean(eval_mse_loss), epoch)
    writer.add_scalar("Loss/" + mode + "/metric_loss", np.mean(eval_metric_loss), epoch)
    writer.add_scalar("Loss/" + mode + "/mse_intra_loss", np.mean(eval_intra_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/symmetric_loss", np.mean(eval_symmetric_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/identity_loss", np.mean(eval_identity_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/triangle_loss", np.mean(eval_triangle_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/intra_meta_loss", np.mean(eval_symmetric_loss) + np.mean(eval_identity_loss) + np.mean(eval_triangle_loss), epoch)
    print(
        "Epoch: {:02d}".format(epoch + 1),
        "--Eval:",
        "eval_loss: {:.4f}".format(np.mean(eval_loss)),
        "eval_mse_loss: {:.4f}".format(np.mean(eval_mse_loss)),
        "eval_metric_loss: {:.4f}".format(np.mean(eval_metric_loss)),
        "eval_intra_loss: {:.4f}".format(np.mean(eval_intra_loss)),
        "eval_intra_meta_loss: {:.4f}".format(np.mean(eval_symmetric_loss) + np.mean(eval_identity_loss) + np.mean(eval_triangle_loss))
    )
    softResults = np.mean(eval_metrics['Metrics_topk/soft'])
    return softResults


def train(model, train_iterator):

    model.state = 'train'
    model.train_start()
    train_loss, train_err, train_acc, train_acctopk_soft, train_acctopk_hard, train_metric_loss, train_mse_loss, train_intra_loss, train_symmetric_loss, train_identity_loss, train_triangle_loss =[], [], [], [], [], [], [], [], [], [], []
    for data_list in tqdm.tqdm(train_iterator):
        loss, metric_loss, mse_loss, intra_loss, symmetric_loss, identity_loss, triangle_loss = model.data_epoch(data_list, model.args, isTrain=True)
        train_metric_loss.append(metric_loss)
        train_mse_loss.append(mse_loss)
        train_loss.append(loss)
        train_intra_loss.append(intra_loss)
        train_symmetric_loss.append(symmetric_loss)
        train_identity_loss.append(identity_loss)
        if triangle_loss != None:
            train_triangle_loss.append(triangle_loss)


    train_metric, hist_guess = get_aux_metrics(train_iterator, model)
    writer.add_histogram("Dist/Train", np.array(hist_guess), epoch)
    mode = "Train"
    for key,val in train_metric.items():
        print(key, np.mean(val))
        writer.add_scalar(str(key) + mode, np.mean(val), epoch)
    writer.add_scalar("Loss/" + mode, np.mean(train_loss), epoch)
    writer.add_scalar("Loss/" + mode + "/mse_loss", np.mean(train_mse_loss), epoch)
    writer.add_scalar("Loss/" + mode + "/metric_loss", np.mean(train_metric_loss), epoch)
    writer.add_scalar("Loss/" + mode + "/intra_loss", np.mean(train_intra_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/symmetric_loss", np.mean(train_symmetric_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/identity_loss", np.mean(train_identity_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/triangle_loss", np.mean(train_triangle_loss), epoch)
    writer.add_scalar("LossIntra/" + mode + "/intra_meta_loss", np.mean(train_symmetric_loss) + np.mean(train_identity_loss) + np.mean(train_triangle_loss), epoch)


    print(
        "Epoch: {:02d}".format(epoch + 1),
        "--Train:",
        "train_loss: {:.4f}".format(np.mean(train_loss)),
        "train_mse_loss: {:.4f}".format(np.mean(train_mse_loss)),
        "train_metric_loss: {:.4f}".format(np.mean(train_metric_loss)),
        "train_intra_loss: {:.4f}".format(np.mean(train_intra_loss)),
        "train_intra_meta_loss: {:.4f}".format(np.mean(train_symmetric_loss) + np.mean(train_identity_loss) + np.mean(train_triangle_loss))
        )

#    datasetAfter = deepcopy(train_iterator.dataset)
#    for bef, aft in zip(datasetBefore.data_list, datasetAfter.data_list):
#        assert torch.equal(bef.x, aft.x)
#        assert torch.equal(bef.y, aft.y)
#        assert torch.equal(bef.pos, aft.pos)
#        assert torch.equal(bef.edge_attr, aft.edge_attr)
#        assert torch.equal(bef.edge_index, aft.edge_index)
#        assert torch.equal(bef.edge_pos, aft.edge_pos)
#        assert torch.equal(bef.goal_feat, aft.goal_feat)
#        assert bef.met_distances == aft.met_distances
#        assert bef.num_edge_features == aft.num_edge_features
#        assert bef.num_edges == aft.num_edges
#        assert bef.num_faces == aft.num_faces
#        assert bef.num_features == aft.num_features
#        assert bef.num_node_features == aft.num_node_features
#        assert bef.num_nodes == aft.num_nodes
#        assert torch.equal(bef.pred_feat, aft.pred_feat)
    


if __name__ == "__main__":
    start_time = time.time()
    # Load data
    seed_everything(args.seed)
    loader = load_data()
    if args.test:
        val_iterator = DataListLoader(
            loader.datasets["valUnseen"], batch_size=args.batch_size, shuffle=True
        )
    else:
        train_iterator = DataListLoader(
            loader.datasets["train"], batch_size=args.batch_size, shuffle=True
        )
        val_iterator = DataListLoader(
            loader.datasets["valUnseen"], batch_size=args.batch_size, shuffle=True
        )

    # Create Model
    if args.model == "XRN":
        model = XRN(args)
    elif args.model == "XRNNorm":
        model = XRNNorm(args)
    elif args.model == "XRNNRNS":
        model = XRNNRNS(args)
        
    model.writer = writer
    # Train Model
    print("Starting Training...")
    best_val_acc = float("-inf")
    best_model = None
    patience = 0
    hist_true = []
    hist_nn_true = []
    if not args.test:
        for data_list in tqdm.tqdm(train_iterator):
            for dl in data_list:
                for key, dist in dl.met_distances.items():
                    hist_nn_true.append(dist)

                for dist in dl.y:
                    hist_true.append(dist)
        writer.add_histogram("GTDist/Train", np.array(hist_true), 0)
        writer.add_histogram("GTDist/Train", np.array(hist_true), 1)
        writer.add_histogram("GTDist/NNTrain", np.array(hist_nn_true), 0)
        writer.add_histogram("GTDist/NNTrain", np.array(hist_nn_true), 1)
    if not args.train:
        hist_true = []
        hist_nn_true = []
        for data_list in tqdm.tqdm(val_iterator):
            for dl in data_list:
                for dist in dl.y:
                    hist_true.append(dist)
                for key, dist in dl.met_distances.items():
                    hist_nn_true.append(dist)
        writer.add_histogram("GTDist/Val", np.array(hist_true), 0)
        writer.add_histogram("GTDist/Val", np.array(hist_true), 1)
        writer.add_histogram("GTDist/NNVal", np.array(hist_nn_true), 0)
        writer.add_histogram("GTDist/NNVal", np.array(hist_nn_true), 1)



    for epoch in range(args.epochs):
        model.epoch = epoch
        if args.test:
            val_acc = evaluate(model, val_iterator)
        elif args.train:
            train(model, train_iterator)
            val_acc = 0.0
        else:
            train(model, train_iterator)
            val_acc = evaluate(model, val_iterator)

        best_model = model.get_model()
        print(torch.sum(best_model.state_dict()['module.conv1.att.0.weight']))
        save_path = os.path.join(
            save_path_global,
            args.run_name + "_unseenAcc{:.2f}_epoch{}.pt".format(val_acc, epoch),
        )
        checkpoint = {
            'state_dict' : best_model.module.state_dict(),
            'epoch' : epoch,
        }

        best_val_acc = val_acc
        if not args.debug:
            torch.save(checkpoint, save_path)
        print("Saved model at:", str(save_path))
        if val_acc > best_val_acc:
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stopping:
                print("Patience reached... ended training")
                break

        print("Patience:", patience)

    print("Training Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))

    test_iterator = DataListLoader(
        loader.datasets["valUnseen"], batch_size=args.batch_size, shuffle=True
    )

    # Evaluate Best Model
    model.set_model(best_model)
    model.eval_start()
    #test_acc = evaluate(model, test_iterator)
    print("Testing Finished!")
    writer.close()
