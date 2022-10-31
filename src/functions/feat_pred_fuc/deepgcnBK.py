import numpy as np
from tqdm import tqdm
import time
from copy import deepcopy
import torch
import torch.nn as nn
from itertools import combinations
from pathlib import Path
import pickle
import torch.nn.functional as F
from torch_geometric.nn import DataParallel, GCNConv, CGConv
from torch_geometric.nn.conv import GCN2Conv
from torch.nn import Linear as Lin, ReLU, BatchNorm1d, LayerNorm
from src.functions.feat_pred_fuc.layers import AttentionConv
import random
import scipy.sparse
from scipy.sparse import dok_matrix

class TopoGCN(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, pos_size=3, edge_size=16):
        super(TopoGCN, self).__init__()
        """v1: graph attention layers"""
        self.conv1 = AttentionConv(input_size, dropout=0.0)
        self.conv2 = AttentionConv(input_size, dropout=0.0)
        self.conv3 = GCNConv(input_size, hidden_size)
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.distance_layers = nn.Sequential(
            Lin(hidden_size + input_size, input_size),  
            ReLU(True),
            Lin(input_size, 1),  
        )
        self.node_node_distance_layers = nn.Sequential(
            Lin(2 * hidden_size, input_size),  
            BatchNorm1d(input_size),
#            LayerNorm(input_size),
            ReLU(True),
            Lin(input_size, 1),  
            ReLU(True)
        )

    """v2: graph conv layers"""
    def forward(self, data):
        time_start = time.time()
        num_nodes = data.batch.size()[0]
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv3(x, data.edge_index))
        pred_dist = self.distance_layers(
            torch.cat((x, data.goal_feat), dim=1)
        )
        mask = None
        try:
            mask = data.pred_feat_mask[0]
            mask_list = data.pred_feat_mask_by_datalist[0]
            mask_graph = data.pred_feat_mask_by_graph[0]
        except Exception as e:
            pass
        if mask is None:
            return pred_dist, x

        else:
            x_masked = x[mask]
            pairs = []
            pairs_embeds = []
            for mask_g in mask_graph:
                for mask_i in mask_g:
                    for mask_j in mask_g:
                        embed_i = x_masked[mask.index(mask_i)]
                        embed_j = x_masked[mask.index(mask_j)]
                        pairs_embeds.append(torch.cat((embed_i, embed_j)))
                        pairs.append((mask_i, mask_j))

            pairs_embeds = torch.vstack(pairs_embeds)
            dist_pred = self.node_node_distance_layers(pairs_embeds)
            return pred_dist, (dist_pred, pairs)



class XRN(object):
    def __init__(self, args):
        self.epoch = -1
        self.state = None
        self.writer = None
        self.args = args
        self.dist_criterion = torch.nn.MSELoss()
        self.embed_criterion = torch.nn.MSELoss()
        self.class_criterion = torch.nn.CrossEntropyLoss()
        self.intra_dist_criterion = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.prev_weights != "":
            w = torch.load(args.prev_weights)
            if isinstance(w, dict):
                self.model = TopoGCN()
                self.model.load_state_dict(w['state_dict'])
                self.model = DataParallel(self.model)
            else:
                self.model = w
        else:
            self.model = TopoGCN()
            self.model = DataParallel(self.model)
            self.model = self.model.to(self.device)
        self.learning_rate = 0.001
        self.inf_counter = 0
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.learning_rate
        )
#        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
#            optimizer=self.optimizer, gamma=0.96
#        )


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train_start(self):
        self.model.train()

    def eval_start(self):
        self.model.eval()

    def calculate_error(self, output, dist_score, distance):
        error = []

        for pred, true in zip(output, dist_score):
            error.append(abs(true - pred).item())
        acc = np.mean(np.where(np.asarray(error) <= distance, 1, 0))
        error = np.mean(error)
        return error, acc

    def top_k_acc(self, output, dist_score, dist_out):
        if dist_out:
            smallest_dist = dist_score.min().item()
            smallest_pred_dist_loc = output.argmin().item()
            smallest_pred_dist = dist_score[smallest_pred_dist_loc].item()
            if np.abs(smallest_dist - smallest_pred_dist) <= 0.25:
                soft = 1
            else:
                soft = 0
            if output.argmin() == dist_score.argmin():
                hard= 1
            else:
                hard= 0
        else:
            if output.argmax() == dist_score.argmin():
                hard = 1
            else:
                hard = 0
            smallest_dist = dist_score.max().item()
            smallest_pred_dist_loc = output.argmax().item()
            smallest_pred_dist = dist_score[smallest_pred_dist_loc].item()

            smallest_dist = dist_score.min().item()
            smallest_pred_dist_loc = output.argmax().item()
            smallest_pred_dist = dist_score[smallest_pred_dist_loc].item()
            if np.abs(smallest_dist - smallest_pred_dist) <= 0.25:
                soft = 1
            else:
                soft = 0
        return soft, hard

    def get_other_metric_auxiliaries(self,data_list, pred):
#        print("FIX")
#        foo = torch.tensor(-1.0).to(self.device)
#        return foo, foo, foo, foo, foo, foo
        #Get symmetry stuff
        pred = pred.cpu().detach().squeeze(1)
        for dl in data_list:
            x = dl.x
            dl.x = dl.goal_feat
            dl.goal_feat = x
        pred_dist_symm, _ = self.model(data_list)
        symmetry = [torch.abs(i - j).item() for i, j in zip(pred, pred_dist_symm)]
        symmetry_average = np.mean(symmetry)
        symmetry_accuracy = len(np.where(np.abs(symmetry) < 0.1)[0]) / len(symmetry)
        #identity property
        for dl in data_list:
            dl.x = dl.goal_feat
        identity, _ = self.model(data_list)
        identity_average = torch.mean(identity).item()
        identity_accuracy = len(torch.where(torch.abs(identity) < 0.1)[0]) / len(identity)
        #Get negative predictions stuff
        negative_accuracy = len(np.where(pred < 0.0)[0]) / len(symmetry)
        negatives = pred[np.where(pred < 0.0)[0]]
        if len(negatives) == 0:
            negative_average = 0
        else:
            negative_average = torch.mean(negatives).item()
        return symmetry_average, symmetry_accuracy, negative_average, negative_accuracy, identity_average, identity_accuracy

    def get_intra_aux_metrics(self,  pred_dist, pred_feat_mask_by_graph, pred_feat_mask_by_datalist, get_metrics = False):
        epsilon = 0.001
        penalty = 5
        (dist_pred, pairs_embeds) = pred_dist
        mse = torch.nn.MSELoss()
        dist_pred = dist_pred.flatten()
        size = np.max(pairs_embeds) + 1
        node_by_graph = {}
        for key, item in pred_feat_mask_by_datalist.items():
            if item[0] in node_by_graph:
                node_by_graph[item[0]].append(key)
            else:
                node_by_graph[item[0]] = [key]
        node_pairs_to_dist = {}
        for pair, dist in zip(pairs_embeds, dist_pred):
            node_pairs_to_dist[pair] = dist

        symmetric_loss_local = []
        a = []
        b = []
        symm_embeds = {}
        for i,j in pairs_embeds:
            if (j,i) not in symm_embeds and i != j:
                symm_embeds[(i,j)] = True
        symm_embeds = list(symm_embeds.keys())
        for i,j in symm_embeds:
            a.append(node_pairs_to_dist[i, j])
            b.append(node_pairs_to_dist[j, i])
        a = torch.stack(a)
        b = torch.stack(b)
        symmetric_loss = mse(a,b)
        rymmetric_metrics = (a-b)**2
        if get_metrics:
            bads = torch.where(symmetric_metrics > 0.1)[0]
            symmetric_acc = 1 - (len(bads) / len(symmetric_metrics))
            if len(bads) > 0:
                symmetric_dist = torch.mean(torch.tensor(symmetric_metrics)[bads]).item()
            else:
                symmetric_dist = 0
        identity_loss_local = []
        for i in range(size):
            if (i,i) in node_pairs_to_dist:
                dist = node_pairs_to_dist[i, i]
                identity_loss_local.append((dist)**2)
        identity_loss = torch.mean(torch.stack(identity_loss_local), dim=0)
        if get_metrics:
            bads = torch.where(identity_loss > 0.1)[0]
            id_acc = 1 - (len(bads) / len(identity_loss_local))
            if len(bads) > 0:
                id_dist = torch.mean(torch.tensor(identity_loss_local)[bads]).item()
            else:
                id_dist = 0
        triangle_loss = []
        dxys, dxzs, dyzs = [], [], []
        for top_map, nodes in node_by_graph.items():
            pairs = list(combinations(nodes,3))
            #There should be a way to do as one big array!
            for pair in pairs:
                dxys.append(node_pairs_to_dist[pair[0], pair[1]])
                dxzs.append(node_pairs_to_dist[pair[0], pair[2]])
                dyzs.append(node_pairs_to_dist[pair[1], pair[2]])
        if len(dxys) == 0:
            triangle_loss = None
            if get_metrics:
                tri_acc = None
                tri_dist = None
        else:
            dxys = torch.stack(dxys)
            dxzs = torch.stack(dxzs)
            dyzs = torch.stack(dyzs)
            x = (dxys - (dxzs + dyzs) + penalty * epsilon) / (epsilon + C)
            y = (dxzs - (dxys + dyzs) + penalty * epsilon) / (epsilon + C)
            z = (dyzs - (dxys + dxzs) + penalty * epsilon) / (epsilon + C)
            triangle_loss = torch.vstack([x,y,z]).T
            triangle_loss = torch.nn.functional.relu(triangle_loss)
            triangle_loss = torch.sum(triangle_loss, dim=1)
            if get_metrics:
                bads = torch.where(triangle_loss > 0.1)[0]
                tri_acc = 1 - (len(bads) / len(triangle_loss))
                if len(bads) > 0:
                    tri_dist = torch.mean(torch.tensor(triangle_loss)[bads]).item()
                else:
                    tri_dist = 0
            triangle_loss = mse(triangle_loss, torch.zeros_like(triangle_loss))
        if get_metrics:
            return symmetric_loss, identity_loss, triangle_loss, symmetric_acc, symmetric_dist, id_acc, id_dist, tri_acc, tri_dist
        else:
            return symmetric_loss, identity_loss, triangle_loss


    def metric_loss(self, metric_distances, pos, pred, truth, pred_dxy = None):
        #metrics are d(*,*) >= 0
        #mask_neg = torch.where(pred <= 0)[0]
        #loss_neg = torch.abs(torch.sum(pred[mask_neg]))
        #identity of indiscernibiles
        #d(x,y) = 0 <=> x=y
        #TODO
        
        #symmetry
        #d(x,y) = d(y,x)
        #TODO

        #Triangle Inequality
        #d(x,y) <= d(x,z) + d(z,y)
        #let x and y be two points in the GNN we predicted distances for
        #let z be the goal point
        #assume symmetry
        #TODO
        triangle_metric = []
        triangle_loss = torch.tensor(0.0).to(self.device)
        inf_counter = 0
        for p1, dxz, dxz_true in zip(pos, pred, truth):
            for p2, dyz, dyz_true in zip(pos, pred, truth):
                p1_t = tuple(p1.tolist())
                p2_t = tuple(p2.tolist())
#                dxy = torch.FloatTensor([metric_distances[(p1_t,p2_t)]]).to(self.device)
                if pred_dxy:
                    pu.db
                else:
                    dxy = metric_distances[(p1_t,p2_t)]
                if dxy == np.inf:
                    inf_counter += 1
                    continue
                if self.args.triangle_gts == 2:
                    dyz = dyz_true
                elif self.args.triangle_gts == 3:
                    dyz = dyz_true
                    dxz = dxz_true
                a = triangle_loss.item()
                triangle_loss += torch.nn.functional.relu( dxy - ( dxz + dyz))
                triangle_loss += torch.nn.functional.relu( dxz - ( dxy + dyz))
                triangle_loss += torch.nn.functional.relu( dyz - ( dxz + dxy))
                b = triangle_loss.item()
                if a != b:
                    triangle_metric.append(b - a)
        if len(triangle_metric) == 0:
            triangle_average = 0
        else:
            triangle_average = np.mean(triangle_metric).item()

        triangle_accuracy = len(triangle_metric) / (len(pos)**2)
        triangle_loss /= (len(pos)**2)
        return triangle_loss, inf_counter, triangle_average, triangle_accuracy


    def get_embed_loss(self, data_list):    
        embed_feat_nodes = [list(set(list(range(dl.x.shape[0]))) - set(dl.pred_feat.tolist())) for dl in data_list]
        offset = 0
        gt_embeds = []
        gt_embeds_id = []
        for feats, dl in zip(embed_feat_nodes, data_list):
            random_guess = random.sample(feats, k=1)[0]
            gt_embed = deepcopy(dl.x[random_guess])
            gt_embeds.append(gt_embed)
            dl.x[random_guess] = torch.zeros_like(gt_embed)
            gt_embeds_id.append(random_guess + offset)
            offset += dl.x.shape[0]
            assert torch.sum(gt_embed) > 0.01
        _ , embed = self.model(data_list)
        gt_embeds = torch.stack(gt_embeds).to(self.device)
        embed_true = embed[gt_embeds_id]
        return self.embed_criterion(gt_embeds, embed_true)
    
    def intra_loss(self, pred, data_list, pred_feat_mask):
        (dist_pred, pairs_embeds) = pred
        gt_distance = []
        mask = []
        for counter, embeds in enumerate(pairs_embeds):
            embed1 = pred_feat_mask[embeds[0]]
            pos1 = tuple(data_list[embed1[0]].pos[embed1[1]].tolist())
            embed2 = pred_feat_mask[embeds[1]]
            pos2 = tuple(data_list[embed2[0]].pos[embed2[1]].tolist())
            distance = data_list[embed1[0]].met_distances[(pos1,pos2)]
            if distance != np.inf:
                gt_distance.append(float(distance))
                mask.append(counter)

        dist_pred = dist_pred.flatten()[mask]
        dist = []
        for d in dist_pred:
            dist.append(d.item())
        self.writer.add_histogram("Dist/NN" + self.state, np.array(dist), self.epoch)

        gt_distance = torch.FloatTensor(gt_distance).to(self.device)
        loss = self.intra_dist_criterion(gt_distance, dist_pred)
        return loss

    def data_epoch(self, data_list, args, isTrain):
        lossesUsed = []
        if isTrain:
            self.optimizer.zero_grad()
        y_shapes = [dl.y.shape[0] for dl in data_list]
        batch_labels = (
            torch.cat([data.y for data in data_list])
            .to(self.device)
            .to(torch.float32)
        )

        offset = 0
        intra_loss = torch.tensor(0.0).to(self.device)
        loss = torch.tensor(0.0).to(self.device)
        pred_feat_mask = []
        pred_feat_mask_by_graph = []
        pred_feat_mask_by_datalist = {}
        for counter, (shape, dl) in enumerate(zip(y_shapes, data_list)):
            pred_feat_mask.extend([i + offset for i in dl.pred_feat.tolist()])
            pred_feat_mask_by_graph.append([i + offset for i in dl.pred_feat.tolist()])
            for feat in dl.pred_feat.tolist():
                pred_feat_mask_by_datalist[feat + offset] = (counter, feat)
            offset += shape
        data_list[0].pred_feat_mask = pred_feat_mask
        data_list[0].pred_feat_mask_by_datalist = pred_feat_mask_by_datalist
        data_list[0].pred_feat_mask_by_graph = pred_feat_mask_by_graph
        assert not (args.intra_loss and args.embed_loss) 
        assert not (args.intra_loss_meta and args.embed_loss) 
        pred_dist, intra_pred= self.model(data_list)
        del data_list[0].pred_feat_mask
        del data_list[0].pred_feat_mask_by_datalist
        del data_list[0].pred_feat_mask_by_graph
        intra_loss = self.intra_loss( intra_pred, data_list, pred_feat_mask_by_datalist)
        if args.intra_loss:
            loss += args.intra_weight * intra_loss
            lossesUsed.append("NN_mse")

        
        intra_meta_loss = torch.tensor(0.0).to(self.device)
        symmetric_loss, identity_loss, triangle_loss = self.get_intra_aux_metrics(intra_pred, pred_feat_mask_by_graph, pred_feat_mask_by_datalist)
        if args.intra_symm:
            loss += symmetric_loss
            lossesUsed.append("intra_symm")
        if args.intra_identity:
            loss += identity_loss
            lossesUsed.append("intra_ident")
        if args.intra_triangle and triangle_loss != None:
            loss += triangle_loss
            lossesUsed.append("intra_tri")
        pred_feat_mask = np.array(pred_feat_mask)
        a = pred_dist.squeeze(1)[pred_feat_mask]
        b = batch_labels[pred_feat_mask]
        mse_loss = self.dist_criterion(
                    a,
                    b
            )
        if not args.no_mse_loss:
            loss += mse_loss
            lossesUsed.append("NG_mse")
        mse_loss = mse_loss.item()
        sum_metric_loss = torch.tensor(0.0).to(self.device)
        sum_metric_loss = (sum_metric_loss) / (len(data_list))
        if args.metric_loss:
            loss +=  sum_metric_loss * args.metric_weight
            lossesUsed.append("NG_triangle")
        if isinstance(sum_metric_loss, torch.Tensor):
            sum_metric_loss = sum_metric_loss.item()
        else:
            sum_metric_loss = sum_metric_loss
        if isTrain and loss.requires_grad:
            loss.backward()
            self.optimizer.step()
        if triangle_loss == None:
            return loss.item(), sum_metric_loss, mse_loss, intra_loss.item(), symmetric_loss.item(), identity_loss.item(), None
        else:
            return loss.item(), sum_metric_loss, mse_loss, intra_loss.item(), symmetric_loss.item(), identity_loss.item(), triangle_loss.item()
