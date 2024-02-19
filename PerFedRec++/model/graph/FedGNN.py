import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import *
from util.loss_torch import bpr_loss,l2_reg_loss
import random
import copy
from base.torch_interface import TorchGraphInterface
from util.conf import OptionConf
from data.augmentor import GraphAugmentor


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class FedGNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set,valid_set):
        super(FedGNN, self).__init__(conf, training_set, test_set,valid_set)
        args = OptionConf(self.config['FedGNN'])
        self.n_layers = int(args['-n_layer'])
        self.model = FedGNN_LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.msg = conf['training.set']

    def train(self):
        model = self.model.cuda()
        model_para_list = []
        N_client = 256
        self.N_client = N_client
        loc, scale = 0., 0.1
        delta = 0.3
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate*N_client)
        for epoch in range(self.maxEpoch):
            losses = []
            if epoch >= 0:
                user_list = list(self.data.user.keys())
                random.shuffle(user_list)
                select_user_list = user_list[:N_client]
                not_select_user_list=user_list[N_client:]
                select_user_list_num = [self.data.user[_] for _ in select_user_list]
                not_select_user_list_num = [self.data.user[_] for _ in not_select_user_list]

            for n, batch in enumerate(next_batch_pairwise_fl_pse(self.data, self.batch_size, select_user_list)):
                model_ini = copy.deepcopy(model.state_dict())
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model(perturbed=False)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                losses.append(batch_loss.item())
                model_aft = copy.deepcopy(model.state_dict())
                model_para_list += [model_aft]
                model.load_state_dict(model_ini)
  
            print('Avg Loss:', sum(losses)/len(losses))
            w_ = FedAvg(model_para_list)
            model.load_state_dict(w_)
            model_para_list = []
            # LDP
            add_noise = True
            if add_noise:
                i_random_noise = torch.tensor(np.random.laplace(loc=loc, scale=scale, size=(N_client,rec_item_emb.shape[0],rec_item_emb.shape[1])) )
                i_random_noise = torch.mean(i_random_noise, dim=0).float().to('cuda')
                model.add_noise_(i_random_noise)

            with torch.no_grad():
                self.user_emb, self.item_emb = model.get_emb()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = copy.deepcopy(self.model.get_emb())

    def get_client_mat(self, drop_client_list):
        dropped_mat = None
        dropped_mat_ = GraphAugmentor.client_select_drop(self.data.interaction_mat, drop_client_list)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat_)
        return dropped_mat_, TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class FedGNN_LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(FedGNN_LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def get_emb(self):
        return self.embedding_dict['user_emb'].data, self.embedding_dict['item_emb'].data
    
    def set_emb(self, emb):
        self.embedding_dict['item_emb'] = torch.nn.Parameter(emb)

    def add_noise_(self, noise):
        self.embedding_dict['item_emb'].data = self.embedding_dict['item_emb'].data + noise

    def forward(self, perturbed=False, perturbed_adj=None):
        self.eps=0.1

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda() #torch.sign(ego_embeddings) *
                ego_embeddings +=  F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings


