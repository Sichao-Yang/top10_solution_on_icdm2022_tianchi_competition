import math
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

############################################################################################
# NARS subset aggregator
class WeightedAggregator(nn.Module):
    def __init__(self, subset_dim, feat_dim, num_hops):
        super().__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(subset_dim, feat_dim)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats):
        new_feats = []
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats
############################################################################################
# adapted from https://github.com/chennnM/GBP
class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output

# MLP applied with initial residual
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,alpha,bns=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.alpha = alpha
        self.reset_parameters()
        self.bns = bns
        self.bias = nn.BatchNorm1d(out_features)
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input ,h0):
        support = (1-self.alpha)*input+self.alpha*h0
        output = torch.mm(support, self.weight)
        if self.bns:
            output=self.bias(output)
        if self.in_features==self.out_features:
            output = output+input
        return output
############################################################################################
# feature transform nets
# adapted from dgl sign
class FeedForwardNet(nn.Module):
    def __init__(self, feat_dim, hidden, out_feats, n_layers, dropout, bns=True):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(feat_dim, out_feats))
        else:
            self.layers.append(nn.Linear(feat_dim, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.norm=bns
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers -1: 
                if self.norm:
                    x = self.dropout(self.prelu(self.bns[layer_id](x)))
                else:
                    x = self.dropout(self.prelu(x))
        return x

class FeedForwardNetII(nn.Module):
    def __init__(self, feat_dim, hidden, out_feats, n_layers, dropout,alpha,bns=False):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.feat_dim=feat_dim
        self.hidden=hidden
        self.out_feats=out_feats
        if n_layers == 1:
            self.layers.append(Dense(feat_dim, out_feats))
        else:
            self.layers.append(Dense(feat_dim, hidden))
            for i in range(n_layers - 2):
                self.layers.append(GraphConvolution(hidden, hidden, alpha, bns))
            self.layers.append(Dense(hidden, out_feats))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(self, x):
        x=self.layers[0](x)
        h0=x
        for layer_id, layer in enumerate(self.layers):
            if layer_id==0:
                continue
            elif layer_id == self.n_layers-1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x,h0)
                #x = self.dropout(self.prelu(x))
        return x
############################################################################################
class R_GAMLP(nn.Module):  # recursive GAMLP
    def __init__(self, feat_dim, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2, act="relu", pre_process=False, residual=False,pre_dropout=False,bns=False):
        super(R_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha,bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(feat_dim, hidden, hidden, 2, dropout, bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(feat_dim + feat_dim, 1)
            self.lr_output = FeedForwardNetII(
                feat_dim, hidden, nclass, n_layers_2, dropout, alpha,bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(feat_dim, hidden)
        self.residual = residual
        self.pre_dropout=pre_dropout
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        final_feat = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            final_feat = final_feat + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            final_feat += self.res_fc(feature_list[0])
            final_feat = self.dropout(self.prelu(final_feat))
        if self.pre_dropout:
            final_feat=self.dropout(final_feat)
        yhat = self.lr_output(final_feat)
        return yhat


class JK_GAMLP(nn.Module):
    def __init__(self, feat_dim, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, alpha, n_layers_1, n_layers_2, act='relu',
                 pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(JK_GAMLP, self).__init__()
        self.num_hops = num_hops
        self.prelu = nn.PReLU()
        self.pre_dropout=pre_dropout
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(feat_dim, hidden)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.residual = residual        
        if pre_process:
            self.process = nn.ModuleList([FeedForwardNet(feat_dim, hidden, hidden, n_layers_1, dropout, bns) for i in range(num_hops)])
            feat_dim2 = hidden
        else:
            feat_dim2 = feat_dim
        self.lr_jk_ref = FeedForwardNetII(num_hops*feat_dim2, hidden, hidden, n_layers_1, dropout, alpha, bns)
        self.lr_att = nn.Linear(feat_dim2 + hidden, 1)
        self.lr_output = FeedForwardNetII(feat_dim2, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        # get jumping-knowledge reference vector
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        # concat ref vec with original feature to get attention score per node per hop
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in input_list]
        # normalize over hop depth dimension
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        # combine feature vec along hop dimension using attention score
        final_feat = torch.mul(input_list[0], self.att_drop(W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            final_feat = final_feat + torch.mul(input_list[i], self.att_drop(W[:, i].view(num_node, 1)))
        # if use residual, add the preprocessed hop-0 node feature
        if self.residual:
            final_feat += self.res_fc(feature_list[0])
            final_feat = self.dropout(self.prelu(final_feat))
        if self.pre_dropout:
            final_feat=self.dropout(final_feat)
        yhat = self.lr_output(final_feat)
        return yhat


class NARS_JK_GAMLP(nn.Module):
    def __init__(self, feat_dim, hidden, nclass, num_hops, subset_dim, alpha, n_layers_1, 
                 n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, 
                 attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False,
                 pre_dropout=False,bns=False):
        super(NARS_JK_GAMLP, self).__init__()
        self.aggregator = WeightedAggregator(subset_dim, feat_dim, num_hops)
        self.model = JK_GAMLP(feat_dim, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                              alpha, n_layers_1, n_layers_2, act, pre_process, residual, pre_dropout, bns)

    def forward(self, feats_dict):
        feats = self.aggregator(feats_dict)
        out = self.model(feats)
        return out


class NARS_R_GAMLP(nn.Module):
    def __init__(self, feat_dim, hidden, nclass, num_hops, subset_dim, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False,pre_dropout=False,bns=False):
        super(NARS_R_GAMLP, self).__init__()
        self.aggregator = WeightedAggregator(subset_dim, feat_dim, num_hops)
        self.model = R_GAMLP(feat_dim, hidden, nclass, num_hops, dropout, input_drop,
                             attn_drop, alpha, n_layers_1, n_layers_2, act, pre_process, residual,pre_dropout,bns)

    def forward(self, feats_dict):
        feats = self.aggregator(feats_dict)
        out = self.model(feats)
        return out


class JK_GAMLP_RLU(nn.Module):
    def __init__(self, feat_dim, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, 
                 act, pre_process=False, residual=False, pre_dropout=False, bns=False):
        super(JK_GAMLP_RLU, self).__init__()
        self.num_hops = num_hops
        self.pre_dropout = pre_dropout
        self.prelu = nn.PReLU()
        self.res_fc = nn.Linear(feat_dim, hidden, bias=False)
        if pre_process:
            self.process = nn.ModuleList(
                [FeedForwardNet(feat_dim, hidden, hidden, 2, dropout,bns) for i in range(num_hops)])
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*hidden, hidden, hidden, n_layers_1, dropout, alpha, bns)
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns) 
        else:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*feat_dim, hidden, hidden, n_layers_1, dropout, alpha, bns)
            self.lr_att = nn.Linear(feat_dim + hidden, 1)
            self.lr_output = FeedForwardNetII(
                feat_dim, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)
        self.pre_process = pre_process
        self.label_fc = FeedForwardNet(
            nclass, hidden, nclass, n_layers_3, dropout)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.residual = residual

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_emb):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                            input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        final_feat = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            final_feat = final_feat + \
                torch.mul(input_list[i], self.att_drop(
                    W[:, i].view(num_node, 1)))
        if self.residual:
            final_feat += self.res_fc(feature_list[0])
            final_feat = self.dropout(self.prelu(final_feat))
        if self.pre_dropout:
            final_feat=self.dropout(final_feat)
        yhat = self.lr_output(final_feat)
        yhat += self.label_fc(self.label_drop(label_emb))
        return yhat


class R_GAMLP_RLU(nn.Module):  # recursive GAMLP
    def __init__(self, feat_dim, hidden, nclass, num_hops,
                 dropout, input_drop, att_dropout, label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process=False, residual=False,pre_dropout=False,bns=False):
        super(R_GAMLP_RLU, self).__init__()
        self.num_hops = num_hops
        self.pre_dropout=pre_dropout
        self.prelu = nn.PReLU()
        if pre_process:
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
            self.process = nn.ModuleList(
                [FeedForwardNet(feat_dim, hidden, hidden, 2, dropout,bns) for i in range(num_hops)])
        else:
            self.lr_att = nn.Linear(feat_dim + feat_dim, 1)
            self.lr_output = FeedForwardNetII(
                feat_dim, hidden, nclass, n_layers_2, dropout, alpha, bns)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.pre_process = pre_process
        self.res_fc = nn.Linear(feat_dim, hidden)
        self.label_drop = nn.Dropout(label_drop)
        self.residual = residual
        self.label_fc = FeedForwardNet(
            nclass, hidden, nclass, n_layers_3, dropout)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_emb):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(self.num_hops):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        final_feat = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            final_feat = final_feat + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            final_feat += self.res_fc(feature_list[0])
            final_feat = self.dropout(self.prelu(final_feat))
        if self.pre_dropout:
            final_feat=self.dropout(final_feat)
        yhat = self.lr_output(final_feat)
        yhat += self.label_fc(self.label_drop(label_emb))
        return yhat


class NARS_JK_GAMLP_RLU(nn.Module):
    def __init__(self, feat_dim, hidden, nclass, num_hops, subset_dim, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False,pre_dropout=False,bns=False):
        super(NARS_JK_GAMLP_RLU, self).__init__()
        self.aggregator = WeightedAggregator(subset_dim, feat_dim, num_hops)
        self.model = JK_GAMLP_RLU(feat_dim, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                                  label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, 
                                  pre_process, residual,pre_dropout, bns)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out = self.model(feats, label_emb)
        return out


class NARS_R_GAMLP_RLU(nn.Module):
    def __init__(self, feat_dim, hidden, nclass, num_hops, subset_dim, alpha, n_layers_1, n_layers_2, n_layers_3, act="relu", dropout=0.5, input_drop=0.0, attn_drop=0.0, label_drop=0.0, pre_process=False, residual=False,pre_dropout=False,bns=False):
        super(NARS_R_GAMLP_RLU, self).__init__()
        self.aggregator = WeightedAggregator(subset_dim, feat_dim, num_hops)
        self.model = R_GAMLP_RLU(feat_dim, hidden, nclass, num_hops, dropout, input_drop, attn_drop,
                                 label_drop, alpha, n_layers_1, n_layers_2, n_layers_3, act, pre_process, residual,pre_dropout,bns)

    def forward(self, feats_dict, label_emb):
        feats = self.aggregator(feats_dict)
        out = self.model(feats, label_emb)
        return out


# original mlp layer from NARS
class SIGNV1(nn.Module):
    def __init__(self, subset_dim, feat_dim, hidden, nclass, num_hops, 
                 n_layers_1, n_layers_2, dropout, input_drop, alpha, bns
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.fc_layers = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        self.agg_feats = WeightedAggregator(subset_dim, feat_dim, num_hops)

        for _ in range(num_hops):
            self.fc_layers.append(
                FeedForwardNet(feat_dim, hidden, hidden, n_layers_1, dropout)
            )
        self.lr_output = FeedForwardNet(
            num_hops * hidden, hidden, nclass, n_layers_2, dropout
        )

    def forward(self, feats):
        new_feats = self.agg_feats(feats)
        hidden = []
        for feat, ff in zip(new_feats, self.fc_layers):
            feat = self.input_drop(feat)
            hidden.append(ff(feat))
        out = self.lr_output(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return torch.log_softmax(out, dim=-1)

# modified mlp layer from SIGN
class SIGNV2(nn.Module):
    def __init__(self, subset_dim, feat_dim, hidden, nclass, num_hops, 
                 n_layers_1, n_layers_2, dropout, input_drop, alpha, bns
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(subset_dim, feat_dim)))
            nn.init.xavier_uniform_(self.agg_feats[-1])
        
        for _ in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(feat_dim, hidden, hidden, n_layers_1, dropout, bns)
            )
        self.lr_output = FeedForwardNetII(num_hops * hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)

    def forward(self, feats):
        # drop input feats: [Khops][Batch,Subset,Dim_feat]
        feats = [self.input_drop(x) for x in feats]
        new_feats = []
        # weight [S,D]
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        hidden = []
        for feat, ff in zip(new_feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.lr_output(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

# smoothed the classifier layer
class SIGNV3(nn.Module):
    def __init__(self, subset_dim, feat_dim, hidden, nclass, num_hops, 
                 n_layers_1, n_layers_2, dropout, input_drop, alpha, bns
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.fc_layers = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        self.encode_ffs = nn.ModuleList()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(subset_dim, feat_dim)))
            nn.init.xavier_uniform_(self.agg_feats[-1])
        
        for _ in range(num_hops):
            self.fc_layers.append(
                FeedForwardNet(feat_dim, hidden, hidden, n_layers_1, dropout, bns)
            )
        self.lr_output = nn.Sequential(
                        FeedForwardNetII(num_hops*hidden, num_hops*hidden//2, num_hops*hidden//2, n_layers_2, dropout, alpha, bns),
                        FeedForwardNetII(num_hops*hidden//2, num_hops*hidden//4, num_hops*hidden//4, n_layers_2, dropout, alpha, bns),
                        FeedForwardNetII(num_hops*hidden//4, hidden, nclass, n_layers_2, dropout, alpha, bns)
        )

    def forward(self, feats):
        # drop input feats: [Khops][Batch,Subset,Dim_feat]
        feats = [self.input_drop(x) for x in feats]
        new_feats = []
        # weight [S,D]
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        hidden = []
        for feat, ff in zip(new_feats, self.fc_layers):
            hidden.append(ff(feat))
        out = self.lr_output(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out


class SIGNV4(nn.Module):
    def __init__(self, subset_dim, feat_dim, hidden, nclass, num_hops, 
                 n_layers_1, n_layers_2, dropout, input_drop, alpha, bns, 
                 num_heads=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.fc_layers = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        self.encode_ffs = nn.ModuleList()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(subset_dim, feat_dim)))
            nn.init.xavier_uniform_(self.agg_feats[-1])
        
        for _ in range(num_hops):
            self.fc_layers.append(
                FeedForwardNet(feat_dim, hidden, hidden, n_layers_1, dropout, bns)
            )
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads)      # [Seq, Batch, Feat]
        self.lr_output = nn.Sequential(
                        FeedForwardNetII(num_hops*hidden, num_hops*hidden//2, num_hops*hidden//2, n_layers_2, dropout, alpha, bns),
                        FeedForwardNetII(num_hops*hidden//2, num_hops*hidden//4, num_hops*hidden//4, n_layers_2, dropout, alpha, bns),
                        FeedForwardNetII(num_hops*hidden//4, hidden, nclass, n_layers_2, dropout, alpha, bns)
        )

    def forward(self, feats):
        # drop input feats: [Khops][Batch,Subset,Dim_feat]
        feats = [self.input_drop(x) for x in feats]
        new_feats = []
        # subset aggragation from NARS, weight [S,D]
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        hidden = []
        # ff layer per hop feat
        for feat, ff in zip(new_feats, self.fc_layers):
            hidden.append(ff(feat))
        # attention for all K-hop features
        seq = torch.cat([h.unsqueeze(dim=0) for h in hidden], dim=0)
        attn_output = self.multihead_attn(query=seq, key=seq, value=seq, need_weights=False)[0]
        # ff layers with skip-connection
        # out = self.lr_output(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        len_seq = attn_output.shape[0]
        out = self.lr_output(self.dropout(self.prelu(torch.cat([attn_output[i,:,:] for i in range(len_seq)], dim=-1))))
        return out


# smoothed the classifier layer
class SIGNV5(nn.Module):
    def __init__(self, subset_dim, feat_dim, hidden, nclass, num_hops, 
                 n_layers_1, n_layers_2, dropout, input_drop, alpha, bns
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.fc_layers = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        self.encode_ffs = nn.ModuleList()
        self.agg_feats = nn.ParameterList()
        self.agg_hops = nn.Conv1d(in_channels=num_hops, out_channels=1, kernel_size=1)
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(subset_dim, feat_dim)))
            nn.init.xavier_uniform_(self.agg_feats[-1])
        
        for _ in range(num_hops):
            self.fc_layers.append(
                FeedForwardNet(feat_dim, hidden, hidden, n_layers_1, dropout, bns)
            )
        self.lr_output = nn.Sequential(
                        FeedForwardNetII(hidden, hidden, nclass, n_layers_2, dropout, alpha, bns)
        )

    def forward(self, feats):
        # drop input feats: [Khops][Batch,Subset,Dim_feat]
        feats = [self.input_drop(x) for x in feats]
        new_feats = []
        # weight [S,D]
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        hidden = []
        for feat, ff in zip(new_feats, self.fc_layers):
            hidden.append(ff(feat))
        
        # in:[N,Ci=num_hops,Li=feat_dim], out:[N,Co=1,Lo_feat_dim]
        seq = torch.cat([h.unsqueeze(dim=1) for h in hidden], dim=1)
        feat_agg = self.agg_hops(seq).squeeze(1)
        out = self.lr_output(self.dropout(self.prelu(feat_agg)))
        return out