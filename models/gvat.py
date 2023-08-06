import torch
from torch.nn import Module, ModuleList, LeakyReLU, LayerNorm, Linear
from torch_scatter import scatter_sum
from math import pi as PI
import torch.nn as nn
EPS = 1e-6


def get_encoder(config, num_edge_types):

    return CFTransformerEncoderVN(
        node_hiddens = [config.node_hiddens, config.node_hiddens_vec],
        edge_hidden = config.edge_hidden,
        key_channels = config.key_channels,  # not use
        num_heads = config.num_heads,  # not use
        num_interactions = config.num_interactions,
        k = config.knn,
        cutoff = config.cutoff,
        num_edge_types = num_edge_types
    )


class CFTransformerEncoderVN(Module):
    
    def __init__(self, node_hiddens=[256, 64], edge_hidden=64, num_edge_types=5, key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.node_hiddens = node_hiddens
        self.edge_hidden = edge_hidden
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                node_hiddens=node_hiddens,
                edge_hidden=edge_hidden,
                num_edge_types=num_edge_types,
                key_channels=key_channels,
                num_heads=num_heads,
                cutoff = cutoff
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.node_hiddens[0]
    
    @property
    def out_vec(self):
        return self.node_hiddens[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):

        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h

class AttentionInteractionBlockVN(Module):

    def __init__(self, node_hiddens, edge_hidden, num_edge_types, key_channels, num_heads=1, cutoff=10.):
        super().__init__()
        self.num_heads = num_heads
        # edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_hidden - num_edge_types)
        self.vector_expansion = EdgeExpansion(edge_hidden)  # Linear(in_features=1, out_features=edge_hidden, bias=False)
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(node_hiddens[0], node_hiddens[1], edge_hidden, edge_hidden,
                                                                                node_hiddens[0], node_hiddens[1], cutoff)

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(node_hiddens[0], node_hiddens[1], node_hiddens[0], node_hiddens[1])
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(node_hiddens[1])
        self.out_transform = GVLinear(node_hiddens[0], node_hiddens[1], node_hiddens[0], node_hiddens[1])

        self.layernorm_sca = LayerNorm([node_hiddens[0]])
        self.layernorm_vec = LayerNorm([node_hiddens[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector) 

        msg_j_sca, msg_j_vec = self.message_module(x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  #.view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  #.view(N, -1, 3) # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class MessageModule(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = GVLinear(node_sca, node_vec, out_sca, out_vec)
        self.edge_gvp = GVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec)

        self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gvlienar = GVLinear(out_sca, out_vec, out_sca, out_vec)

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=None, annealing=False):
        node_scalar, node_vector = self.node_gvlinear(node_features)
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]
        edge_scalar, edge_vector = self.edge_gvp(edge_features)

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector

        output = self.out_gvlienar((y_scalar, y_vector))

        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]   # (A, 1)
        return output


class GVPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = GVLinear(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        sca, vec = self.gv_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class EdgeExpansion(nn.Module):
    def __init__(self, edge_hidden):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_hidden, bias=False)
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GVLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vector = VNLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_vector = VNGroupLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_scalar = Conv1d(in_scalar + dim_hid, out_scalar, 1, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)  # (N_samples, dim_hid)
        feat_scalar_cat = torch.cat([feat_vector_norm, feat_scalar], dim=-1)  # (N_samples, dim_hid+in_scalar)

        out_scalar = self.lin_scalar(feat_scalar_cat)
        out_vector = self.lin_vector2(feat_vector_inter)

        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim = -1)
        out_vector = gating * out_vector
        return out_scalar, out_vector


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x_out = self.map_to_feat(x.transpose(-2,-1)).transpose(-2,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        d = self.map_to_dir(x.transpose(-2,-1)).transpose(-2,-1)  # (N_samples, N_feat, 3)
        dotprod = (x*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        x_out = (self.negative_slope * x +
                (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d)))
        return x_out
