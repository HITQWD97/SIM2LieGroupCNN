from batchnorm import MaskBatchNormNd
import numpy as np
import torch
import torch.nn as nn
from utils import Named, export, Pass, Expression
from lie_group import SIM2
from utils import FarthestSubsample, knn_point, index_points


@export
def Swish():
    return Expression(lambda x: x * torch.sigmoid(x))


def LinearBNact(chin, chout, act='swish', bn=True):
    assert act in ('relu', 'swish')
    normlayer = MaskBatchNormNd(chout)
    return nn.Sequential(
        Pass(nn.Linear(chin, chout), dim=1),
        normlayer if bn else nn.Sequential(),
        Pass(Swish() if act == 'swish' else nn.ReLU(), dim=1))

def WeightNet(in_dim, out_dim, act, bn, k=32):
    return nn.Sequential(
        *LinearBNact(in_dim, k, act, bn),
        *LinearBNact(k, k, act, bn),
        *LinearBNact(k, out_dim, act, bn))



class PointConv(nn.Module):
    def __init__(self, chin, chout, mc_samples=32, xyz_dim=3, ds_frac=1, knn_channels=None, act='swish', bn=False,
                 mean=False):
        super().__init__()
        self.chin = chin
        self.cmco_ci = 16
        self.xyz_dim = xyz_dim
        self.knn_channels = knn_channels
        self.mc_samples = mc_samples
        self.weightnet = WeightNet(xyz_dim, self.cmco_ci, act, bn)
        self.linear = nn.Linear(self.cmco_ci * chin, chout)
        self.mean = mean
        assert ds_frac == 1
        self.subsample = FarthestSubsample(ds_frac, knn_channels=knn_channels)

    def extract_neighborhood(self, inp, query_xyz):
        inp_xyz, inp_vals, mask = inp
        neighbor_idx = knn_point(min(self.mc_samples, inp_xyz.shape[1]),
                                 inp_xyz[:, :, :self.knn_channels], query_xyz[:, :, :self.knn_channels], mask)
        neighbor_xyz = index_points(inp_xyz, neighbor_idx)
        neighbor_values = index_points(inp_vals, neighbor_idx)
        neighbor_mask = index_points(mask, neighbor_idx)
        return neighbor_xyz, neighbor_values, neighbor_mask

    def point_convolve(self, embedded_group_elems, nbhd_vals, nbhd_mask):
        bs, m, nbhd, ci = nbhd_vals.shape
        _, penult_kernel_weights, _ = self.weightnet((None, embedded_group_elems, nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1), penult_kernel_weights,
                                              torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1), nbhd_vals, torch.zeros_like(nbhd_vals))
        partial_convolved_vals = (nbhd_vals_m.transpose(-1, -2) @ penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1, keepdim=True).clamp(min=1)
        return convolved_vals

    def get_embedded_group_elems(self, output_xyz, nbhd_xyz):
        return output_xyz - nbhd_xyz

    def forward(self, inp):
        query_xyz, sub_vals, sub_mask = self.subsample(inp)
        nbhd_xyz, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_xyz)
        deltas = self.get_embedded_group_elems(query_xyz.unsqueeze(2), nbhd_xyz)
        convolved_vals = self.point_convolve(deltas, nbhd_vals, nbhd_mask)
        convolved_wzeros = torch.where(sub_mask.unsqueeze(-1), convolved_vals, torch.zeros_like(convolved_vals))
        return query_xyz, convolved_wzeros, sub_mask


def FPSindices(dists, frac, mask):
    m = int(np.round(frac * dists.shape[1]))
    device = dists.device
    bs, n = dists.shape[:2]
    chosen_indices = torch.zeros(bs, m, dtype=torch.long, device=device)
    distances = torch.ones(bs, n, device=device) * 1e8
    a = torch.randint(0, n, (bs,), dtype=torch.long, device=device)
    idx = a % mask.sum(-1) + torch.cat([torch.zeros(1, device=device).long(), torch.cumsum(mask.sum(-1), dim=0)[:-1]],
                                       dim=0)
    farthest = torch.where(mask)[1][idx]
    B = torch.arange(bs, dtype=torch.long, device=device)
    for i in range(m):
        chosen_indices[:, i] = farthest
        dist = torch.where(mask, dists[B, farthest],
                           -100 * torch.ones_like(distances))
        closer = dist < distances
        distances[closer] = dist[closer]
        farthest = torch.max(distances, -1)[1]
    return chosen_indices


class FPSsubsample(nn.Module):
    def __init__(self, ds_frac, cache=False, group=None):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache = cache
        self.cached_indices = None
        self.group = group

    def forward(self, inp,  withquery=False):
        abq_pairs, vals, mask = inp

        dist = self.group.distance if self.group else lambda ab: norm(ab, dim=-1)
        if self.ds_frac != 1:
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = FPSindices(dist(abq_pairs), self.ds_frac, mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = FPSindices(dist(abq_pairs), self.ds_frac, mask)

            B = torch.arange(query_idx.shape[0], device=query_idx.device).long()[:, None]
            subsampled_abq_pairs = abq_pairs[B, query_idx][B, :, query_idx]
            subsampled_values = vals[B, query_idx]
            subsampled_mask = mask[B, query_idx]
        else:
            subsampled_abq_pairs = abq_pairs
            subsampled_values = vals
            subsampled_mask = mask
            query_idx = None
        if withquery: return (subsampled_abq_pairs, subsampled_values, subsampled_mask, query_idx)
        return (subsampled_abq_pairs, subsampled_values, subsampled_mask)


class LieConv(PointConv):
    def __init__(self, *args, imageh, group=SIM2, ds_frac=1, fill=1 / 3, cache=False, knn=False, **kwargs):
        kwargs.pop('xyz_dim', None)
        super().__init__(*args, xyz_dim=group.lie_dim + 2 * group.q_dim, **kwargs)
        self.imageh = imageh
        self.group = group
        self.register_buffer('r', torch.tensor(2.))
        self.fill_frac = min(fill,
                             1.)
        self.knn = knn
        self.subsample = FPSsubsample(ds_frac, cache=cache, group=self.group)
        self.coeff = .5
        self.fill_frac_ema = fill

    def extract_neighborhood(self, inp, query_indices):

        pairs_abq, inp_vals, mask = inp
        if query_indices is not None:
            B = torch.arange(inp_vals.shape[0], device=inp_vals.device).long()[:, None]
            abq_at_query = pairs_abq[B, query_indices]
            mask_at_query = mask[B, query_indices]
        else:
            abq_at_query = pairs_abq
            mask_at_query = mask
        vals_at_query = inp_vals
        dists = self.group.distance(abq_at_query, self.imageh)
        dists = torch.where(mask[:, None, :].expand(*dists.shape), dists, 1e8 * torch.ones_like(dists))
        k = min(self.mc_samples, inp_vals.shape[1])

        if self.knn:
            nbhd_idx = torch.topk(dists, k, dim=-1, largest=False, sorted=False)[1]
            valid_within_ball = (nbhd_idx > -1) & mask[:, None, :] & mask_at_query[:, :, None]
            assert not torch.any(nbhd_idx > dists.shape[-1])
        else:
            bs, m, n = dists.shape
            within_ball = (dists < self.r) & mask[:, None, :] & mask_at_query[:, :, None]
            B = torch.arange(bs)[:, None, None]
            M = torch.arange(m)[None, :, None]
            noise = torch.zeros(bs, m, n, device=within_ball.device)
            noise.uniform_(0, 1)
            valid_within_ball, nbhd_idx = torch.topk(within_ball + noise, k, dim=-1, largest=True, sorted=False)
            valid_within_ball = (valid_within_ball > 1)

        B = torch.arange(inp_vals.shape[0], device=inp_vals.device).long()[:, None, None].expand(*nbhd_idx.shape)
        M = torch.arange(abq_at_query.shape[1], device=inp_vals.device).long()[None, :, None].expand(*nbhd_idx.shape)
        nbhd_abq = abq_at_query[B, M, nbhd_idx]
        nbhd_vals = vals_at_query[B, nbhd_idx]
        nbhd_mask = mask[B, nbhd_idx]

        if self.training and not self.knn:
            navg = (within_ball.float()).sum(-1).sum() / mask_at_query[:, :, None].sum()
            avg_fill = (navg / mask.sum(-1).float().mean()).cpu().item()
            self.r += self.coeff * (self.fill_frac - avg_fill)
            self.fill_frac_ema += .1 * (avg_fill - self.fill_frac_ema)
        return nbhd_abq, nbhd_vals, (nbhd_mask & valid_within_ball.bool())

    def point_convolve(self, embedded_group_elems, nbhd_vals, nbhd_mask):
        bs, m, nbhd, ci = nbhd_vals.shape
        _, penult_kernel_weights, _ = self.weightnet((None, embedded_group_elems, nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1), penult_kernel_weights,
                                              torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1), nbhd_vals, torch.zeros_like(nbhd_vals))
        partial_convolved_vals = (nbhd_vals_m.transpose(-1, -2) @ penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1, keepdim=True).clamp(min=1)
        return convolved_vals

    def forward(self, inp):
        sub_abq, sub_vals, sub_mask, query_indices = self.subsample(inp, withquery=True)
        nbhd_abq, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_indices)
        convolved_vals = self.point_convolve(nbhd_abq, nbhd_vals, nbhd_mask)
        convolved_wzeros = torch.where(sub_mask.unsqueeze(-1), convolved_vals, torch.zeros_like(convolved_vals))
        return sub_abq, convolved_wzeros, sub_mask


class BottleBlock(nn.Module):
    def __init__(self, chin, chout, conv, bn=False, act='swish', fill=None):
        super().__init__()
        assert chin <= chout
        nonlinearity = Swish if act == 'swish' else nn.ReLU
        self.conv = conv(chin // 4, chout // 4, fill=fill) if fill is not None else conv(chin // 4, chout // 4)
        self.net = nn.Sequential(
            MaskBatchNormNd(chin) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            Pass(nn.Linear(chin, chin // 4), dim=1),
            MaskBatchNormNd(chin // 4) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            self.conv,
            MaskBatchNormNd(chout // 4) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            Pass(nn.Linear(chout // 4, chout), dim=1),
        )
        self.chin = chin

    def forward(self, inp):
        sub_coords, sub_values, mask = self.conv.subsample(inp)
        new_coords, new_values, mask = self.net(inp)
        new_values[..., :self.chin] += sub_values
        return new_coords, new_values, mask


class GlobalPool(nn.Module):
    def __init__(self, mean=False):
        super().__init__()
        self.mean = mean

    def forward(self, x):
        if len(x) == 2: return x[1].mean(1)
        coords, vals, mask = x
        summed = torch.where(mask.unsqueeze(-1), vals, torch.zeros_like(vals)).sum(1)
        if self.mean:
            summed /= mask.sum(-1).unsqueeze(-1)
        return summed


@export
class SIM2ResNet(nn.Module, metaclass=Named):

    def __init__(self, chin, imageh, ds_frac=1, num_outputs=1, k=1536, nbhd=np.inf,
                 act="swish", bn=True, num_layers=6, mean=True, per_point=True, pool=True,
                 liftsamples=1, fill=1 / 4, group=SIM2, knn=False, cache=False, **kwargs):
        super().__init__()
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(ki, ko, imageh=imageh, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=mean,
                                            group=group, fill=fill, cache=cache, knn=knn, **kwargs)
        self.net = nn.Sequential(
            Pass(nn.Linear(chin, k[0]), dim=1),
            *[BottleBlock(k[i], k[i + 1], conv, bn=bn, act=act, fill=fill[i]) for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish() if act == 'swish' else nn.ReLU(), dim=1),
            Pass(nn.Linear(k[-1], num_outputs), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]),
        )
        self.liftsamples = liftsamples
        self.per_point = per_point
        self.group = group

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        return self.net(lifted_x)


@export
class SIM2LieGroup(SIM2ResNet):
    def __init__(self, imageh=20, chin=1, total_ds=1 / 64, num_layers=6, group=SIM2, fill=1/32, k=128,
                 knn=False, bn=True, cache=False, nbhd=8, num_targets=100, act='swish', pool=True,
                 mean=True, increase_channels=True, **kwargs):
        ds_frac = (total_ds) ** (1 / num_layers)
        fill = [fill / ds_frac ** i for i in range(num_layers)]
        if increase_channels:
            k = [int(k / ds_frac ** (i / 2)) for i in range(num_layers + 1)]
        super().__init__(imageh=imageh, chin=chin, ds_frac=ds_frac, num_layers=num_layers, nbhd=nbhd, mean=True,
                         group=group, fill=fill, k=k, num_outputs=num_targets, cache=True, knn=knn, **kwargs)
        self.lifted_coords = None
        conv = lambda ki, ko, fill: LieConv(ki, ko, imageh=imageh, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=mean,
                                            group=group, fill=fill, cache=cache, knn=knn, **kwargs)
        self.net1 = nn.Sequential(
            MaskBatchNormNd(chin) if bn else nn.Sequential(),
            Pass(Swish() if act == 'swish' else nn.ReLU(), dim=1),
            Pass(nn.Linear(chin, k[0]), dim=1),
            *[BottleBlock(k[i], k[i + 1], conv, bn=bn, act=act, fill=fill[i]) for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish() if act == 'swish' else nn.ReLU(), dim=1),
            Pass(nn.Linear(k[-1], num_targets), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]),
        )

    def forward(self, x, coord_transform=None):
        bs, c, h, w = x.shape
        i = torch.linspace(-h / 2., h / 2., h)
        j = torch.linspace(-w / 2., w / 2., w)
        coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()
        center_mask = coords.norm(dim=-1) < 1.
        coords = coords[center_mask].view(-1, 2).unsqueeze(0).repeat(bs, 1, 1).to(x.device)
        if coord_transform is not None: coords = coord_transform(coords)
        values = x.permute(0, 2, 3, 1)[:, center_mask, :].reshape(bs, -1, c)
        mask = torch.ones(bs, values.shape[1], device=x.device) > 0
        z = (coords, values, mask)
        with torch.no_grad():
            if self.lifted_coords is None or self.lifted_coords.shape[0] != coords.shape[0]:
                self.lifted_coords, lifted_vals, lifted_mask = self.group.lift(z, self.liftsamples)
            else:
                lifted_vals, lifted_mask = self.group.expand_like(values, mask, self.lifted_coords)
        return self.net1((self.lifted_coords, lifted_vals, lifted_mask))