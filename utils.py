import numpy as np
import torch
import torch.nn as nn
import sys
import random
import collections


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


export(export)


@export
class Named(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


@export
class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


@export
def conv2d(in_channels, out_channels, kernel_size=3, coords=False, dilation=1, **kwargs):
    assert 'padding' not in kwargs, "assumed to be padding = same "
    assert not coords, "coordconv not currently supported"
    same = (kernel_size // 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=same, dilation=dilation, **kwargs)


@export
class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)

def cosLr(num_epochs,cycle_mult=1):
    if isinstance(num_epochs, collections.abc.Iterable):
        num_epochs = sum(num_epochs)
    def lrSched(epoch):
        r = cycle_mult + 1e-8
        L = num_epochs
        current_cycle = np.floor(np.log(1+(r-1)*epoch/L)/np.log(r))
        current_cycle_length = L*r**current_cycle
        cycle_iter = epoch - L*(r**current_cycle - 1)/(r-1)
        cos_scale = max(.5*(1 + np.cos(np.pi*cycle_iter/current_cycle_length)),1e-3)
        return cos_scale
    return lrSched


@export
class RandomZrotation(nn.Module):
    def __init__(self, max_angle=np.pi):
        super().__init__()
        self.max_angle = max_angle

    def forward(self, x):
        if self.training:
            bs, c, n = x.shape;
            assert c == 3
            angles = (2 * torch.rand(bs) - 1) * self.max_angle
            R = torch.zeros(bs, 3, 3)
            R[:, 2, 2] = 1
            R[:, 0, 0] = R[:, 1, 1] = angles.cos()
            R[:, 0, 1] = R[:, 1, 0] = angles.sin()
            R[:, 1, 0] *= -1
            return R.to(x.device) @ x
        else:
            return x


@export
class GaussianNoise(nn.Module):

    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        else:
            return x


@export
class Pass(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs


def square_distance(src, dst):
    if src.shape[1] == 1 or dst.shape[1] == 1:
        return torch.sum((src - dst) ** 2, -1)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, ...]
    return new_points


def farthest_point_sample(xyz, npoint, distance=square_distance):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distances = torch.ones(B, N).to(device) * 1e8
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = distance(xyz, centroid)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, -1)[1]
    return centroids


def farthest_ball_point(radius, nsample, xyz, new_xyz, distance=square_distance):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nbhd, xyz, new_xyz, mask, distance=square_distance):
    sqrdists = distance(new_xyz.unsqueeze(-2), xyz.unsqueeze(-3))
    sqrdists[~mask[:, None, :].expand(*sqrdists.shape)] = 1e8
    _, group_idx = torch.topk(sqrdists, nbhd, dim=-1, largest=False, sorted=False)
    if torch.any(group_idx > 10000):
        print("greater than 10k :(")
        print(xyz.shape)
        print(new_xyz.shape)
        print(xyz[0])
        print(new_xyz[0])
        raise Exception
    return group_idx


def pthash(xyz):
    return hash(tuple(xyz.cpu().data.numpy().reshape(-1)) + (xyz.device,))


@export
class FarthestSubsample(nn.Module):
    def __init__(self, ds_frac=0.5, knn_channels=None, distance=square_distance, cache=False):
        super().__init__()
        self.ds_frac = ds_frac
        self.subsample_lookup = {}
        self.knn_channels = knn_channels
        self.distance = distance
        self.cache = cache

    def forward(self, x, coords_only=False):
        coords, values, mask = x
        if self.ds_frac == 1:
            if coords_only:
                return coords
            else:
                return x
        num_downsampled_points = int(np.round(coords.shape[1] * self.ds_frac))
        fps_idx = farthest_point_sample(coords[:, :, :self.knn_channels],
                                        num_downsampled_points, distance=self.distance)
        new_coords = index_points(coords, fps_idx)
        if coords_only: return new_coords
        new_values = index_points(values, fps_idx)
        new_mask = index_points(mask, fps_idx)
        return new_coords, new_values, new_mask