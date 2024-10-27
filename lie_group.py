import torch
import numpy as np


# @export
def norm(x, dim):
    return (x ** 2).sum(dim=dim).sqrt()


class LieGroup(object):
    rep_dim = NotImplemented
    lie_dim = NotImplemented
    q_dim = NotImplemented

    def __init__(self, alpha=.2):
        super().__init__()
        self.alpha = alpha

    def exp(self, a):
        raise NotImplementedError

    def log(self, u):
        raise NotImplementedError

    def lifted_elems(self, xyz, nsamples):
        raise NotImplementedError

    def inv(self, g):
        return self.exp(-self.log(g))

    def distance(self, abq_pairs):
        ab_dist = norm(abq_pairs[..., :self.lie_dim], dim=-1)
        qa = abq_pairs[..., self.lie_dim:self.lie_dim + self.q_dim]
        qb = abq_pairs[..., self.lie_dim + self.q_dim:self.lie_dim + 2 * self.q_dim]
        qa_qb_dist = norm(qa - qb, dim=-1)
        return ab_dist * self.alpha + (1 - self.alpha) * qa_qb_dist

    def lift(self, x, nsamples, **kwargs):
        p, v, m = x
        expanded_a, expanded_q = self.lifted_elems(p, nsamples, **kwargs)
        nsamples = expanded_a.shape[-2] // m.shape[-1]
        expanded_v = v[..., None, :].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples, 1))
        expanded_v = expanded_v.reshape(*expanded_a.shape[:-1], v.shape[-1])
        expanded_mask = m[..., None].repeat((1,) * len(v.shape[:-1]) + (nsamples,))
        expanded_mask = expanded_mask.reshape(*expanded_a.shape[:-1])
        paired_a = self.elems2pairs(expanded_a)
        if expanded_q is not None:
            q_in = expanded_q.unsqueeze(-2).expand(*paired_a.shape[:-1], 1)
            q_out = expanded_q.unsqueeze(-3).expand(*paired_a.shape[:-1], 1)
            embedded_locations = torch.cat([paired_a, q_in, q_out], dim=-1)
        else:
            embedded_locations = paired_a
        return (embedded_locations, expanded_v, expanded_mask)

    def expand_like(self, v, m, a):
        nsamples = a.shape[-2] // m.shape[-1]
        expanded_v = v[..., None, :].repeat(
            (1,) * len(v.shape[:-1]) + (nsamples, 1))
        expanded_v = expanded_v.reshape(*a.shape[:2], v.shape[-1])
        expanded_mask = m[..., None].repeat((1,) * len(v.shape[:-1]) + (nsamples,))
        expanded_mask = expanded_mask.reshape(*a.shape[:2])
        return expanded_v, expanded_mask

    def elems2pairs(self, a):
        vinv = self.exp(-a.unsqueeze(-3))
        u = self.exp(a.unsqueeze(-2))
        uv = (vinv @ u).unsqueeze(-3)
        return self.log(uv).squeeze(-2)

    def BCH(self, a, b, order=2):
        assert order <= 4
        B = self.bracket
        z = a + b
        if order == 1: return z
        ab = B(a, b)
        z += (1 / 2) * ab
        if order == 2: return z
        aab = B(a, ab)
        bba = B(b, -ab)
        z += (1 / 12) * (aab + bba)
        if order == 3: return z
        baab = B(b, aab)
        z += -(1 / 24) * baab
        return z

    def bracket(self, a, b):
        A = self.components2matrix(a)
        B = self.components2matrix(b)
        return self.matrix2components(A @ B - B @ A)

    def __str__(self):
        return f"{self.__class__}({self.alpha})" if self.alpha != .2 else f"{self.__class__}"

    def __repr__(self):
        return str(self)


# @export
def LieSubGroup(liegroup, generators):
    class subgroup(liegroup):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.orig_dim = self.lie_dim
            self.lie_dim = len(generators)
            self.q_dim = self.orig_dim - len(generators)

        def exp(self, a_small):
            a_full = torch.zeros(*a_small.shape[:-1], self.orig_dim,
                                 device=a_small.device, dtype=a_small.dtype)
            a_full[..., generators] = a_small
            return super().exp(a_full)

        def log(self, U):
            return super().log(U)[..., generators]

        def components2matrix(self, a_small):
            a_full = torch.zeros(*a_small.shape[:-1], self.orig_dim,
                                 device=a_small.device, dtype=a_small.dtype)
            a_full[..., generators] = a_small
            return super().components2matrix(a_full)

        def matrix2components(self, A):
            return super().matrix2components(A)[..., generators]

        def lifted_elems(self, pt, nsamples=1):
            a_full, q = super().lifted_elems(pt, nsamples)
            a_sub = a_full[..., generators]
            complement_generators = list(set(range(self.orig_dim)) - set(generators))
            new_qs = a_full[..., complement_generators]
            q_sub = torch.cat([q, new_qs], dim=-1) if q is not None else new_qs
            return a_sub, q_sub

    return subgroup


thresh = 7e-2


def sinc(x):
    x2 = x * x
    usetaylor = (x.abs() < thresh)
    return torch.where(usetaylor, 1 - x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42)), x.sin() / x)


def sincc(x):
    x2 = x * x
    usetaylor = (x.abs() < thresh)
    return torch.where(usetaylor, 1 / 6 * (1 - x2 / 20 * (1 - x2 / 42 * (1 - x2 / 72))), (x - x.sin()) / x ** 3)


def cosc(x):
    x2 = x * x
    usetaylor = (x.abs() < thresh)
    return torch.where(usetaylor, 1 / 2 * (1 - x2 / 12 * (1 - x2 / 30 * (1 - x2 / 56))), (1 - x.cos()) / x ** 2)


def coscc(x):
    x2 = x * x
    usetaylor = (x.abs() < thresh)
    texpand = 1 / 12 * (1 + x2 / 60 * (1 + x2 / 42 * (1 + x2 / 40)))
    costerm = (2 * (1 - x.cos())).clamp(min=1e-6)
    full = (1 - x * x.sin() / costerm) / x ** 2
    output = torch.where(usetaylor, texpand, full)
    return output


def sinc_inv(x):
    usetaylor = (x.abs() < thresh)
    texpand = 1 + (1 / 6) * x ** 2 + (7 / 360) * x ** 4
    assert not torch.any(torch.isinf(texpand) | torch.isnan(texpand)), 'sincinv texpand inf' + torch.any(
        torch.isinf(texpand))
    return torch.where(usetaylor, texpand, x / x.sin())


# @export
class SO2(LieGroup):
    lie_dim = 1
    rep_dim = 2
    q_dim = 1

    def exp(self, a):
        R = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        sin = a[..., 0].sin()
        cos = a[..., 0].cos()
        R[..., 0, 0] = cos
        R[..., 1, 1] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        return R

    def log(self, R):
        return torch.atan2(R[..., 1, 0] - R[..., 0, 1], R[..., 0, 0] + R[..., 1, 1])[..., None]

    def components2matrix(self, a):
        A = torch.zeros(*a.shape[:-1], 2, 2, device=a.device, dtype=a.dtype)
        A[..., 0, 1] = -a[..., 0]
        A[..., 1, 0] = a[..., 0]
        return A

    def matrix2components(self, A):
        a = torch.zeros(*A.shape[:-1], 1, device=A.device, dtype=A.dtype)
        a[..., :1] = (A[..., 1, :1] - A[..., :1, 1]) / 2
        return a

    def lifted_elems(self, pt, nsamples=1):
        assert nsamples == 1
        bs, n, D = pt.shape[:3]
        assert D == 2
        r = norm(pt, dim=-1).unsqueeze(-1)
        theta = torch.atan2(pt[..., 1], pt[..., 0]).unsqueeze(-1)
        return theta, r

    def distance(self, abq_pairs):
        angle_pairs = abq_pairs[..., 0]
        ra = abq_pairs[..., 1]
        rb = abq_pairs[..., 2]
        return angle_pairs.abs() * self.alpha + (1 - self.alpha) * (ra - rb).abs() / (ra + rb + 1e-3)


# @export
class SE2(SO2):
    lie_dim = 3
    rep_dim = 3
    q_dim = 0

    def log(self, g):
        theta = super().log(g[..., :2, :2])
        I = torch.eye(2, device=g.device, dtype=g.dtype)
        K = super().components2matrix(torch.ones_like(theta))
        theta = theta.unsqueeze(-1)
        Vinv = (sinc(theta) / (2 * cosc(theta))) * I - theta * K / 2
        a = torch.zeros(g.shape[:-1], device=g.device, dtype=g.dtype)
        a[..., 0] = theta[..., 0, 0]
        a[..., 1:] = (Vinv @ g[..., :2, 2].unsqueeze(-1)).squeeze(-1)
        return a

    def exp(self, a):
        theta = a[..., 0].unsqueeze(-1)
        I = torch.eye(2, device=a.device, dtype=a.dtype)
        K = super().components2matrix(torch.ones_like(a))
        theta = theta.unsqueeze(-1)
        V = sinc(theta) * I + theta * cosc(theta) * K
        g = torch.zeros(*a.shape[:-1], 3, 3, device=a.device, dtype=a.dtype)
        g[..., :2, :2] = theta.cos() * I + theta.sin() * K
        g[..., :2, 2] = (V @ a[..., 1:].unsqueeze(-1)).squeeze(-1)
        g[..., 2, 2] = 1
        return g

    def components2matrix(self, a):
        A = torch.zeros(*a.shape, 3, device=a.device, dtype=a.dtype)
        A[..., 2, :2] = a[..., 1:]
        A[..., 0, 1] = a[..., 0]
        A[..., 1, 0] = -a[..., 0]
        return A

    def matrix2components(self, A):
        a = torch.zeros(*A.shape[:-1], device=A.device, dtype=A.dtype)
        a[..., 1:] = A[..., :2, 2]
        a[..., 0] = (A[..., 1, 0] - A[..., 0, 1]) / 2
        return a

    def lifted_elems(self, pt, nsamples=1):
        d = self.rep_dim
        thetas = torch.linspace(-np.pi, np.pi, nsamples + 1)[1:].to(pt.device)
        for _ in pt.shape[:-1]:
            thetas = thetas.unsqueeze(0)
        thetas = thetas + torch.rand(*pt.shape[:-1], 1).to(pt.device) * 2 * np.pi
        R = torch.zeros(*pt.shape[:-1], nsamples, d, d).to(pt.device)
        sin, cos = thetas.sin(), thetas.cos()
        R[..., 0, 0] = cos
        R[..., 1, 1] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        R[..., 2, 2] = 1

        T = torch.zeros_like(R)
        T[..., 0, 0] = 1
        T[..., 1, 1] = 1
        T[..., 2, 2] = 1
        T[..., :2, 2] = pt.unsqueeze(-2)
        flat_a = self.log(T @ R).reshape(*pt.shape[:-2], pt.shape[-2] * nsamples, d)
        return flat_a, None

    def distance(self, abq_pairs):
        d_theta = abq_pairs[..., 0].abs()
        d_r = norm(abq_pairs[..., 1:], dim=-1)
        return d_theta * self.alpha + (1 - self.alpha) * d_r

class SIM2(SO2):

    lie_dim = 4
    rep_dim = 3
    q_dim = 0

    def component2matrix(self, k):
        J = torch.zeros(2, 2, device=k.device, dtype=k.dtype)
        J[..., 0, 1] = -k[..., 0]
        J[..., 1, 0] = k[..., 0]
        return J

    def log(self, g):
        device = g.device
        theta = torch.atan2(g[..., 1, 0] - g[..., 0, 1], g[..., 0, 0] + g[..., 1, 1])[..., None]
        I = torch.eye(2, device=g.device, dtype=g.dtype)
        K = torch.tensor([[0, -1],
                          [1, 0]])
        K = K.to(device)
        lamda = torch.log(g[..., 2, 2])[..., None]
        A = sinc(theta)
        B = cosc(theta)
        C = sincc(theta)
        alpha = lamda ** 2 / (lamda ** 2 + theta ** 2 + 1e-5)
        s = torch.exp(lamda)
        X = alpha * ((s - 1) / (lamda + 1e-5)) + (1 - alpha) * (A + lamda * B)
        Y = alpha * ((s  - 1 - lamda) / (lamda ** 2 + 1e-5)) + (1 - alpha) * (B + lamda * C)
        frac_xy = X ** 2 + (theta * Y) ** 2
        Vinv = (X/frac_xy)*I - ((theta*Y)/frac_xy)*K
        a = torch.zeros(*g.shape[:-2], 4, device=g.device, dtype=g.dtype)
        a[..., 0] = theta[..., 0, 0].unsqueeze(-1)
        a[..., 1:3] = (Vinv @ g[..., :2, 2].squeeze(-2).unsqueeze(-1)).squeeze(-1).unsqueeze(-2)
        a[..., 3] = lamda[..., 0, 0].unsqueeze(-1)
        return a

    def exp(self, a):
        device = a.device
        theta = a[..., 0].unsqueeze(-1)
        lamda = a[..., -1].unsqueeze(-1)
        I = torch.eye(2, device=a.device, dtype=a.dtype)
        K = torch.tensor([[0, -1],
                          [1, 0]])
        K = K.to(device)
        alpha = lamda ** 2 / (lamda ** 2 + theta ** 2 + 1e-5)
        A = sinc(theta)
        B = cosc(theta)
        C = sincc(theta)
        s = torch.exp(lamda)
        X = alpha * ((s - 1) / (lamda + 1e-5)) + (1 - alpha) * (A + lamda * B)
        X = X.unsqueeze(-1)
        Y = alpha * ((s  - 1 - lamda) / (lamda ** 2 + 1e-5)) + (1 - alpha) * (B + lamda * C)
        V = X * I + (theta * Y).unsqueeze(-1) * K

        g = torch.zeros(*a.shape[:-1], 3, 3, device=a.device, dtype=a.dtype)
        g[..., :2, :2] = (theta.cos()).unsqueeze(-1) * I + (theta.sin()).unsqueeze(-1) * K
        g[..., :2, 2] = (V @ a[..., 1:3].unsqueeze(-1)).squeeze(-1)
        g[..., 2, 2] = (s).squeeze(-1)
        return g

    def matrix2components(self, A):
        a = torch.zeros(*A.shape[:-2], 4, device=A.device, dtype=A.dtype)
        a[..., 1:] = A[..., :2, 2]
        a[..., 0] = (A[..., 1, 0] - A[..., 0, 1]) / 2
        a[..., -1] = -A[..., 2, 2]
        return a

    def lifted_elems(self, pt, nsamples=1):
        d = self.rep_dim
        bs, n, D = pt.shape
        thetas = torch.linspace(-np.pi, np.pi, nsamples + 1)[1:].to(pt.device)
        for _ in pt.shape[:-1]:
            thetas = thetas.unsqueeze(0)

        thetas = thetas + torch.rand(bs, n, 1).to(pt.device) * 2 * np.pi
        R = torch.zeros(*pt.shape[:-1], nsamples, d, d).to(pt.device)
        sin, cos = thetas.sin(), thetas.cos()
        R[..., 0, 0] = cos
        R[..., 1, 1] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        R[..., 2, 2] = 1
        T = torch.zeros_like(R)
        T[..., 0, 0] = 1
        T[..., 1, 1] = 1
        T[..., 2, 2] = 1
        T[..., :2, 2] = pt.unsqueeze(-2)
        r = norm(pt, dim=-1).unsqueeze(-1)
        T[..., 2, 2] = r
        flat_a = self.log(T @ R).reshape(*pt.shape[:-2], pt.shape[-2] * nsamples, 4)
        return flat_a, None

    def distance(self, abq_pairs, imageh):
        d_theta = abq_pairs[..., 0].abs()
        d_r = norm(abq_pairs[..., 1:3], dim=-1)
        d_rr = abq_pairs[..., 3].abs()
        return d_theta * self.alpha + (1 - self.alpha) * (d_r+(d_rr * imageh))