# Adaptation of the original code from 
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
# Modifications Copyright (c) 2021 Kazuki Irie

import torch
from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

mod_causal_dot_product_cuda = load(
    name="fast_weight_forward",
    sources=["utils/fast_fast_weight/fast_weight_cuda.cu"], verbose=True)
mod_causal_dot_backward_cuda = load(
    name="fast_weight_backward",
    sources=["utils/fast_fast_weight/fast_weight_cuda.cu"], verbose=True)


causal_dot_product_cuda = mod_causal_dot_product_cuda.fast_weight_forward
causal_dot_backward_cuda = mod_causal_dot_backward_cuda.fast_weight_backward


class FastWeightMemory(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V, beta, W):

        # Create the output tensor
        device = Q.device
        N, H, L, E = Q.shape
        _, _, _, M = V.shape

        product = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)
        # W = torch.zeros((N, H, E, M), device=device, dtype=Q.dtype)
        V_old = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)
        V_insert = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)

        # Actually perform the dot product
        FastWeightMemory.dot[device.type](
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old,
            V_insert,
            W,
            product
        )

        ctx.save_for_backward(Q, K, V, beta, V_old, V_insert, W)

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V, beta, V_old, V_insert, W = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_beta = torch.zeros_like(beta)

        # Compute the gradients
        FastWeightMemory.dot_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old.data,
            V_insert.data,
            grad_out,
            W.data,
            grad_Q,
            grad_K,
            grad_V,
            grad_beta
        )

        return grad_Q, grad_K, grad_V, grad_beta, None


# Alias the autograd functions to python style snake case naming
fast_weight_memory = FastWeightMemory.apply


if __name__ == '__main__':
    import torch
    # Tests pass if the relative difference compared with
    # the corresponding torch autograd computation
    # is smaller than a threshold.

    # Ideally should be tested with double...
    rel_threshold = 1e-3

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-6):
        return torch.abs((a-b) / (torch.abs(a) + eps)).max().item()

    print('##########################')
    print('# Test forward pass')
    print('##########################')

    bsz, n_head, slen, d_head = 3, 5, 7, 11
    v_dim = 4
    # (B, H, len, dim)
    q0 = torch.rand(3, 5, 7, 11).to(0)
    k0 = torch.rand(3, 5, 7, 11).to(0)
    v0 = torch.rand(3, 5, 7, 4).to(0)
    beta0 = torch.sigmoid(torch.rand(3, 5, 7, 1).to(0))

    q0 = q0 / q0.sum(dim=-1, keepdim=True)
    k0 = k0 / k0.sum(dim=-1, keepdim=True)

    q1 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    k1 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    v1 = torch.zeros(3, 5, 7, v_dim, requires_grad=True).to(0)
    beta1 = torch.zeros(3, 5, 7, 1, requires_grad=True).to(0)
    q1.data = q0.data
    k1.data = k0.data
    v1.data = v0.data
    beta1.data = beta0.data

    W1 = torch.zeros(3, 5, 11, v_dim).to(0)
    out1 = fast_weight_memory(q1, k1, v1, beta1, W1)

    # compute using torch
    q2 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    k2 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    v2 = torch.zeros(3, 5, 7, v_dim, requires_grad=True).to(0)
    beta2 = torch.zeros(3, 5, 7, 1, requires_grad=True).to(0)

    q2.data = q0.data
    k2.data = k0.data
    v2.data = v0.data
    beta2.data = beta0.data

    # (len, B, H, dim)
    q_2 = q2.permute(2, 0, 1, 3)
    slen, bsz, n_head, d_head = q_2.shape

    q_2 = q_2.reshape(slen, bsz * n_head, d_head)

    k_2 = k2.permute(2, 0, 1, 3)
    k_2 = k_2.reshape(slen, bsz * n_head, d_head)

    v_2 = v2.permute(2, 0, 1, 3)
    v_2 = v_2.reshape(slen, bsz * n_head, v_dim)

    beta_2 = beta2.permute(2, 0, 1, 3)
    beta_2 = beta_2.reshape(slen, bsz * n_head, 1)

    W = torch.zeros(3 * 5, v_dim, 11).to(0)

    out_list = []

    for pos in range(slen):
        v_old = torch.bmm(W, k_2[pos].unsqueeze(2)).squeeze()
        v_insert = beta_2[pos] * (v_2[pos] - v_old)
        W = W + torch.bmm(v_insert.unsqueeze(2), k_2[pos].unsqueeze(1))
        out_t = torch.bmm(W, q_2[pos].unsqueeze(2)).squeeze()
        out_list.append(out_t.clone())

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, v_dim)
 
    out1 = out1.permute(2, 0, 1, 3)

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"out1: {out1[s][b][h]}")
                print(f"out2: {out2[s][b][h]}")
                assert max_relative_error(
                    out1[s][b][h], out2[s][b][h]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad 
    loss1 = out1.sum()
    q1.retain_grad()
    k1.retain_grad()
    v1.retain_grad()
    beta1.retain_grad()

    loss1.backward()

    loss2 = out2.sum()
    q2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()
    beta2.retain_grad()

    loss2.backward()

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"s={s}, b={b}, h={h}")
                print(f"grad query1: {q1.grad[b][h][s]}")
                print(f"grad query2: {q2.grad[b][h][s]}")
                assert max_relative_error(
                    q1.grad[b][h][s], q2.grad[b][h][s]) < rel_threshold
                print("pass!")
   
                print(f"grad key1: {k1.grad[b][h][s]}")
                print(f"grad key2: {k2.grad[b][h][s]}")
                assert max_relative_error(
                    k1.grad[b][h][s], k2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad values1: {v1.grad[b][h][s]}")
                print(f"grad values2: {v2.grad[b][h][s]}")
                assert max_relative_error(
                    v1.grad[b][h][s], v2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad beta1: {beta1.grad[b][h][s]}")
                print(f"grad beta2: {beta2.grad[b][h][s]}")
                assert max_relative_error(
                    beta1.grad[b][h][s], beta2.grad[b][h][s]) < rel_threshold
                print("pass!")

    print("All tests pass.")

