# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hadamard matrix module for distillation"""
import math
import os

import torch

from memoizer import memoize


def get_hadamard_matrix(feature_dim: int, allow_approx: bool = True):
    """
    Get or generate a Hadamard matrix of specified dimension.

    A Hadamard matrix is a square matrix whose entries are either +1 or -1 and whose rows
    are mutually orthogonal. This function uses caching to avoid recomputing matrices and
    supports various construction methods including Sylvester and Paley constructions.

    Args:
        feature_dim (int): The dimension of the square Hadamard matrix to generate.
        allow_approx (bool, optional): Whether to allow approximate matrices (Bernoulli)
            when exact Hadamard construction is not possible. Defaults to True.

    Returns:
        torch.Tensor: A normalized orthogonal matrix of shape (feature_dim, feature_dim)
            on CUDA device with dtype float64.

    Raises:
        ValueError: If feature_dim is invalid and allow_approx is False.
        AssertionError: If the constructed matrix fails orthogonality checks.
    """
    cache_dir = os.path.join(torch.hub.get_dir(), 'evfm', 'hadamard')
    cache_path = os.path.join(cache_dir, f'{feature_dim}.pth')

    if os.path.exists(cache_path):
        H = torch.load(cache_path, weights_only=True, map_location='cuda')
    else:
        H = _get_hadamard_matrix(feature_dim, allow_approx)

        n = H.norm(dim=1, keepdim=True)
        H /= n
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(H.cpu(), cache_path)

    assert H.shape[0] == feature_dim and H.shape[1] == feature_dim, "Invalid Hadamard matrix!"
    f = H @ H.T
    assert torch.allclose(f, torch.eye(feature_dim, dtype=H.dtype, device=H.device)), "Invalid orthogonal construction!"

    return H


def _get_hadamard_matrix(feature_dim: int, allow_approx: bool):
    """
    Internal function to construct a Hadamard matrix using various algorithms.

    This function attempts to construct an exact Hadamard matrix using different methods:
    1. Sylvester construction for powers of 2
    2. Paley constructions 1 and 2 for certain prime-related dimensions
    3. Combination of Sylvester and Paley constructions
    4. Bernoulli approximation if exact construction fails and allow_approx is True

    Args:
        feature_dim (int): The dimension of the square matrix to construct.
        allow_approx (bool): Whether to allow Bernoulli approximation when exact
            construction is not possible.

    Returns:
        torch.Tensor: A Hadamard matrix of shape (feature_dim, feature_dim).

    Raises:
        ValueError: If feature_dim is invalid or exact construction is not possible
            and allow_approx is False.
    """
    if feature_dim <= 0:
        raise ValueError("Invalid `feature_dim`. Must be a positive integer!")
    if feature_dim > 2 and feature_dim % 4 != 0:
        if allow_approx:
            return get_bernoulli_matrix(feature_dim)
        raise ValueError("`feature_dim` certainly needs to be divisible by 4, or be 1 or 2!")

    pw2 = math.log2(feature_dim)
    is_pow2 = int(pw2) == pw2

    if is_pow2:
        return get_sylvester_hadamard_matrix(feature_dim)

    # Not so simple anymore. We need to see if we can use Paley's construction
    prime_factors = _get_prime_factors(feature_dim)
    num_2s = sum(1 for v in prime_factors if v == 2)

    for i in range(num_2s - 1, -1, -1):
        syl_size = 2 ** i
        paley_size = 1
        for k in range(i, len(prime_factors)):
            paley_size *= prime_factors[k]

        # Paley Construction 1
        paley = None
        if _is_paley_construction_1(paley_size):
            paley = get_paley_hadamard_matrix_1(paley_size)
        elif _is_paley_construction_2(paley_size):
            paley = get_paley_hadamard_matrix_2(paley_size)

        if paley is not None:
            if syl_size > 1:
                syl = get_sylvester_hadamard_matrix(syl_size)
                return get_joint_hadamard_matrix(syl, paley)
            return paley

    if allow_approx:
        return get_bernoulli_matrix(feature_dim)
    raise ValueError("Unsupported `feature_dim`.")


def get_sylvester_hadamard_matrix(feature_dim: int):
    """
    Generate a Hadamard matrix using Sylvester's construction method.

    Sylvester's construction is a recursive method that generates Hadamard matrices
    for dimensions that are powers of 2. It starts with a 1x1 matrix [1] and
    recursively builds larger matrices by the Kronecker product with [[1,1],[1,-1]].

    Args:
        feature_dim (int): The dimension of the matrix. Must be a power of 2.

    Returns:
        torch.Tensor: A Hadamard matrix of shape (feature_dim, feature_dim)
            with dtype float64 on CUDA device.

    Raises:
        ValueError: If feature_dim is not a power of 2.
        AssertionError: If the construction algorithm fails (internal check).
    """
    pw2 = math.log2(feature_dim)
    is_pow2 = int(pw2) == pw2
    if not is_pow2:
        raise ValueError("The `feature_dim` must be a power of 2 for this algorithm!")

    A = torch.ones(1, 1, dtype=torch.float64, device='cuda')
    while A.shape[0] < feature_dim:
        B = A.repeat(2, 2)
        B[-A.shape[0]:, -A.shape[0]:] *= -1

        A = B

    assert A.shape[0] == feature_dim, "Invalid algorithm!"

    return A


def get_paley_hadamard_matrix_1(feature_dim: int):
    """
    Generate a Hadamard matrix using Paley Construction I.

    Paley Construction I generates Hadamard matrices of size q+1 where q is a prime
    power ≡ 3 (mod 4). The construction uses the quadratic character over the finite
    field of size q to determine the signs in the matrix.

    Args:
        feature_dim (int): The dimension of the matrix. Must be q+1 where q is a
            prime power congruent to 3 modulo 4.

    Returns:
        torch.Tensor: A Hadamard matrix of shape (feature_dim, feature_dim)
            with dtype float32 on CUDA device.
    """
    q = feature_dim - 1
    Q = _get_paley_q(q)

    H = torch.eye(feature_dim, dtype=torch.float32, device='cuda')
    H[0, 1:].fill_(1)
    H[1:, 0].fill_(-1)
    H[1:, 1:] += Q

    return H


def get_paley_hadamard_matrix_2(feature_dim: int):
    """
    Generate a Hadamard matrix using Paley Construction II.

    Paley Construction II generates Hadamard matrices of size 2(q+1) where q is a
    prime power ≡ 1 (mod 4). This construction creates a larger matrix by using
    2x2 blocks based on the quadratic character matrix from Paley Construction I.

    Args:
        feature_dim (int): The dimension of the matrix. Must be 2(q+1) where q is a
            prime power congruent to 1 modulo 4.

    Returns:
        torch.Tensor: A Hadamard matrix of shape (feature_dim, feature_dim)
            with dtype float32 on CUDA device.
    """
    q = feature_dim // 2 - 1
    Q = _get_paley_q(q)

    inner = torch.zeros(q + 1, q + 1, dtype=Q.dtype, device=Q.device)
    inner[0, 1:].fill_(1)
    inner[1:, 0].fill_(1)
    inner[1:, 1:].copy_(Q)

    zero_cells = torch.tensor([
        [1, -1],
        [-1, -1],
    ], dtype=inner.dtype, device=inner.device)

    pos_cells = torch.tensor([
        [1, 1],
        [1, -1],
    ], dtype=inner.dtype, device=inner.device)
    neg_cells = -pos_cells

    full_zero = zero_cells.repeat(*inner.shape)
    full_pos = pos_cells.repeat(*inner.shape)
    full_neg = neg_cells.repeat(*inner.shape)

    full_inner = inner.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    full_zero = torch.where(full_inner == 0, full_zero, 0)
    full_pos = torch.where(full_inner > 0, full_pos, 0)
    full_neg = torch.where(full_inner < 0, full_neg, 0)

    H = full_zero + full_pos + full_neg

    return H


def _get_paley_q(q: int):
    """
    Generate the quadratic character matrix Q for Paley constructions.

    This function computes a matrix Q where Q[i,j] represents the quadratic character
    of (i-j) modulo q. The quadratic character is +1 if the value is a quadratic
    residue, -1 if it's a non-residue, and 0 if it's zero modulo q.

    Args:
        q (int): A prime power used in Paley constructions.

    Returns:
        torch.Tensor: A matrix Q of shape (q, q) with entries in {-1, 0, 1}
            representing the quadratic character, with dtype float32 on CUDA device.
    """
    b_opts = torch.arange(1, q, dtype=torch.float32, device='cuda').pow_(2) % q

    indexer = torch.arange(1, q + 1, dtype=torch.float32, device='cuda')
    m_indexer = indexer[:, None] - indexer[None]
    m_indexer = m_indexer % q

    is_zero = m_indexer == 0
    is_square = torch.any(m_indexer[..., None] == b_opts[None, None], dim=2)

    sq_vals = is_square.float().mul_(2).sub_(1)

    Q = torch.where(is_zero, 0, sq_vals)
    return Q


def get_joint_hadamard_matrix(syl: torch.Tensor, paley: torch.Tensor):
    """
    Combine Sylvester and Paley Hadamard matrices using Kronecker product.

    This function creates a larger Hadamard matrix by taking the Kronecker product
    of a Sylvester construction matrix and a Paley construction matrix. This allows
    for constructing Hadamard matrices of composite dimensions.

    Args:
        syl (torch.Tensor): A Hadamard matrix from Sylvester construction.
        paley (torch.Tensor): A Hadamard matrix from Paley construction.

    Returns:
        torch.Tensor: A Hadamard matrix of shape (syl.shape[0] * paley.shape[0],
            syl.shape[1] * paley.shape[1]) created by Kronecker product.
    """
    ret = torch.kron(syl, paley)
    return ret


def get_bernoulli_matrix(feature_dim: int):
    """
    Generate an approximate Hadamard matrix using Bernoulli random variables.

    When exact Hadamard matrix construction is not possible for a given dimension,
    this function creates an approximate orthogonal matrix by generating a random
    matrix with Bernoulli entries (+1/-1) and then applying QR decomposition to
    orthogonalize it.

    Args:
        feature_dim (int): The dimension of the square matrix to generate.

    Returns:
        torch.Tensor: An approximately orthogonal matrix of shape (feature_dim, feature_dim)
            with dtype float64 on CUDA device. The matrix is normalized so that its
            diagonal elements are positive.
    """
    A = (torch.rand(feature_dim, feature_dim, dtype=torch.float32, device='cuda') > 0.5).double().mul_(2).sub_(1)

    Q, _ = torch.linalg.qr(A)

    Q = torch.where((Q.diag() > 0)[None], Q, -Q)

    Q = Q.T.contiguous()
    return Q


def _is_paley_construction(q: int, modulo: int):
    """
    Check if a Paley construction is applicable for given parameters.

    This function determines if a Paley construction can be used by checking if
    q is a prime and if any small power of q is congruent to the specified modulo.

    Args:
        q (int): The candidate prime for Paley construction.
        modulo (int): The required modulo value (1 for Paley II, 3 for Paley I).

    Returns:
        bool: True if Paley construction is applicable, False otherwise.
    """
    is_paley = False
    if _is_prime(q):
        for z in range(1, 11):
            qz = q ** z
            if qz % 4 == modulo:
                is_paley = True
                break
    return is_paley


def _is_paley_construction_1(feature_dim: int):
    """
    Check if Paley Construction I is applicable for the given dimension.

    Paley Construction I works when feature_dim = q+1 where q is a prime power
    congruent to 3 modulo 4.

    Args:
        feature_dim (int): The target matrix dimension.

    Returns:
        bool: True if Paley Construction I can be used, False otherwise.
    """
    q = feature_dim - 1
    return _is_paley_construction(q, modulo=3)


def _is_paley_construction_2(feature_dim: int):
    """
    Check if Paley Construction II is applicable for the given dimension.

    Paley Construction II works when feature_dim = 2(q+1) where q is a prime power
    congruent to 1 modulo 4.

    Args:
        feature_dim (int): The target matrix dimension.

    Returns:
        bool: True if Paley Construction II can be used, False otherwise.
    """
    q = feature_dim // 2 - 1
    return _is_paley_construction(q, modulo=1)


def _is_prime(n: int):
    """
    Check if a number is prime.

    This function determines primality by checking if the number has exactly
    one prime factor (itself).

    Args:
        n (int): The number to check for primality.

    Returns:
        bool: True if n is prime, False otherwise.
    """
    factors = _get_prime_factors(n)
    return len(factors) == 1


@memoize
def _get_prime_factors(n: int):
    """
    Get the prime factorization of a number.

    This function finds all prime factors of n using trial division.
    Results are memoized for efficiency.

    Args:
        n (int): The number to factorize.

    Returns:
        list: A list of prime factors of n (with repetition for powers).
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
