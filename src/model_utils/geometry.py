import numpy as np
import torch


def decode_pca(coeff, mean_shape,  blend_shapes,  n_components=64, num_points=100):
    x = mean_shape + blend_pca(coeff[:, :n_components], blend_shapes[:n_components])

    x = torch.fft.irfft(torch.complex(x[..., :3], x[..., 3:]), n=num_points - 1, dim=-2, norm='ortho')

    return x
    

def project_pca(data: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Project data to the subspace spanned by bases.

    Args:
        data (torch.Tensor): Hair data of shape (batch_size, ...).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Projected parameters of shape (batch_size, num_blend_shapes).
    """
    return torch.einsum('bn,cn->bc', data.flatten(start_dim=1), basis.flatten(start_dim=1))


def blend_pca(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Blend parameters and the corresponding blend shapes.

    Args:
        coeff (torch.Tensor): Parameters (blend shape coefficients) of shape (batch_size, num_blend_shapes).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Blended results of shape (batch_size, ...).
    """
    return torch.einsum('bn,n...->b...', coeff, basis)



def compute_similarity_transform(A, B):
    """
    Computes similarity transform (sR, t) such that:
    B ≈ s * R @ A + t

    A: source (N x 3)
    B: target (N x 3)

    Returns:
        s: scale (float)
        R: rotation (3 x 3)
        t: translation (3,)
    """
    assert A.shape == B.shape
    N = A.shape[0]

    # Compute centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # Center the point clouds
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB / N

    # SVD of covariance
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_A = np.sum(AA ** 2) / N
    scale = np.sum(S) / var_A

    # Compute translation
    t = centroid_B - scale * R @ centroid_A

    return scale, R, t


def can2world_transform(can_mesh, s, R, t):
       
    world_mesh = s * (R @ can_mesh.T).T + t 
    return world_mesh
