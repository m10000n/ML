import math

import numpy as np
import torch
import torch.nn.functional as F


def get_window(x: torch.Tensor, window_width: int, start_idx: int | None = None) -> torch.Tensor:
    assert x.ndim == 4, f"The number of dimensions of `x` must be 4."
    assert window_width > 0, f"`window_width` must be > 0."

    t_dim = x.shape[0]

    if start_idx is None:
        assert window_width <= t_dim, f"`window_width` must be <= the number of volumes."
        start_idx_ = np.random.randint(low=0, high=t_dim - window_width + 1)
    else:
        assert start_idx >= 0, f"`start_idx` must be >= 0."
        assert start_idx + window_width <= t_dim, f"`start_idx` + `window_width` must be <= number of volumes."
        start_idx_ = 0

    return x[start_idx_ : start_idx_ + window_width, ...]


def random_transform(
    x: torch.Tensor, angle_max_deg: float, trans_max_voxel: float, scale_max: float, chunk: int | None = None
) -> torch.Tensor:
    assert x.ndim == 4, f"The number of dimensions of `x` must be 4."
    assert 0 <= scale_max < 1, f"`scale_max` must be in [0, 1)."
    assert chunk is None or chunk > 0, f"The chunk size must be a positive integer or None."

    t_dim, x_dim, y_dim, z_dim = x.shape

    assert min(x_dim, y_dim, z_dim) > 1, "All spatial dimensions must be > 1."

    dtype_original = x.dtype
    x = x.to(torch.get_default_dtype()).contiguous()

    chunk_ = chunk if chunk is not None and chunk < t_dim else t_dim

    rx, ry, rz = torch.empty(3).uniform_(-angle_max_deg, angle_max_deg) * math.pi / 180
    tx, ty, tz = torch.empty(3).uniform_(-trans_max_voxel, trans_max_voxel)
    s = 1 + torch.empty(()).uniform_(-scale_max, scale_max)

    # rotation
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.eye(3)
    Rx[1, 1], Rx[1, 2], Rx[2, 1], Rx[2, 2] = cx, -sx, sx, cx

    Ry = torch.eye(3)
    Ry[0, 0], Ry[0, 2], Ry[2, 0], Ry[2, 2] = cy, sy, -sy, cy

    Rz = torch.eye(3)
    Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = cz, -sz, sz, cz

    R = torch.eye(4)
    R[:3, :3] = Rz @ Ry @ Rx

    # translation
    def T_translate(tx: float, ty: float, tz: float) -> torch.Tensor:
        M = torch.eye(4)
        M[0, 3], M[1, 3], M[2, 3] = tx, ty, tz
        return M

    cx0, cy0, cz0 = (x_dim - 1) / 2, (y_dim - 1) / 2, (z_dim - 1) / 2
    T_c = T_translate(cx0, cy0, cz0)
    T_neg_c = T_translate(-cx0, -cy0, -cz0)
    T_t = T_translate(tx, ty, tz)

    # scale
    S = torch.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = s

    # transformation
    T = T_t @ T_c @ S @ R @ T_neg_c  # move to origin => rotate => scale => move_back => translate
    T_inv = torch.linalg.inv(T)  # grid_sample expects output/input mapping

    ## affine_grid/grid_sample uses normalized coordinates and the grid axes correspond to Z, Y, X.
    M_vox_to_norm = torch.tensor(
        [[0.0, 0.0, 1.0 / cz0, -1.0], [0.0, 1.0 / cy0, 0.0, -1.0], [1.0 / cx0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    M_norm_to_vox = torch.tensor(
        [[0.0, 0.0, cx0, cx0], [0.0, cy0, 0.0, cy0], [cz0, 0.0, 0.0, cz0], [0.0, 0.0, 0.0, 1.0]]
    )

    A = (M_vox_to_norm @ T_inv @ M_norm_to_vox)[:3, :4].unsqueeze(0)  # (1, 3, 4)

    with torch.no_grad():
        out = torch.empty_like(x)
        grid = F.affine_grid(A, size=[1, 1, x_dim, y_dim, z_dim], align_corners=True).expand(chunk_, -1, -1, -1, -1)

        for i in range(0, t_dim, chunk_):
            j = min(i + chunk_, t_dim)
            n = j - i
            grid_ = grid if n == chunk_ else grid[0:n, ...]
            x_ = x[i:j].unsqueeze(1)
            warped = F.grid_sample(x_, grid_, mode="bilinear", padding_mode="border", align_corners=True)
            out[i:j].copy_(warped.squeeze(1))

    return out.to(dtype_original)
