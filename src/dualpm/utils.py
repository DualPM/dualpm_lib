"""
Copyright 2025 University of Oxford
Author: Ben Kaye
Licence: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn.functional as F
import einops


def perspective_matrix(
    tan_half_fov: torch.Tensor | float,
    aspect: float = 1.0,
    near: float = 1e-1,
    far: float = 30.0,
) -> torch.Tensor:
    """
    OpenGL perspective matrix with clip space coordinates [-1, 1]

    Args:
        tan_half_fov: tan of half the vertical field of view
        aspect: aspect ratio
        near: near clipping plane distance
        far: far clipping plane distance

    Returns:
        Perspective matrix of shape (B, 4, 4)

    IF tan_half_fov is scalar, B=1
    """
    if not isinstance(tan_half_fov, torch.Tensor):
        tan_half_fov = torch.tensor([tan_half_fov], dtype=torch.float32)
    elif tan_half_fov.dim() == 0:
        tan_half_fov = tan_half_fov[None]

    device = tan_half_fov.device
    batch_size = tan_half_fov.shape[0]

    clip_projection = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )[None].repeat(batch_size, 1, 1)

    clip_projection[:, 0, 0] = tan_half_fov / aspect
    clip_projection[:, 1, 1] = -tan_half_fov

    return clip_projection


def tan_half_fov(focal_length: float, sensor_height: float) -> float:
    #  construct tan of half the vertical field of view

    return focal_length / sensor_height / 2


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose the last two dimensions of a tensor.

    Convenience function for matrix transposition that works with arbitrary
    leading batch dimensions. Equivalent to x.transpose(-2, -1) but uses
    einops for clarity.

    Args:
        x: Input tensor with at least 2 dimensions.

    Returns:
        Tensor with last two dimensions transposed.

    Example:
        >>> x = torch.randn(2, 3, 4, 5)
        >>> y = transpose(x)
        >>> print(y.shape)  # torch.Size([2, 3, 5, 4])
    """
    return einops.rearrange(x, "... r c -> ... c r")


def apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a 3D affine transformation to points.

    Applies a 4x4 homogeneous transformation matrix to 3D points. The transformation
    should be in the form [R|t] where R is a 3x3 rotation matrix and t is a 3x1
    translation vector. The bottom row should be [0, 0, 0, 1].

    Args:
        points: 3D points to transform. Shape (..., N, 3) where ... represents
            arbitrary batch dimensions.
        transform: Transformation matrix. Shape (4, 4) for single transform or
            (..., 4, 4) for batched transforms matching points batch dimensions.

    Returns:
        Transformed points with same shape as input points.

    Raises:
        AssertionError: If points don't have 3 coordinates or transform bottom
            row is not [0, 0, 0, ?].

    Note:
        This function assumes the transformation matrix follows the convention
        where the bottom row is [0, 0, 0, 1], typical in computer graphics.

    Example:
        >>> points = torch.randn(100, 3)
        >>> # Create a rotation around Z-axis
        >>> angle = torch.pi / 4
        >>> transform = torch.eye(4)
        >>> transform[0, 0] = torch.cos(angle)
        >>> transform[0, 1] = -torch.sin(angle)
        >>> transform[1, 0] = torch.sin(angle)
        >>> transform[1, 1] = torch.cos(angle)
        >>> rotated_points = apply_transform(points, transform)
        >>> print(rotated_points.shape)  # torch.Size([100, 3])
    """
    assert points.shape[-1] == 3, "Points must have 3 coordinates"
    assert transform.dim() == 2 or transform.dim() == 3, "Transform must be 2D or 3D"

    # Validate that transform has correct structure [R|t; 0 0 0 1]
    if transform.dim() == 2:
        _test = transform[None]
    else:
        _test = transform

    assert all(
        t[3, :3].allclose(torch.zeros(3, device=transform.device)) for t in _test
    ), "Transform must be [R | t; 0 0 0 1] format"

    # Apply transformation: points * R.T + t
    transform = transpose(transform)
    return points @ transform[..., :3, :3] + transform[..., 3, :3]


def apply_projection(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply projection matrix to 3D points to get clip space coordinates.

    Transforms 3D points to 4D homogeneous clip space coordinates by applying
    a projection matrix. The points are augmented with a homogeneous coordinate
    of 1.0 before matrix multiplication.

    Args:
        points: 3D points in camera space. Shape (..., N, 3).
        matrix: Projection matrix. Shape (..., 4, 4) where batch dimensions
            should broadcast with points.

    Returns:
        Homogeneous clip space coordinates of shape (..., N, 4). The w-component
        contains the homogeneous coordinate for perspective division.
    """
    assert points.shape[-1] == 3, "Points must be in 3D"

    # Pad points to homogeneous coordinates (x, y, z, 1)
    homogeneous_points = F.pad(points, pad=(0, 1), mode="constant", value=1.0)

    # Apply projection: (points * P.T) where P is projection matrix
    return torch.matmul(
        homogeneous_points,
        einops.rearrange(matrix, "... i j -> ... j i"),
    )
