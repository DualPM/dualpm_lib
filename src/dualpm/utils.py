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


def perspective_transform(
    fov_y: torch.Tensor, aspect: float = 1.0, near: float = 0.1, far: float = 1000.0
) -> torch.Tensor:
    """Create a perspective projection matrix from field of view parameters.

    Constructs a standard perspective projection matrix suitable for 3D rendering
    pipelines. The matrix transforms from camera space to normalized device coordinates.

    Args:
        fov_y: Vertical field of view in radians. Shape (B,) for batch processing.
        aspect: Aspect ratio (width/height). Defaults to 1.0 for square aspect.
        near: Near clipping plane distance. Must be positive.
        far: Far clipping plane distance. Must be greater than near.

    Returns:
        Perspective projection matrix of shape (B, 4, 4) where B matches fov_y batch size.
        The matrix follows OpenGL conventions with NDC range [-1, 1].

    Note:
        Uses standard perspective projection formulation. The resulting matrix
        maps camera space coordinates to clip space coordinates that can be used
        with graphics rasterization pipelines.

    Example:
        >>> fov = torch.tensor([60.0]) * torch.pi / 180  # Convert degrees to radians
        >>> proj = perspective_transform(fov, aspect=16/9, near=0.1, far=100.0)
        >>> print(proj.shape)  # torch.Size([1, 4, 4])
    """
    device = fov_y.device
    y = torch.tan(fov_y / 2)
    batch_size = fov_y.shape[0]
    perspective_matrix = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )[None, ...].repeat(batch_size, 1, 1)

    perspective_matrix[:, 0, 0] = 1 / (y * aspect)
    perspective_matrix[:, 1, 1] = 1 / -y

    return perspective_matrix


def perspective_from_focal_length(
    normalised_focal_length: torch.Tensor,
    aspect: float = 1.0,
    near: float = 1e-1,
    far: float = 30.0,
) -> torch.Tensor:
    """Create a perspective projection matrix from normalized focal length.

    Constructs a perspective projection matrix using focal length parameterization
    instead of field of view. The focal length should be normalized by image dimensions.

    Args:
        normalised_focal_length: Focal length normalized by image height. Can be a
            scalar, 0-d tensor, or 1-d tensor for batch processing.
        aspect: Aspect ratio (width/height). Defaults to 1.0.
        near: Near clipping plane distance. Must be positive.
        far: Far clipping plane distance. Must be greater than near.

    Returns:
        Perspective projection matrix of shape (B, 4, 4) where B is the batch size
        of normalised_focal_length.

    Note:
        The normalized focal length should be computed as focal_length_pixels / image_height.
        This parameterization is common in computer vision applications where camera
        intrinsics are specified in pixel units.

    Example:
        >>> # For a camera with 800px focal length and 600px image height
        >>> norm_focal = torch.tensor([800.0 / 600.0])
        >>> proj = perspective_from_focal_length(norm_focal, aspect=4/3)
        >>> print(proj.shape)  # torch.Size([1, 4, 4])
    """
    if not isinstance(normalised_focal_length, torch.Tensor):
        normalised_focal_length = torch.tensor(
            [normalised_focal_length], dtype=torch.float32
        )
    elif normalised_focal_length.dim() == 0:
        normalised_focal_length = normalised_focal_length[None]

    device = normalised_focal_length.device
    batch_size = normalised_focal_length.shape[0]

    scale = 2 * normalised_focal_length
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

    clip_projection[:, 0, 0] = scale / aspect
    clip_projection[:, 1, 1] = -scale

    return clip_projection


def get_projection(
    fov: float | torch.Tensor, znear: float = 1e-1, zfar: float = 1000.0
) -> torch.Tensor:
    """Create a perspective projection matrix from field of view in degrees.

    Convenience function that creates a perspective projection matrix from field of view
    specified in degrees. Internally converts to radians and calls perspective_transform.

    Args:
        fov: Vertical field of view in degrees. Can be float or tensor.
        znear: Near clipping plane distance. Must be positive.
        zfar: Far clipping plane distance. Must be greater than znear.

    Returns:
        Perspective projection matrix of shape (B, 4, 4) where B is 1 for scalar input
        or matches the batch size of fov tensor.

    Example:
        >>> proj = get_projection(60.0, znear=0.1, zfar=100.0)  # 60 degree FOV
        >>> print(proj.shape)  # torch.Size([1, 4, 4])
    """
    if not isinstance(fov, torch.Tensor):
        fov = torch.tensor([fov], dtype=torch.float32)

    # Convert degrees to radians
    fov_rad = fov * torch.pi / 180.0
    projection = perspective_transform(fov_rad, 1.0, znear, zfar)

    return projection


def get_fov(focal_length: float | torch.Tensor, frame_height: float) -> torch.Tensor:
    """Calculate field of view from focal length and sensor size.

    Computes the vertical field of view angle from focal length and sensor height.
    Uses the standard pinhole camera model relationship.

    Args:
        focal_length: Focal length in the same units as frame_height.
        frame_height: Sensor/frame height in the same units as focal_length.
            Defaults to 36.0mm (full-frame sensor height).

    Returns:
        Vertical field of view in radians as a tensor.

    Note:
        This relationship is: FOV = 2 * arctan(frame_height / (2 * focal_length))
        Common sensor sizes: 36mm (full-frame), 24mm (APS-C), 13.2mm (4/3).

    Example:
        >>> fov_rad = get_fov(50.0, frame_height=36.0)  # 50mm lens on full-frame
        >>> fov_deg = fov_rad * 180 / torch.pi
        >>> print(f"FOV: {fov_deg.item():.1f} degrees")
    """
    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.tensor([focal_length], dtype=torch.float32)
    fov = 2.0 * torch.arctan(frame_height / (2.0 * focal_length))
    return fov


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

    Note:
        After this transformation, perspective division (dividing x,y,z by w)
        yields normalized device coordinates (NDC). This is typically handled
        automatically by graphics hardware or rasterization libraries.

    Example:
        >>> points = torch.randn(100, 3)  # Camera space points
        >>> proj_matrix = get_projection(60.0)  # 60 degree FOV
        >>> clip_coords = apply_projection(points, proj_matrix)
        >>> print(clip_coords.shape)  # torch.Size([1, 100, 4])
        >>> # To get NDC coordinates:
        >>> ndc = clip_coords[..., :3] / clip_coords[..., [3]]
    """
    # Pad points to homogeneous coordinates (x, y, z, 1)
    homogeneous_points = F.pad(points, pad=(0, 1), mode="constant", value=1.0)

    # Apply projection: (points * P.T) where P is projection matrix
    return torch.matmul(
        homogeneous_points,
        einops.rearrange(matrix, "... i j -> ... j i"),
    )
