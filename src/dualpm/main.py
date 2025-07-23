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

from typing import Any

import nvdiffrast.torch as drt
import torch

import dualpm.structs as st
import dualpm.utils as ut


def render_dual_point_map(
    canonical_vertices: torch.Tensor | list[torch.Tensor],
    reconstruction_vertices: torch.Tensor | list[torch.Tensor],
    faces: torch.Tensor | list[torch.Tensor],
    model_view: torch.Tensor,
    projection: torch.Tensor,
    subtract_depth: bool,
    resolution: tuple[int, int],
    num_layers: int,
    return_on_cpu: bool,
    context: drt.RasterizeGLContext | drt.RasterizeCudaContext | None = None,
) -> torch.Tensor:
    """Render a dual point map from dual mesh geometry.

    This function implements the core dual point map rendering pipeline described in
    arXiv:2412.04464. It takes canonical and reconstruction vertices, applies transforms,
    and renders them into a multi-layer representation containing both spatial locations.

    Args:
        canonical_vertices: Canonical vertex positions. Either a tensor of shape
            (B, N, 3) or list of tensors each of shape (N_i, 3) for variable mesh sizes.
            Single mesh input (N, 3) will be auto-batched.
        reconstruction_vertices: Reconstruction vertex positions. Must match the format
            and batch size of canonical_vertices.
        faces: Triangle face indices. Either tensor of shape (B, F, 3) or list of
            tensors each of shape (F_i, 3). Must be consistent with vertex format.
        model_view: Model-view transformation matrix of shape (B, 4, 4) or (4, 4) for
            single batch. Transforms reconstruction vertices to camera space.
        projection: Projection matrix of shape (B, 4, 4) or (4, 4) for single batch.
            Projects camera space to clip space.
        subtract_depth: Whether to subtract depth offset from model-view transformed
            vertices. Used for depth-relative encoding.
        resolution: Output image resolution as (height, width).
        num_layers: Number of depth layers to render. More layers capture more geometry
            but increase computational cost.
        return_on_cpu: Whether to move final result to CPU memory.
        context: Optional nvdiffrast rendering context. If None, creates GL context.

    Returns:
        Dual point map tensor of shape (B, H, W, num_layers, 7) where:
            - [..., :3]: Canonical vertex positions
            - [..., 3:6]: Reconstruction vertex positions (camera space)
            - [..., 6]: Alpha channel (coverage/opacity)

    Note:
        Input tensors must have consistent batch dimensions. The function automatically
        handles batching for single-mesh inputs but requires consistent formats for
        multi-mesh inputs.

    Example:
        >>> canonical = torch.randn(1, 1000, 3)
        >>> reconstruction = torch.randn(1, 1000, 3)
        >>> faces = torch.randint(0, 1000, (1, 1800, 3))
        >>> model_view = torch.eye(4).unsqueeze(0)
        >>> projection = get_projection(torch.tensor([60.0]))
        >>> dpm = render_dual_point_map(
        ...     canonical, reconstruction, faces, model_view, projection,
        ...     subtract_depth=True, resolution=(512, 512), num_layers=4,
        ...     return_on_cpu=False
        ... )
        >>> print(dpm.shape)  # torch.Size([1, 512, 512, 4, 7])
    """
    # Determine input format: tensor mode vs list mode for variable mesh sizes
    list_mode: bool = not isinstance(canonical_vertices, torch.Tensor)

    # Auto-batch single mesh inputs by adding batch dimension
    if not list_mode and canonical_vertices.dim() == 2:
        canonical_vertices = canonical_vertices[None]  # (N, 3) -> (1, N, 3)
        reconstruction_vertices = reconstruction_vertices[None]
        faces = faces[None]

    # Transform vertices to clip space and prepare attributes for rasterization
    list_attributes, list_clip_verts = dual_mesh_to_attributes(
        canonical_vertices,
        reconstruction_vertices,
        model_view,
        projection,
        subtract_depth,
    )

    # Convert to batched format suitable for nvdiffrast
    inputs = st.RenderInput.from_list(
        list_clip_verts,
        list_attributes,
        faces,
    )

    # Perform multi-layer rasterization to generate depth-peeled representations
    canon_pose_map, alpha_map = rasterize(
        inputs.clip_vertices_pos,
        inputs.faces,
        inputs.vertex_attributes,
        inputs.ranges,
        resolution,
        num_layers,
        return_on_cpu,
        context,
    )

    # Concatenate canonical+reconstruction attributes with alpha channel
    return torch.cat([canon_pose_map, alpha_map], dim=-1)


def dual_mesh_to_attributes(
    canonical_vertices: torch.Tensor | list[torch.Tensor],
    reconstruction_vertices: torch.Tensor | list[torch.Tensor],
    model_view: torch.Tensor,
    projection: torch.Tensor,
    subtract_depth: bool,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert dual mesh vertices to rendering attributes and clip space coordinates.

    This function performs the coordinate space transformations required for dual point
    map rendering. It applies model-view and projection transforms to reconstruction
    vertices while keeping canonical vertices unchanged, then concatenates them as
    vertex attributes for rasterization.

    Processing steps:
    1. Apply model-view transform to reconstruction vertices
    2. Apply projection transform to get clip space coordinates
    3. Optionally subtract depth offset for relative depth encoding
    4. Concatenate canonical and transformed reconstruction vertices as attributes

    Args:
        canonical_vertices: Canonical vertex positions in object space. Format should
            match reconstruction_vertices (tensor or list of tensors).
        reconstruction_vertices: Reconstruction vertex positions in object space.
            Will be transformed to camera space via model_view.
        model_view: Model-view transformation matrix. Shape (4, 4) for single batch
            or (B, 4, 4) for multiple batches.
        projection: Projection matrix for camera parameters. Shape (4, 4) or (B, 4, 4).
        subtract_depth: Whether to subtract camera depth from transformed vertices.
            When True, subtracts model_view[2, 3] (or model_view[:, 2, 3] for batched).

    Returns:
        If input is tensor mode:
            - attributes: Combined vertex attributes of shape (B, N, 6) containing
                         canonical positions and camera-space reconstruction positions
            - clip_verts: Clip space vertex positions of shape (B, N, 4)

        If input is list mode:
            - attributes: List of attribute tensors, each of shape (N_i, 6)
            - clip_verts: List of clip space tensors, each of shape (N_i, 4)

    Note:
        The subtract_depth option is useful for encoding relative depth information
        rather than absolute camera-space depth, which can improve numerical stability
        and representation quality in certain applications.
    """
    # Determine processing mode based on input type
    list_mode: bool = not isinstance(canonical_vertices, torch.Tensor)

    if list_mode:
        # Process each mesh in the list individually
        # Apply model-view transform to convert reconstruction vertices to camera space
        model_view_verts = [
            ut.apply_transform(v, mv)
            for v, mv in zip(reconstruction_vertices, model_view, strict=True)
        ]

        # Apply projection to get clip space coordinates for rasterization
        clip_verts = [
            ut.apply_projection(v, p)
            for v, p in zip(model_view_verts, projection, strict=True)
        ]

        # Optionally subtract depth offset for relative depth encoding
        if subtract_depth:
            # Extract z-translation from model-view matrix (camera depth)
            model_view_verts = [
                v
                - torch.tensor([0, 0, mv[2, 3]], device=v.device)[
                    None, :
                ]  # Subtract only z-component
                for v, mv in zip(model_view_verts, model_view, strict=True)
            ]

        # Concatenate canonical and transformed reconstruction vertices as attributes
        attributes = [
            torch.cat([canon_verts, reconstruction_verts], dim=-1)
            for canon_verts, reconstruction_verts in zip(
                canonical_vertices, model_view_verts, strict=True
            )
        ]
        return attributes, clip_verts

    model_view_verts = ut.apply_transform(reconstruction_vertices, model_view)
    clip_verts = ut.apply_projection(model_view_verts, projection)

    if subtract_depth:
        # Subtract depth offset for relative encoding
        # model_view[2, 3] contains the z-translation (camera depth)
        model_view_verts = (
            model_view_verts - torch.tensor([0, 0, model_view[2, 3]])[None, None, :]
        )
    attributes = torch.cat([canonical_vertices, model_view_verts], dim=-1)

    return attributes, clip_verts


def rasterize(
    vertex_clip_pos: torch.Tensor,
    faces: torch.Tensor,
    attributes: torch.Tensor,
    ranges: torch.Tensor | None,
    resolution: tuple[int, int],
    num_layers: int,
    return_on_cpu: bool,
    context: drt.RasterizeGLContext | drt.RasterizeCudaContext | None = None,
    device: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform multi-layer rasterization with depth peeling to generate dual point maps.

    This function uses nvdiffrast's depth peeling functionality to render multiple depth
    layers of a mesh, interpolating vertex attributes at each layer. This creates a
    multi-layer representation that can capture complex geometry and occlusions.

    The rasterization process:
    1. Initialize depth peeler with vertex positions and faces
    2. For each layer, rasterize the next depth layer
    3. Interpolate vertex attributes using barycentric coordinates
    4. Generate alpha channel for coverage information
    5. Stack all layers into final multi-layer representation

    Args:
        vertex_clip_pos: Vertex positions in clip space coordinates of shape (B, N, 4).
            These are used for depth testing and triangle setup.
        faces: Triangle face indices of shape (B, F, 3) or (F, 3). Defines mesh topology.
        attributes: Per-vertex attributes to interpolate of shape (B, N, C). Can contain
            any vertex data (positions, colors, normals, etc.).
        ranges: Optional batch ranges for variable-size meshes. If None, assumes
            uniform batch processing.
        resolution: Output image resolution as (height, width) tuple.
        num_layers: Number of depth layers to render. Each layer captures geometry
            at increasing depth from the camera.
        return_on_cpu: Whether to transfer final results to CPU memory.
        context: Optional nvdiffrast rendering context. If None, creates new GL context.
        device: Device for context creation. Only used if context is None.

    Returns:
        tuple containing:
            - rasterized_attributes: Interpolated vertex attributes of shape
              (B, H, W, num_layers, C). Contains per-pixel attribute values.
            - rasterized_alpha: Alpha/coverage values of shape (B, H, W, num_layers, 1).
              Indicates pixel coverage by mesh geometry.

    Note:
        The depth peeling approach allows rendering of complex scenes with multiple
        overlapping surfaces, which is essential for accurate dual point map generation.
        Each layer represents a depth slice of the scene geometry.

    Raises:
        AssertionError: If vertex_clip_pos and attributes don't have matching vertex counts.

    returns
    rasterized_attributes: (B, H, W, N, C)
    rasterized_alpha: (B, H, W, N, 1)
    """

    assert (
        ranges is None
        and vertex_clip_pos.shape[:2] == attributes.shape[:2]
        or ranges is not None
        and vertex_clip_pos.shape[0] == attributes.shape[0]
    ), "must be one attribute per vertex!"
    assert faces.dtype == torch.int32, "faces must be int32"
    assert ranges is None or ranges.dtype == torch.int32, "ranges must be int32"

    if context is None:
        context = drt.RasterizeGLContext(output_db=False, device=device)

    rasterized_attributes = []
    rasterized_alpha = []
    with drt.DepthPeeler(
        context, vertex_clip_pos, faces, resolution, ranges, grad_db=False
    ) as peeler:
        for _i in range(num_layers):
            rast, *_ = peeler.rasterize_next_layer()

            # Rasterize the attributes, and rasterize alpha
            layer_attributes, *_ = drt.interpolate(
                attributes,
                rast,
                faces,
            )
            layer_alpha, *_ = drt.interpolate(
                torch.ones_like(attributes[..., [0]]), rast, faces
            )
            rasterized_alpha.append(layer_alpha)
            rasterized_attributes.append(layer_attributes)

    # Stack, with shape (B, H, W, N, C) and (B, H, W, N, 1)
    rasterized_attributes, rasterized_alpha = (
        torch.stack(tensor_list, dim=-2)
        for tensor_list in (rasterized_attributes, rasterized_alpha)
    )

    if return_on_cpu:
        rasterized_attributes, rasterized_alpha = (
            tensor.cpu() for tensor in (rasterized_attributes, rasterized_alpha)
        )
    return rasterized_attributes, rasterized_alpha
