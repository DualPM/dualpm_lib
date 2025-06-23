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

import dataclasses
import torch


@dataclasses.dataclass
class DualMesh:
    """Container for dual mesh geometry with canonical and reconstruction vertices.

    A dual mesh represents geometry in two coordinate spaces: canonical (reference)
    and reconstruction (deformed/transformed). This is fundamental to dual point map
    representation as described in the paper.

    Attributes:
        reconstruction: Vertex positions in reconstruction space. Shape (N, 3).
        canonical: Vertex positions in canonical/reference space. Shape (N, 3).
        faces: Triangle face indices. Shape (F, 3) with values indexing into vertices.

    Note:
        Both vertex arrays must have the same number of vertices N, and faces must
        contain valid indices into the vertex arrays.
    """

    reconstruction: torch.Tensor
    canonical: torch.Tensor
    faces: torch.Tensor


class DualPointMap:
    """Dual point map representation with lazy extraction of components.

    This class provides a convenient interface for working with dual point map tensors,
    which encode both canonical and reconstruction coordinates at each pixel location
    across multiple depth layers.

    The underlying tensor format is (B, H, W, N, C) where:
    - B: Batch size
    - H, W: Image height and width
    - N: Number of depth layers
    - C: Channels (7 or 8) = [canonical_xyz, reconstruction_xyz, alpha, confidence?]

    Attributes:
        has_confidence: Whether the dual point map includes confidence values (8 channels vs 7).
        layers: Number of depth layers in the representation.
    """

    def __init__(self, dual_point_map: torch.Tensor) -> None:
        """Initialize dual point map from tensor.

        Args:
            dual_point_map: Dual point map tensor of shape (B, H, W, N, C) where
                C is 7 (without confidence) or 8 (with confidence).
        """
        self._tensor = dual_point_map
        self.layers = dual_point_map.shape[-2]

        # Lazy extraction attributes
        self._canonical: torch.Tensor | None = None
        self._reconstruction: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._confidence: torch.Tensor | None = None
        self.has_confidence: bool = False

    def _extract(self) -> None:
        """Extract components from the dual point map tensor.

        Lazily parses the tensor into separate components for canonical positions,
        reconstruction positions, alpha, and optional confidence values.

        Raises:
            ValueError: If dual point map tensor is not initialized.
            AssertionError: If tensor doesn't have expected 5D shape.
        """
        if self._tensor is None:
            raise ValueError("DualPointMap must be initialized with a tensor")

        assert self._tensor.dim() == 5, "Tensor must be (B, H, W, N, C=7|8)"

        self._canonical = self._tensor[..., :3]
        self._reconstruction = self._tensor[..., 3:6]
        self._alpha = self._tensor[..., 6]

        if self._tensor.shape[-1] == 8:
            self._confidence = self._tensor[..., 7]
            self.has_confidence = True

    def __iter__(self):
        """Iterate over dual point map components.

        Yields:
            canonical: Canonical positions tensor (B, H, W, N, 3)
            reconstruction: Reconstruction positions tensor (B, H, W, N, 3)
            alpha: Alpha/coverage values tensor (B, H, W, N)
            confidence: Confidence values tensor (B, H, W, N) or None
        """
        if self._canonical is None:
            self._extract()
        for component in [
            self._canonical,
            self._reconstruction,
            self._alpha,
            self._confidence,
        ]:
            yield component

    def as_list(
        self, confidence_threshold: float | None = None, alpha_threshold: float = 0.5
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor] | None,
    ]:
        """Extract valid points as lists filtered by thresholds.

        Converts the dual point map into lists of point coordinates by applying
        alpha and confidence thresholds to filter out invalid or low-quality points.

        Args:
            confidence_threshold: Minimum confidence value to include points. If None,
                confidence filtering is disabled even if confidence values exist.
            alpha_threshold: Minimum alpha value to include points. Points with
                alpha below this threshold are considered background.

        Returns:
            Tuple of (canonical_lists, reconstruction_lists, alpha_lists, confidence_lists)
            where each is a list of tensors (one per batch). confidence_lists is None
            if confidence values are not available.

        Note:
            The returned lists have one tensor per batch element, and each tensor
            contains only the points that pass the filtering criteria.

        Example:
            >>> dpm = DualPointMap(dual_point_map_tensor)
            >>> canon, recon, alpha, conf = dpm.as_list(
            ...     confidence_threshold=0.8, alpha_threshold=0.5
            ... )
            >>> print(f"Batch 0 has {canon[0].shape[0]} valid points")
        """
        canon, recon, alpha, conf = self

        # Create mask based on alpha threshold
        mask = (alpha >= alpha_threshold)[..., 0]  # Remove layer dimension for indexing

        # Add confidence threshold if available and requested
        if conf is not None and confidence_threshold is not None:
            mask = mask & (conf >= confidence_threshold)[..., 0]

        # Extract points that pass the thresholds for each batch
        canon_list = [c[m] for c, m in zip(canon, mask)]
        recon_list = [r[m] for r, m in zip(recon, mask)]
        alpha_list = [a[m] for a, m in zip(alpha, mask)]
        conf_list = [c[m] for c, m in zip(conf, mask)] if conf is not None else None

        return canon_list, recon_list, alpha_list, conf_list


@dataclasses.dataclass
class RenderInput:
    """Structured input for rasterization with support for batched and instanced rendering.

    This class encapsulates all the data needed for rasterization and provides utilities
    for converting between different input formats (lists vs batched tensors) and
    determining the optimal rendering strategy.

    Attributes:
        clip_vertices_pos: Vertex positions in clip space. Shape (B, N, 4) for batched
            or (N, 4) for single mesh.
        faces: Triangle face indices. Shape (F, 3)
        vertex_attributes: Per-vertex attributes to interpolate. Shape matches
            clip_vertices_pos leading dimensions.
        ranges: Optional ranges for concatenated geometry. Shape (B, 2) with
            [start_index, length] pairs. None for batched/instanced modes.
    """

    clip_vertices_pos: torch.Tensor
    faces: torch.Tensor
    vertex_attributes: torch.Tensor
    ranges: torch.Tensor | None

    def __post_init__(self) -> None:
        """Validate input data consistency.

        Raises:
            AssertionError: If input dimensions are inconsistent or invalid.
        """
        assert self.clip_vertices_pos.dim() in [2, 3], (
            "clip_vertices_pos must be (B, num_vertices, 4) or (num_vertices, 4)"
        )

        assert self.ranges is None or (
            self.ranges is not None and self.clip_vertices_pos.dim() == 2
        ), "clip_vertices_pos must be concatenated (2D) if ranges is provided"

        range_mode = self.ranges is not None

        assert (
            not range_mode
            and self.clip_vertices_pos.shape[:2] == self.vertex_attributes.shape[:2]
        ) or (
            range_mode
            and self.clip_vertices_pos.shape[0] == self.vertex_attributes.shape[0]
        ), (
            "clip_vertices_pos and vertex_attributes must have matching leading dimensions"
        )

        assert self.faces.dim() == 2 and self.faces.shape[-1] == 3, (
            "faces must represent triangles with shape (..., 3)"
        )

    @staticmethod
    def is_instance_mode(faces: list[torch.Tensor]) -> bool:
        """
        Determine if input data can use instanced rendering.

        This requires the geometry to have identical faces, and different vertex positions.
        """
        # Check if all face arrays are identical

        len_ = len(faces[0])
        diff_lengths = any(len(f) != len_ for f in faces[1:])

        if diff_lengths:
            return False

        faces_0 = faces[0]
        different_contents = any(not f.allclose(faces_0) for f in faces[1:])
        return not different_contents

    @staticmethod
    def from_list(
        clip_vertices_pos: list[torch.Tensor],
        vertex_attributes: list[torch.Tensor],
        faces: list[torch.Tensor],
    ) -> "RenderInput":
        """Create RenderInput from lists of per-mesh data.

        Converts list-based input (variable mesh sizes) into the appropriate tensor
        format for rasterization. Automatically chooses between instanced and
        concatenated modes based on geometry similarity.

        Args:
            clip_vertices_pos: List of clip space positions, one tensor per mesh.
                Each tensor has shape (N_i, 4).
            vertex_attributes: List of vertex attributes, one tensor per mesh.
                Each tensor has shape (N_i, C) where C is the attribute dimension.
            faces: List of face indices, one tensor per mesh. Each tensor has
                shape (F_i, 3).

        Returns:
            RenderInput configured for either instanced or concatenated rendering
            based on the input geometry.

        Note:
            Instanced mode is used when all meshes have identical faces
            Range mode is used when meshes have different faces
        """
        faces = [f.type(torch.int32) for f in faces]

        # Use instanced rendering if geometry is identical across all meshes
        if RenderInput.is_instance_mode(faces):
            return RenderInput(
                clip_vertices_pos=torch.stack([c for c in clip_vertices_pos]),
                vertex_attributes=torch.stack([c for c in vertex_attributes]),
                faces=faces[0],
                ranges=None,
            )

        # prepare inputs for range mode as specified in https://nvlabs.github.io/nvdiffrast/#geometry-and-minibatches-range-mode-vs-instanced-mode

        def starts_and_lengths(
            tensor_list: list[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _lens = torch.tensor([0] + [len(t) for t in tensor_list])
            start_indices = torch.cumsum(_lens[:-1], dim=0)
            return start_indices, _lens[1:]

        vertex_start_indices, _vertex_lens = starts_and_lengths(clip_vertices_pos)
        faces_start_indices, faces_lens = starts_and_lengths(faces)

        # offset the faces to refer to correct vertices
        faces = torch.cat(
            [
                f + start_i
                for f, start_i in zip(faces, vertex_start_indices, strict=True)
            ]
        )

        # ranges is minibatch indices for faces
        # eg which range of faces corresponds to which batch index
        ranges = torch.stack((faces_start_indices, faces_lens), dim=-1)
        ranges = ranges.type(torch.int32)

        clip_vertices_pos = torch.cat([c for c in clip_vertices_pos])
        vertex_attributes = torch.cat([c for c in vertex_attributes])

        return RenderInput(
            clip_vertices_pos=clip_vertices_pos,
            vertex_attributes=vertex_attributes,
            faces=faces,
            ranges=ranges,
        )
