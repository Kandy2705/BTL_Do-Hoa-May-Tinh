"""Mesh loading wrapper with procedural fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from btl2.utils.math3d import AABB


@dataclass
class MeshData:
    """CPU-side mesh buffers and bounds used to upload geometry later."""

    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    aabb: AABB
    texcoords: np.ndarray | None = None
    texture_path: Path | None = None


class ObjectLoader:
    """Load OBJ and simple ASCII PLY meshes, or fall back to primitives."""

    def __init__(self, asset_root: str | Path) -> None:
        self.asset_root = Path(asset_root)
        self._cache: dict[str, MeshData] = {}

    def load_or_primitive(self, relative_path: str | None, primitive_name: str) -> MeshData:
        """Try an external file first and use a procedural primitive on failure."""
        key = relative_path or primitive_name
        if key in self._cache:
            return self._cache[key]

        mesh: MeshData
        if relative_path:
            asset_path = self.asset_root / relative_path
            if asset_path.exists():
                suffix = asset_path.suffix.lower()
                if suffix == ".obj":
                    mesh = self._load_obj(asset_path)
                elif suffix == ".ply":
                    mesh = self._load_ply_ascii(asset_path)
                else:
                    mesh = self._primitive(primitive_name)
            else:
                mesh = self._primitive(primitive_name)
        else:
            mesh = self._primitive(primitive_name)

        self._cache[key] = mesh
        return mesh

    def _load_obj(self, path: Path) -> MeshData:
        """Load a minimal subset of OBJ: v, vt, vn, mtllib/usemtl, and faces."""
        positions: list[list[float]] = []
        texcoords: list[list[float]] = []
        normals: list[list[float]] = []
        vertices_out: list[list[float]] = []
        texcoords_out: list[list[float]] = []
        normals_out: list[list[float]] = []
        indices_out: list[int] = []
        vertex_map: dict[tuple[int, int | None, int | None], int] = {}
        materials: dict[str, Path] = {}
        current_material: str | None = None
        texture_path: Path | None = None

        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                head = parts[0]
                if head == "v":
                    positions.append([float(v) for v in parts[1:4]])
                elif head == "vt":
                    u = float(parts[1]) if len(parts) > 1 else 0.0
                    v = float(parts[2]) if len(parts) > 2 else 0.0
                    texcoords.append([u, v])
                elif head == "vn":
                    normals.append([float(v) for v in parts[1:4]])
                elif head == "mtllib":
                    mtl_name = " ".join(parts[1:])
                    mtl_path = Path(mtl_name) if Path(mtl_name).is_absolute() else path.parent / mtl_name
                    materials.update(self._load_mtl_textures(mtl_path))
                elif head == "usemtl":
                    current_material = " ".join(parts[1:]) if len(parts) > 1 else None
                    if texture_path is None and current_material in materials:
                        texture_path = materials[current_material]
                elif head == "f":
                    face_indices = [
                        self._parse_obj_corner(token, len(positions), len(texcoords), len(normals))
                        for token in parts[1:]
                    ]
                    triangles = self._triangulate_polygon(face_indices)
                    for tri in triangles:
                        for pos_idx, tex_idx, norm_idx in tri:
                            cache_key = (pos_idx, tex_idx, norm_idx)
                            if cache_key not in vertex_map:
                                vertex_map[cache_key] = len(vertices_out)
                                vertices_out.append(positions[pos_idx])
                                if tex_idx is not None and texcoords:
                                    texcoords_out.append(texcoords[tex_idx])
                                else:
                                    texcoords_out.append([0.0, 0.0])
                                if norm_idx is not None and normals:
                                    normals_out.append(normals[norm_idx])
                                else:
                                    normals_out.append([0.0, 1.0, 0.0])
                            indices_out.append(vertex_map[cache_key])

        if texture_path is None:
            texture_path = self._guess_folder_texture_path(path.parent)

        return self._finalize_mesh(vertices_out, normals_out, indices_out, texcoords_out, source_path=path, texture_path=texture_path)

    @staticmethod
    def _load_mtl_textures(path: Path) -> dict[str, Path]:
        """Read diffuse texture references from an MTL file."""
        textures: dict[str, Path] = {}
        if not path.exists():
            return textures

        current_material: str | None = None
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0].lower()
                if key == "newmtl" and len(parts) > 1:
                    current_material = " ".join(parts[1:])
                elif key == "map_kd" and current_material and len(parts) > 1:
                    texture_name = " ".join(parts[1:])
                    texture_path = Path(texture_name)
                    if not texture_path.is_absolute():
                        texture_path = path.parent / texture_path
                    if texture_path.exists():
                        textures[current_material] = texture_path.resolve()
        return textures

    @staticmethod
    def _guess_folder_texture_path(folder: Path) -> Path | None:
        """Pick a likely diffuse/base-color texture from the OBJ folder."""
        if not folder.exists():
            return None

        image_paths: list[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tga", "*.bmp"):
            image_paths.extend(sorted(folder.glob(pattern)))
        if not image_paths:
            return None

        bad_tokens = ("normal", "rough", "metal", "ao", "opacity", "alpha", "wire", "clay")
        preferred_tokens = ("basecolor", "base_color", "diffuse", "albedo", "color")

        def score(path: Path) -> tuple[int, str]:
            stem = path.stem.lower()
            if any(token in stem for token in bad_tokens):
                return (2, stem)
            if any(token in stem for token in preferred_tokens):
                return (0, stem)
            return (1, stem)

        return sorted(image_paths, key=score)[0].resolve()

    def _load_ply_ascii(self, path: Path) -> MeshData:
        """Load a minimal ASCII PLY with vertex positions and face lists."""
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = [line.rstrip("\n") for line in handle]

        if not lines or lines[0] != "ply":
            return self._primitive("box")

        vertex_count = 0
        face_count = 0
        header_end = 0
        for idx, line in enumerate(lines):
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                header_end = idx + 1
                break

        vertex_lines = lines[header_end : header_end + vertex_count]
        face_lines = lines[header_end + vertex_count : header_end + vertex_count + face_count]

        positions = np.array([[float(v) for v in line.split()[:3]] for line in vertex_lines], dtype=np.float32)
        normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (positions.shape[0], 1))
        indices: list[int] = []

        for line in face_lines:
            values = [int(v) for v in line.split()]
            count = values[0]
            face = values[1 : 1 + count]
            for tri in self._triangulate_index_list(face):
                indices.extend(tri)

        return self._finalize_mesh(positions.tolist(), normals.tolist(), indices, source_path=path)

    @staticmethod
    def _parse_obj_corner(token: str, num_positions: int, num_texcoords: int, num_normals: int) -> tuple[int, int | None, int | None]:
        """Convert OBJ face corner syntax into zero-based indices."""
        values = token.split("/")
        pos_idx = int(values[0])
        pos_idx = pos_idx - 1 if pos_idx > 0 else num_positions + pos_idx
        tex_idx = None
        if len(values) >= 2 and values[1]:
            parsed = int(values[1])
            tex_idx = parsed - 1 if parsed > 0 else num_texcoords + parsed
        norm_idx = None
        if len(values) >= 3 and values[2]:
            parsed = int(values[2])
            norm_idx = parsed - 1 if parsed > 0 else num_normals + parsed
        return pos_idx, tex_idx, norm_idx

    @staticmethod
    def _triangulate_polygon(face_indices: list[tuple[int, int | None, int | None]]) -> list[list[tuple[int, int | None, int | None]]]:
        """Triangulate an OBJ face using a fan around the first vertex."""
        if len(face_indices) < 3:
            return []
        return [[face_indices[0], face_indices[i], face_indices[i + 1]] for i in range(1, len(face_indices) - 1)]

    @staticmethod
    def _triangulate_index_list(face: list[int]) -> list[list[int]]:
        """Triangulate a list of vertex indices using a fan."""
        if len(face) < 3:
            return []
        return [[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)]

    def _primitive(self, primitive_name: str) -> MeshData:
        """Return one of the built-in fallback meshes."""
        primitive_name = primitive_name.lower()
        if primitive_name == "plane":
            vertices = np.array(
                [
                    [-0.5, 0.0, -0.5],
                    [0.5, 0.0, -0.5],
                    [0.5, 0.0, 0.5],
                    [-0.5, 0.0, 0.5],
                ],
                dtype=np.float32,
            )
            normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1))
            texcoords = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            return MeshData(vertices, normals, indices, AABB(vertices.min(axis=0), vertices.max(axis=0)), texcoords=texcoords)

        if primitive_name == "cylinder":
            return self._build_cylinder()

        return self._build_box()

    def _build_box(self) -> MeshData:
        """Create a centered unit box used for cars, signs, and lights."""
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        indices = np.array(
            [
                0, 1, 2, 0, 2, 3,
                4, 5, 6, 4, 6, 7,
                0, 1, 5, 0, 5, 4,
                2, 3, 7, 2, 7, 6,
                1, 2, 6, 1, 6, 5,
                0, 3, 7, 0, 7, 4,
            ],
            dtype=np.uint32,
        )
        texcoords = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        return MeshData(vertices, normals, indices, AABB(vertices.min(axis=0), vertices.max(axis=0)), texcoords=texcoords)

    def _build_cylinder(self, segments: int = 16) -> MeshData:
        """Create a simple vertical cylinder placeholder for person-class objects."""
        vertices: list[list[float]] = []
        normals: list[list[float]] = []
        indices: list[int] = []
        for ring_y in (-0.5, 0.5):
            for i in range(segments):
                angle = 2.0 * np.pi * i / segments
                x = 0.5 * np.cos(angle)
                z = 0.5 * np.sin(angle)
                vertices.append([x, ring_y, z])
                normals.append([np.cos(angle), 0.0, np.sin(angle)])

        for i in range(segments):
            j = (i + 1) % segments
            bottom_a, bottom_b = i, j
            top_a, top_b = i + segments, j + segments
            indices.extend([bottom_a, bottom_b, top_b, bottom_a, top_b, top_a])

        vertices_np = np.asarray(vertices, dtype=np.float32)
        normals_np = np.asarray(normals, dtype=np.float32)
        indices_np = np.asarray(indices, dtype=np.uint32)
        texcoords_np = np.zeros((vertices_np.shape[0], 2), dtype=np.float32)
        return MeshData(vertices_np, normals_np, indices_np, AABB(vertices_np.min(axis=0), vertices_np.max(axis=0)), texcoords=texcoords_np)

    @staticmethod
    def _finalize_mesh(
        vertices: list[list[float]],
        normals: list[list[float]],
        indices: list[int],
        texcoords: list[list[float]] | None = None,
        source_path: Path | None = None,
        texture_path: Path | None = None,
    ) -> MeshData:
        """Convert parsed mesh lists to NumPy arrays and build the AABB."""
        if not vertices or not indices:
            loader = ObjectLoader(".")
            return loader._build_box()
        vertices_np = np.asarray(vertices, dtype=np.float32)
        normals_np = np.asarray(normals, dtype=np.float32)
        indices_np = np.asarray(indices, dtype=np.uint32)
        texcoords_np = np.asarray(texcoords, dtype=np.float32) if texcoords else np.zeros((vertices_np.shape[0], 2), dtype=np.float32)
        if texcoords_np.shape != (vertices_np.shape[0], 2):
            texcoords_np = np.zeros((vertices_np.shape[0], 2), dtype=np.float32)
        vertices_np, normals_np = ObjectLoader._apply_source_orientation(vertices_np, normals_np, source_path)
        vertices_np = ObjectLoader._normalize_loaded_vertices(vertices_np)
        return MeshData(
            vertices_np,
            normals_np,
            indices_np,
            AABB(vertices_np.min(axis=0), vertices_np.max(axis=0)),
            texcoords=texcoords_np,
            texture_path=texture_path,
        )

    @staticmethod
    def _apply_source_orientation(
        vertices: np.ndarray,
        normals: np.ndarray,
        source_path: Path | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fix known asset coordinate systems before the mesh is normalized."""
        if source_path is None:
            return vertices, normals

        lowered = str(source_path).replace("\\", "/").lower()
        if not any(token in lowered for token in ("traffic_light", "trafficlights", "stoplight", "signal")):
            return vertices, normals

        vertices = np.nan_to_num(vertices, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        normals = np.nan_to_num(normals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # The bundled stoplight is authored Z-up; the rest of the scene is Y-up.
        rotated_vertices = np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])).astype(np.float32)
        rotated_normals = np.column_stack((normals[:, 0], normals[:, 2], -normals[:, 1])).astype(np.float32)
        return rotated_vertices, rotated_normals

    @staticmethod
    def _normalize_loaded_vertices(vertices: np.ndarray) -> np.ndarray:
        """Center imported meshes and scale their longest side to roughly unit size."""
        if vertices.size == 0:
            return vertices

        centered = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        extent = centered.max(axis=0) - centered.min(axis=0)
        longest_side = float(np.max(extent))
        if longest_side <= 1e-6:
            return centered
        return centered / longest_side
