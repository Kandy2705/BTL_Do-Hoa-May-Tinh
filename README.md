# Computer Graphics Repository: BTL 1 + BTL 2

This repository is a **shared course submission repository** for two assignments:

- **BTL 1**: interactive Computer Graphics application with 2D/3D primitives, file-loaded models, shader modes, lighting, camera control, and SGD visualization.
- **BTL 2**: synthetic road-scene dataset generator that exports RGB, depth, segmentation, bounding boxes, YOLO labels, COCO annotations, and metadata.

The repository is intentionally organized so **both assignments coexist cleanly**:

- BTL 1 keeps its original interactive application flow at the repository root.
- BTL 2 lives in its own package namespace under `btl2/`.
- Shared assets stay in `assets/`.

## Quick Start

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Run BTL 1:

```bash
python3 main.py
```

or

```bash
python3 -m btl1.main
```

Run BTL 2 directly:

```bash
python3 -m btl2.main --config configs/btl2/default.yaml generate
```

Generate the small BTL 2 demo dataset:

```bash
python3 scripts/generate_demo_dataset.py
```

Validate generated BTL 2 outputs:

```bash
python3 scripts/validate_dataset.py
```

Visualize BTL 2 annotations:

```bash
python3 scripts/visualize_annotations.py
```

## Final Structure

```text
.
├── README.md
├── requirements.txt
├── assets/
├── btl1/
├── btl2/
├── components/
├── configs/
│   └── btl2/
├── controller.py
├── core/
├── docs/
├── geometry/
├── libs/
├── main.py
├── model.py
├── outputs/
│   └── btl2/
├── scripts/
├── shaders/
│   ├── btl2/
│   └── ...
└── viewer.py
```

## What Belongs To Which Assignment

### BTL 1

Core legacy interactive app:

- `main.py`
- `controller.py`
- `viewer.py`
- `model.py`
- `components/`
- `geometry/`
- `libs/`
- `core/`
- `btl1/` wrapper

### BTL 2

Synthetic dataset generator:

- `btl2/`
- `configs/btl2/`
- `shaders/btl2/`
- `scripts/`
- `outputs/btl2/`

### Shared

- `assets/`
- `requirements.txt`
- `docs/`

## BTL 1 Notes

BTL 1 remains the original app. It was **not removed** and **not rewritten into BTL 2**.
The only BTL 1 changes in this refactor are:

- clearer menu wording for BTL 2
- a bridge panel that can trigger BTL 2 generation from inside the app
- documentation and packaging cleanup

That means the old BTL 1 workflow still exists.

## BTL 2 Notes

BTL 2 is now isolated under `btl2/` and includes:

- scene generation
- external OBJ / minimal PLY loading
- primitive fallback meshes
- RGB / depth / segmentation render passes
- bbox / YOLO / COCO / metadata export
- train / val split
- validation and visualization scripts

Default config:

- [configs/btl2/default.yaml](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/configs/btl2/default.yaml)

Demo config:

- [configs/btl2/demo_small.yaml](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/configs/btl2/demo_small.yaml)

Main package entrypoint:

- [btl2/main.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/main.py)

## Running BTL 2 From Inside BTL 1

Inside the BTL 1 app:

1. Build or load a scene in Normal Mode.
2. Add one or more cameras into the hierarchy.
3. Open the `BTL 2` menu.
4. Choose whether to use:
   - the current BTL 1 scene, or
   - the procedural road demo
5. Press `Generate Dataset`.

The viewer default camera is **not** treated as a dataset camera.
Only cameras explicitly added to the BTL 1 scene are used.

## Documentation

- Audit: [docs/repo_audit.md](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/docs/repo_audit.md)
- Folder map: [docs/folder_map.md](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/docs/folder_map.md)
- BTL 2 file explanation: [GIAI_THICH_FILE_BTL2.md](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/GIAI_THICH_FILE_BTL2.md)

## Archived / Removed Confusing Files

- `assets/textures/viewer.py`
  Moved to `docs/archive/viewer_duplicate_from_assets_textures.py` because it was a stray copy of viewer code, not a texture asset.

- `outputs/btl2/demo_dataset/README.md` (placeholder file removed during cleanup)
  Removed because it was only a placeholder and runtime code never used it.
