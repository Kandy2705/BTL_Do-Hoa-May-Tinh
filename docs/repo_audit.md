# Repo Audit

## BTL 1

These modules belong to BTL 1 and stay in the legacy interactive application:

- `main.py`
- `controller.py`
- `viewer.py`
- `model.py`
- `components/`
- `geometry/`
- `libs/`
- `core/`

## BTL 2

These modules belong to BTL 2 and were grouped into a clean package:

- `btl2/`
- `configs/btl2/`
- `shaders/btl2/`
- `scripts/`
- `outputs/btl2/`

## Shared Assets

- `assets/models/`
- `assets/textures/`
- `assets/fonts/`
- `requirements.txt`
- `README.md`

## Archived / Dead / Confusing

- `assets/textures/viewer.py`
  This was a duplicate copy of the main viewer code accidentally living inside
  the texture folder. It was moved to `docs/archive/` because it is not a valid
  texture asset and only causes confusion.

- `outputs/demo_dataset/README.md`
  This was a placeholder marker file. It was removed because runtime code never
  used it.
