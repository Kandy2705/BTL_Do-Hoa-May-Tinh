# BTL 2

BTL 2 is implemented as a clean package under `btl2/`.

Main responsibilities:

- road-scene generation
- OBJ/PLY loading with primitive fallback
- RGB / depth / segmentation render passes
- bbox / YOLO / COCO / metadata export
- dataset validation and visualization scripts

Entry point:

```bash
python3 -m btl2.main --config configs/btl2/default.yaml generate
```

Helper scripts:

- `python3 scripts/generate_demo_dataset.py`
- `python3 scripts/validate_dataset.py`
- `python3 scripts/visualize_annotations.py`
