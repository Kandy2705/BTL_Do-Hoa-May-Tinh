# BTL 1

BTL 1 remains implemented by the original interactive OpenGL editor at the
repository root:

- `main.py`
- `controller.py`
- `viewer.py`
- `model.py`
- `components/`
- `geometry/`
- `libs/`
- `core/`

The package `btl1/` is a thin namespace wrapper so the shared repository has a
clear entrypoint:

```bash
python3 -m btl1.main
```

Legacy execution still works:

```bash
python3 main.py
```
