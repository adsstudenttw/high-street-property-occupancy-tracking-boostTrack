#!/usr/bin/env python3

from pathlib import Path

import onnx


def main() -> None:
    mapping_path = Path(onnx.__file__).resolve().parent / "mapping.py"
    text = mapping_path.read_text()
    old = "int(TensorProto.STRING): np.dtype(np.object)"
    new = "int(TensorProto.STRING): np.dtype(object)"

    if old in text:
        mapping_path.write_text(text.replace(old, new))
        print(f"Patched {mapping_path}")
    else:
        print(f"No patch needed in {mapping_path}")


if __name__ == "__main__":
    main()
