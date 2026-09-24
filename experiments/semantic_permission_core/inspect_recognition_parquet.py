"""Validate the pinned test-table contract and freeze class-balanced RGB inputs."""

from pathlib import Path
import hashlib
import io
import json
import sys
import time
import numpy as np
import pyarrow.parquet as parquet
from PIL import Image


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    acquisition = directory.parent / "acquisition02"
    record = json.loads((acquisition / "receipts.json").read_text())[0]
    assert record["status"] == "complete"
    path = acquisition / "cifar100_test.parquet"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    table = parquet.read_table(path)
    assert table.num_rows == settings["test_records"]
    assert set(table.column_names) == {"img", "fine_label", "coarse_label"}
    fine = np.asarray(table["fine_label"].to_numpy(), dtype=np.int64)
    coarse = np.asarray(table["coarse_label"].to_numpy(), dtype=np.int64)
    assert np.all((fine >= 0) & (fine < settings["fine_classes"]))
    assert np.all((coarse >= 0) & (coarse < settings["coarse_classes"]))
    assert np.all(np.bincount(fine, minlength=settings["fine_classes"]) == 100)
    mapping = np.empty(settings["fine_classes"], dtype=np.int64)
    for label in range(settings["fine_classes"]):
        values = np.unique(coarse[fine == label])
        assert len(values) == 1
        mapping[label] = values[0]
    assert np.all(np.bincount(mapping, minlength=settings["coarse_classes"]) == 5)
    generator = np.random.default_rng(settings["selection_seed"])
    selected = np.stack([generator.choice(np.flatnonzero(fine == label), settings["per_class"], replace=False)
                         for label in range(settings["fine_classes"])], axis=1)
    image_values = []
    image_hashes = []
    for item in table["img"].to_pylist():
        data = item["bytes"]
        with Image.open(io.BytesIO(data)) as image:
            assert image.mode == "RGB" and image.size == (32, 32)
            values = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
        image_hashes.append(hashlib.sha256(values.tobytes()).hexdigest())
        image_values.append(values)
    images = np.stack(image_values)
    np.save(directory / "selected_images.npy", images[selected.ravel()], allow_pickle=False)
    np.savez(directory / "evaluation_labels.npz", fine=fine[selected.ravel()],
             coarse=coarse[selected.ravel()], mapping=mapping)
    metadata = {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()}
    result = {"status": "complete", "parquet_sha256": record["sha256"],
              "parquet_schema": str(table.schema), "schema_metadata": metadata,
              "selection_indices": selected.tolist(), "all_decoded_image_hashes": image_hashes,
              "selected_images": int(selected.size), "verified_test_rows": len(images),
              "images_sha256": hashlib.sha256((directory / "selected_images.npy").read_bytes()).hexdigest(),
              "evaluation_labels_sha256": hashlib.sha256((directory / "evaluation_labels.npz").read_bytes()).hexdigest(),
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ["status", "parquet_sha256", "selected_images", "verified_test_rows", "images_sha256", "wall_seconds"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
