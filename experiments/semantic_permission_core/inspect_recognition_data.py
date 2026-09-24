"""Read only specified regular binary archive members and freeze a selection."""

from pathlib import Path
import hashlib
import json
import sys
import tarfile
import time
import numpy as np


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    acquisition = directory.parent / "acquisition01"
    receipts = json.loads((acquisition / "receipts.json").read_text())
    record = next(record for record in receipts if record["filename"] == "cifar100_binary.tar.gz")
    assert record["status"] == "complete"
    path = acquisition / record["filename"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    expected_sizes = {"train.bin": settings["train_records"]*settings["record_bytes"],
                      "test.bin": settings["test_records"]*settings["record_bytes"]}
    allowed = set(expected_sizes) | {"fine_label_names.txt", "coarse_label_names.txt"}
    members, arrays, labels = {}, {}, {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isdir() and member.name.rstrip("/") == "cifar-100-binary":
                continue
            name = Path(member.name).name
            assert member.name == "cifar-100-binary/"+name
            assert member.isreg() and name in allowed and name not in members
            assert member.size == expected_sizes[name] if name in expected_sizes else member.size < 8192
            stream = archive.extractfile(member)
            content = stream.read(member.size)
            assert len(content) == member.size
            members[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            if name.endswith(".bin"):
                arrays[name] = np.frombuffer(content, dtype=np.uint8).reshape(-1, settings["record_bytes"])
            else:
                labels[name] = content.decode("utf-8").splitlines()
    assert set(members) == allowed
    assert len(labels["fine_label_names.txt"]) == settings["fine_classes"]
    assert len(labels["coarse_label_names.txt"]) == settings["coarse_classes"]
    mapping = np.full(settings["fine_classes"], -1, dtype=np.int64)
    for name, data in arrays.items():
        assert np.all(data[:, 0] < settings["coarse_classes"])
        assert np.all(data[:, 1] < settings["fine_classes"])
        expected_count = len(data)//settings["fine_classes"]
        assert np.all(np.bincount(data[:, 1], minlength=settings["fine_classes"]) == expected_count)
        for fine in range(settings["fine_classes"]):
            values = np.unique(data[data[:, 1] == fine, 0])
            assert len(values) == 1
            assert mapping[fine] in (-1, values[0])
            mapping[fine] = values[0]
    assert np.all(np.bincount(mapping, minlength=settings["coarse_classes"]) == 5)
    generator = np.random.default_rng(settings["selection_seed"])
    test = arrays["test.bin"]
    selected = np.stack([generator.choice(np.flatnonzero(test[:, 1] == fine), settings["per_class"], replace=False)
                         for fine in range(settings["fine_classes"])], axis=1)
    selected_rows = test[selected.ravel()]
    images = selected_rows[:, 2:].reshape(-1, 3, 32, 32).copy()
    np.save(directory / "selected_images.npy", images, allow_pickle=False)
    np.savez(directory / "evaluation_labels.npz", fine=selected_rows[:, 1], coarse=selected_rows[:, 0], mapping=mapping)
    result = {"status": "complete", "archive_sha256": record["sha256"], "members": members,
              "selection_indices": selected.tolist(), "labels": labels, "selected_images": len(images),
              "images_sha256": hashlib.sha256((directory / "selected_images.npy").read_bytes()).hexdigest(),
              "evaluation_labels_sha256": hashlib.sha256((directory / "evaluation_labels.npz").read_bytes()).hexdigest(),
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "selected_images": len(images), "members": members,
                      "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
