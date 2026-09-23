"""Freeze scene-disjoint calibration and evaluation inputs before inference."""
from pathlib import Path
import hashlib
import json
import re
import sys
import time
from data import read_frame


def main(directory, acquisition):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    members = json.loads((directory / "members.json").read_text())
    scenes = {record["index"]: record["scene"] for record in
              json.loads((directory / "scenes.json").read_text())}
    groups = {}
    for member in members:
        index = int(Path(member["member"]).stem)
        scene = scenes[index]
        group = re.sub(r"(?<=\d)[a-z]+$", "", scene)
        groups.setdefault(group, []).append(member | {"index": index, "scene": scene, "group": group})
    def order(value):
        return hashlib.sha256(f'{configuration["selection_seed"]}:{value}'.encode()).hexdigest()
    selected = []
    for group in sorted(groups, key=order)[:configuration["scene_count"]]:
        record = min(groups[group], key=lambda member: order(member["member"]))
        _, depth, digest = read_frame(acquisition, record)
        selected.append(record | {"selection_index": len(selected), "sha256": digest,
                                  "partition": "calibration" if len(selected) < configuration["calibration_scenes"] else "evaluation",
                                  "depth_minimum": float(depth.min()), "depth_maximum": float(depth.max())})
    assert len(selected) == configuration["scene_count"]
    assert len({record["group"] for record in selected}) == len(selected)
    (directory / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    summary = {"status": "complete", "available_groups": len(groups), "selected_scenes": len(selected),
               "calibration_scenes": configuration["calibration_scenes"],
               "wall_seconds": time.perf_counter() - started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
