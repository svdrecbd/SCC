"""Read original scene references using bounded, preserved HTTP byte ranges."""
from pathlib import Path
import hashlib
import io
import json
import sys
import time
import urllib.request
import h5py


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, directory, maximum_bytes=64_000_000):
        super().__init__()
        self.url = url
        self.size = size
        self.directory = directory
        self.maximum_bytes = maximum_bytes
        self.position = 0
        self.block_size = 131072
        self.cache = {}
        self.records = []
        self.downloaded_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.size)[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Negative file position")
        return self.position

    def block(self, number):
        if number not in self.cache:
            start = number * self.block_size
            end = min(start + self.block_size, self.size) - 1
            if self.downloaded_bytes + end - start + 1 > self.maximum_bytes:
                raise RuntimeError("Range acquisition byte limit exceeded")
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=15) as response:
                assert response.status == 206, response.status
                expected_range = f"bytes {start}-{end}/{self.size}"
                assert response.headers["Content-Range"] == expected_range
                data = response.read(end - start + 2)
            assert len(data) == end - start + 1
            self.downloaded_bytes += len(data)
            self.cache[number] = data
            name = f"range_{start:012d}_{end:012d}.bin"
            (self.directory / name).write_bytes(data)
            self.records.append({"file": name, "start": start, "end": end,
                                 "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            (self.directory / "ranges.json").write_text(json.dumps(self.records, indent=2) + "\n")
        return self.cache[number]

    def read(self, size=-1):
        remaining = self.size - self.position if size < 0 else min(size, self.size - self.position)
        result = bytearray()
        while remaining > 0:
            number, offset = divmod(self.position, self.block_size)
            block = self.block(number)
            count = min(remaining, len(block) - offset)
            result.extend(block[offset:offset + count])
            self.position += count
            remaining -= count
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def main(directory):
    started = time.perf_counter()
    with urllib.request.urlopen("https://huggingface.co/api/models/andrew-healey/nyuv2", timeout=15) as response:
        metadata_bytes = response.read(1_000_000)
    (directory / "repository_metadata.json").write_bytes(metadata_bytes)
    metadata = json.loads(metadata_bytes)
    listing_url = "https://huggingface.co/api/models/andrew-healey/nyuv2/tree/" + metadata["sha"]
    with urllib.request.urlopen(listing_url, timeout=15) as response:
        listing_bytes = response.read(1_000_000)
    (directory / "repository_listing.json").write_bytes(listing_bytes)
    file_record = next(record for record in json.loads(listing_bytes)
                       if record["path"] == "nyu_depth_v2_labeled.mat")
    url = ("https://huggingface.co/andrew-healey/nyuv2/resolve/" + metadata["sha"]
           + "/nyu_depth_v2_labeled.mat")
    source = RangeFile(url, file_record["size"], directory)
    with h5py.File(source, "r") as original:
        references = original["scenes"][:].reshape(-1)
        records = [{"index": index + 1,
                    "scene": "".join(chr(int(value)) for value in original[reference][:].reshape(-1))}
                   for index, reference in enumerate(references)]
    assert len(records) == 1449
    (directory / "scenes.json").write_text(json.dumps(records, indent=2) + "\n")
    summary = {"status": "complete", "revision": metadata["sha"], "source_url": url,
               "source_file": file_record, "frames": len(records),
               "scene_count": len({record["scene"] for record in records}),
               "downloaded_bytes": source.downloaded_bytes,
               "range_count": len(source.records), "wall_seconds": time.perf_counter() - started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
