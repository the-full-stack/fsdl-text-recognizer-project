#!/usr/bin/env python
"""Update metadata.toml with SHA-256 hash of the current file."""
from pathlib import Path
import argparse

import toml

from text_recognizer import util


def _get_metadata_filename():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to the metadata.toml file to update.")
    args = parser.parse_args()
    return Path(args.filename).resolve()


def main():
    metadata_filename = _get_metadata_filename()

    metadata = toml.load(metadata_filename)

    data_filename = metadata_filename.parents[0] / metadata["filename"]
    supposed_data_sha256 = metadata["sha256"]
    actual_data_sha256 = util.compute_sha256(data_filename)

    if supposed_data_sha256 == actual_data_sha256:
        print("Nothing to update: SHA-256 matches")
        return

    print("Updating metadata SHA-256")
    metadata["sha256"] = actual_data_sha256
    with open(metadata_filename, "w") as f:
        toml.dump(metadata, f)


if __name__ == "__main__":
    main()
