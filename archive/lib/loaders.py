import os
import tensorflow as tf


def read_lines(path_to_file: str, num_lines: int = None):
    with open(path_to_file, 'r', encoding='UTF-8') as file:
        for i, line in enumerate(file):
            if num_lines is not None and i >= num_lines:
                break
            yield line.strip()


def __download_and_extract(url: str, *files: str) -> list[str]:
    subdir, _ = os.path.splitext(os.path.basename(url))
    path_to_zip = tf.keras.utils.get_file(cache_subdir=f"datasets/{subdir}", origin=url, extract=True)
    path_to_dir = os.path.dirname(path_to_zip)
    return [f"{path_to_dir}/{file}" for file in files]
