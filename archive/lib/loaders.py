import os
import tensorflow as tf

from archive.lib.language_index import LanguageIndex
from archive.lib.preprocessors.naive_words import naive_words


def en_sp(num_examples: int) -> tuple[LanguageIndex, LanguageIndex]:
    path_to_file, = __download_and_extract('http://download.tensorflow.org/data/spa-eng.zip', "spa-eng/spa.txt")

    word_pairs = [[naive_words(sentence, punctuations="?.!,¿'").split(' ') for sentence in l.split('\t')] for l in
                  __read_lines(path_to_file, num_examples)]

    return LanguageIndex([inp for inp, targ in word_pairs]), LanguageIndex([targ for inp, targ in word_pairs])


def en_tw(num_examples: int = None) -> tuple[LanguageIndex, LanguageIndex]:
    en_path, tw_path = __download_and_extract('https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-tw.txt.zip',
                                              'NLLB.en-tw.en', 'NLLB.en-tw.tw')

    en = [naive_words(line, punctuations="?.!,'").split(' ') for line in __read_lines(en_path, num_examples)]
    tw = [naive_words(line, punctuations="?.!,¿'", special_chars='ɛƐɔƆ').split(' ') for line in
          __read_lines(tw_path, num_examples)]

    return LanguageIndex(en), LanguageIndex(tw)


def __read_lines(path_to_file: str, num_lines: int = None):
    with open(path_to_file, 'r', encoding='UTF-8') as file:
        if num_lines is None:
            return file.readlines()
        for i, line in enumerate(file):
            if i >= num_lines:
                break
            yield line.strip()


def __download_and_extract(url: str, *files: str) -> list[str]:
    subdir, _ = os.path.splitext(os.path.basename(url))
    path_to_zip = tf.keras.utils.get_file(cache_subdir=f"datasets/{subdir}", origin=url, extract=True)
    path_to_dir = os.path.dirname(path_to_zip)
    return [f"{path_to_dir}/{file}" for file in files]
