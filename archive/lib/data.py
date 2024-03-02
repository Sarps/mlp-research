from .language_index import LanguageIndex
from .loaders import __download_and_extract, read_lines
from .preprocessors.naive_words import naive_words


def en_sp(num_examples: int) -> tuple[LanguageIndex, LanguageIndex]:
    path_to_file, = __download_and_extract('http://download.tensorflow.org/data/spa-eng.zip', "spa-eng/spa.txt")

    word_pairs = [[naive_words(sentence, punctuations="?.!,¿'").split(' ') for sentence in l.split('\t')] for l in
                  read_lines(path_to_file, num_examples)]

    return LanguageIndex('en_sp - en', [inp for inp, targ in word_pairs]), LanguageIndex('en_sp - sp',
                                                                                         [targ for inp, targ in
                                                                                          word_pairs])


def en_tw(num_examples: int = None) -> tuple[LanguageIndex, LanguageIndex]:
    en_path, tw_path = __download_and_extract('https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-tw.txt.zip',
                                              'NLLB.en-tw.en', 'NLLB.en-tw.tw')

    en = [naive_words(line, punctuations="?.!,'").split(' ') for line in read_lines(en_path, num_examples)]
    tw = [naive_words(line, punctuations="?.!,'", special_chars='ɛƐɔƆ').split(' ') for line in
          read_lines(tw_path, num_examples)]

    return LanguageIndex('en_tw-en', en), LanguageIndex('en_tw-tw', tw)
