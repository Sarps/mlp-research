import unicodedata
import re


def naive_words(w: str, punctuations: str = "", special_chars: str = "") -> str:
    w = w.lower().strip()
    w = ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')

    replacements = [
        (r"[‘’´`ʼ]", "'"),
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        (rf"([{re.escape(punctuations)}])", r" \1 "),
        (r'[" "]+', " "),
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        (rf"[^a-zA-Z{re.escape(special_chars)}{re.escape(punctuations)}]+", " "),
    ]

    for regex, replacement in replacements:
        w = re.sub(regex, replacement, w)

    return w.rstrip().strip()
