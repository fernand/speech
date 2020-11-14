CHARS = "_' abcdefghijklmnopqrstuvwxyz"

char_to_i = dict((c, i) for i, c in enumerate(CHARS))
i_to_char = dict((i, c) for i, c in enumerate(CHARS))


def text_to_int(text):
    return [char_to_i[c] for c in text]


def int_to_text(labels):
    return "".join([i_to_char[i] for i in labels])
