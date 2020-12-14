import Levenshtein as Lev
import torch

import text


def wer(s1, s2):
    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance("".join(w1), "".join(w2)) / len(w1)


def cer(s1, s2):
    s1, s2, = (
        s1.replace(" ", ""),
        s2.replace(" ", ""),
    )
    return Lev.distance(s1, s2) / len(s1)


def greedy_decoder(output, labels, label_lengths, blank_label=0):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text.int_to_text(labels[i][: label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                # Collapse repeats.
                if j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text.int_to_text(decode))
    return decodes, targets


def greedy_decode(output, blank_label=0):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                # Collapse repeats.
                if j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text.int_to_text(decode))
    return decodes
