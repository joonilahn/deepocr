import unicodedata

from soynlp.hangle import levenshtein


def replace_char_in_word(word, corpus_dict, replace_char):
    """Change chracters which are not in the corpus_dict to new_char.

    Args:
        word (str): input word.
        corpus_dict (dict): corpus dictionary.

    Returns:
        new_word (str): new word with replaced characters.
    """

    new_word = ""
    for char in word:
        if corpus_dict.get(char) is not None:
            new_word += char
        else:
            new_word += replace_char
    return new_word


def is_correct(pred, gt, exclude=None, mode="ignore_case"):
    """Compare pred and gt and return the result.

    Args:
        pred (str): prediction.
        gt (str): ground truth.
        mode (str): comparison mode. Default to 'strict'.
    """
    if mode == "ignore_case":
        pred = pred.lower()
        gt = gt.lower()

    if exclude is not None:
        for char in exclude:
            pred = pred.replace(char, "")
            gt = gt.replace(char, "")

    if pred == gt:
        return True
    else:
        return False


def fill_str_with_space(input_s: str, max_size=35, fill_char=" ") -> str:
    """Adjust empty spaces between words for pretty print.

    Args:
        input_s (str): Input string.
        max_size (int, optional): Maximum size for the output string. Defaults to 35.
        fill_char (str, optional): Filling character. Defaults to " ".

    Returns:
        str: Adjusted output string.
    """
    l = 0
    for c in input_s:
        if unicodedata.east_asian_width(c) in ["F", "W"]:
            l += 2
        else:
            l += 1
    return input_s + fill_char * (max_size - l)


def reverse_label_list(label_list: list) -> list:
    """Reverse elements for a list of lists. (list[list])

    Args:
        label_list (list):
            A list containing multiple lists.
            Each list may have multiple elements.

    Returns:
        list: Reversed list.
    """
    label_list_rev = []
    for l in label_list:
        label_list_rev.append(l[::-1])
    return label_list_rev


def compute_levenshtein_distance(pred, gt, exclude=None):
    if exclude:
        for char in exclude:
            pred = pred.replace(char, "")
            gt = gt.replace(char, "")
    return levenshtein(pred, gt)