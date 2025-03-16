import itertools
import webcolors
from pathlib import Path


def hex_to_color_name(hex_color):
    try:
        color_name = webcolors.hex_to_name(hex_color)
    except ValueError:
        color_name = hex_color[1:]
    return color_name


def all_sublists(lst):
    sublists = []
    for length in range(1, len(lst) + 1):
        for combination in itertools.combinations(lst, length):
            sublists.append(list(combination))
    return sublists


def check_available_scans(d):
    # Check for a list with 1 element
    for key, value in d.items():
        if isinstance(value, list) and len(value) == 1:
            return key, value

    # Check for N same lists of size N
    list_counts = {}
    for key, value in d.items():
        if isinstance(value, list):
            list_tuple = tuple(value)
            if list_tuple in list_counts:
                list_counts[list_tuple].append(key)
            else:
                list_counts[list_tuple] = [key]

    for list_tuple, keys in list_counts.items():
        if len(keys) == len(list_tuple):
            return keys, tuple(list_tuple)

    return None
