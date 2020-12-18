"""
Basic operations on trees.
"""

import numpy as np


def head_to_adj(sent_len, head, tokens, label, len_, mask, directed=False, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    label_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

    assert not isinstance(head, list)
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    label = label[:len_].tolist()
    # print('tokens', tokens)
    # print('head', head, len(head))
    # print('label', label)
    # print('mask', mask, len(mask))
    asp_idx = [idx for idx in range(len(mask)) if mask[idx] == 1]
    for idx, head in enumerate(head):
        if idx in asp_idx:
            for k in asp_idx:
                adj_matrix[idx][k] = 1
                label_matrix[idx][k] = 2
        if head != 0:
            adj_matrix[idx, head - 1] = 1
            label_matrix[idx, head - 1] = label[idx]
        else:
            if self_loop:
                adj_matrix[idx, idx] = 1
                label_matrix[idx, idx] = 2
                continue
        if not directed:
            adj_matrix[head - 1, idx] = 1
            label_matrix[head - 1, idx] = label[idx]
        if self_loop:
            adj_matrix[idx, idx] = 1
            label_matrix[idx, idx] = 2

    return adj_matrix, label_matrix
