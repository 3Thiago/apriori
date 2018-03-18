import tensorflow as tf
import numpy as np
from numpy import ndarray
from itertools import combinations


def are_group_subsets_frequent(prev_groups, group_idxs, _group_size):
    return all([(frozenset(sub_group) in prev_groups) for sub_group in combinations(group_idxs, _group_size - 1)])


def get_mapped_groups_cube_idxs_and_new_els_count(groups_cube_idxs):
    new_els = sorted(set([i for idx in groups_cube_idxs for i in idx]))
    groups_cube_idxs_map = {idx: i for i, idx in enumerate(new_els)}
    mapped_groups_cube_idxs = np.array([[groups_cube_idxs_map[i] for i in idx] for idx in groups_cube_idxs],
                                       dtype=np.int32)
    return mapped_groups_cube_idxs, len(new_els)


def get_possible_groups_cube_and_mask(prev_group_indices: ndarray, group_size: int, current_mask):
    """
    TODO: Fix for case when no group_size groups are possible
    """
    prev_groups = set(map(frozenset, prev_group_indices))
    possible_el_indices = set(prev_group_indices.flatten())
    possible_el_combinations = list(combinations(possible_el_indices, group_size))
    groups_cube_idxs = [group_idxs for group_idxs in possible_el_combinations
                        if are_group_subsets_frequent(prev_groups, group_idxs, group_size)]
    groups_cube_values = np.array([1 for _ in range(len(groups_cube_idxs))], dtype=np.int32)
    new_possible_el_indices = set([i for idx in groups_cube_idxs for i in idx])
    if len(new_possible_el_indices) == 0:
        raise ValueError('No possible elements')
    mask = [1 if i in new_possible_el_indices else 0 for i in range(len(current_mask))]
    mapped_groups_cube_idxs, new_tot_els = get_mapped_groups_cube_idxs_and_new_els_count(groups_cube_idxs)
    groups_cube_shape = np.array([new_tot_els for _ in range(group_size)], dtype=np.int32)
    return tf.sparse_to_dense(sparse_indices=mapped_groups_cube_idxs,
                              output_shape=groups_cube_shape,
                              sparse_values=groups_cube_values, ), mask
