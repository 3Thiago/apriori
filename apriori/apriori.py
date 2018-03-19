import tensorflow as tf
import numpy as np
from typing import List, Iterable, Tuple, Sized
from collections import namedtuple
from apriori_groups import get_next_mask_and_groups

GroupCounts = namedtuple('GroupCounts', 'group, count')


def get_input_collections_as_binary_arrays(input_collections: Iterable[Iterable[any]]) -> Tuple[List, List]:
    """Takes the original input rows, returns all unique elements and inputs mapped to binary occurrence matrix
    Args:
        input_collections: The inputs in their original format e.g. [['A','B','C', ...], ...].
        The elements can be any sortable type

    Returns:
        all_input_elements: The 'N' sorted, unique elements appearing in the inputs of shape (N,)
        binarised_input_collections: The 'X' inputs converted to binary matrix of shape (X,N,)
    """
    # First get complete set of input elements
    all_input_elements = list(sorted(set([el for row in input_collections for el in row])))
    # For each input collection, create array with 1 for element exists or 0 for not exists
    # (extension: count the number of elements - for use with multiple element counting version)
    binarised_input_collections = [tf.Variable([1 if el in row else 0 for el in all_input_elements], dtype=tf.int32)
                                   for row in input_collections]
    print('Binarised input rows')
    return all_input_elements, binarised_input_collections


class AprioriFrequentSets:
    def __init__(self, group_counts_list: List[GroupCounts], min_supp):
        self.group_counts_list = group_counts_list
        self.min_support = min_supp

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False


def apriori(input_id_to_collections: any, min_support: int = 0) -> AprioriFrequentSets:
    """
    TODO: when input_rows is dict with ids, also collect all the ids which are in the frequent item groups
    """
    if isinstance(input_id_to_collections, dict):
        all_input_elements, vectorised_input_collections = get_input_collections_as_binary_arrays(
            input_id_to_collections.values()
        )
        original_input_rows_count = len(input_id_to_collections.keys())
    elif isinstance(input_id_to_collections, Iterable):
        all_input_elements, vectorised_input_collections = get_input_collections_as_binary_arrays(
            input_id_to_collections
        )
        if isinstance(input_id_to_collections, Sized):
            original_input_rows_count = len(input_id_to_collections)
        else:
            raise ValueError('Input rows iterable must be sizable')
    else:
        raise ValueError('Input rows should be iterable or dict type')

    with tf.Session() as session:

        original_els_count = len(all_input_elements)

        vectorised_inputs_stack = tf.stack(vectorised_input_collections)

        single_el_occurances = tf.reduce_sum(vectorised_inputs_stack, axis=0)

        # Get single set groups as itemsets (in original formats)
        single_member_indices = tf.where(single_el_occurances >= 0)
        single_member_groups = tf.gather_nd(all_input_elements, single_member_indices)
        single_member_group_counts = tf.gather_nd(single_el_occurances, single_member_indices)

        next_el_indices = tf.where(single_el_occurances >= min_support)

        frequent_single_bin_mask = tf.cast(single_el_occurances >= min_support, tf.int32)

        init = tf.global_variables_initializer()
        session.run(init)

        # Add these to the group counts total
        single_mem_indices, single_mem_groups, single_mem_counts, single_el_mask, single_el_indices = \
            session.run([single_member_indices,
                         single_member_groups,
                         single_member_group_counts,
                         frequent_single_bin_mask,
                         next_el_indices])

        # TODO: add as frequent groups single_mem_indices

        gc = {
            "current_N": 1,
            "prev_group_indices": single_el_indices,
            "original_els": all_input_elements,
            "num_original_els": original_els_count,
            "input_rows": vectorised_input_collections,
            "num_input_rows": original_input_rows_count,
            "curr_bin_mask": single_el_mask,
            "num_remaining_els": len(single_el_indices),
            "min_support": min_support
        }
        for dim in range(2, 5):
            gc["current_N"] = dim

            try:
                frequent_groups_indices, frequent_groups_counts, next_possible_groups_indices, next_mask = \
                    get_next_mask_and_groups(gc)
            except ValueError as err:
                print(err)
                break

            group_counts_array = session.run([frequent_groups_indices,
                                              frequent_groups_counts,
                                              next_possible_groups_indices])
            print(group_counts_array)
            gc["prev_group_indices"] = group_counts_array[2]
            gc["curr_bin_mask"] = next_mask
            print(f"next_mask:{next_mask}")
            gc["num_remaining_els"] = np.sum(next_mask)
            # TODO: convert group_counts_array elements to GroupCounts objects and collect in some list
        return group_counts_array, next_mask
