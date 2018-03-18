import tensorflow as tf
import numpy as np
from numpy import ndarray
from typing import List, Iterable, Tuple, Dict
from collections import namedtuple
from apriori_groups import get_next_mask_and_groups

GroupCounts = namedtuple('GroupCounts', 'group, count')


def get_input_collections_as_binary_arrays(input_collections: List[Iterable]) -> Tuple[List, List[ndarray]]:
    """
    TODO: Get binarised_input_collections with tensorflow?
    """
    # First get complete set of input elements
    all_input_elements = list(sorted(set([el for row in input_collections for el in row])))
    # For each input collection, create array with 1 for element exists or 0 for not exists
    # (extension: count the number of elements - for use with multiple element counting version)
    binarised_input_collections = [tf.Variable([1 if el in row else 0 for el in all_input_elements], dtype=tf.int32)
                                   for row in input_collections]
    return (all_input_elements, binarised_input_collections,)


class AprioriFrequentSets:
    def __init__(self, group_counts_list: List[GroupCounts], min_supp):
        self.group_counts_list = group_counts_list
        self.min_support = min_supp

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False


def apriori(input_id_to_collections, min_support: int = 0) -> AprioriFrequentSets:
    """
    TODO: handle case where input_rows is list type
    TODO: when input_rows is dict with ids, also collect all the ids which are in the frequent item groups
    """
    with tf.Session() as session:
        all_input_elements, vectorised_input_collections = get_input_collections_as_binary_arrays(
            input_id_to_collections.values()
        )
        original_els_count = len(all_input_elements)
        original_input_rows_count = len(input_id_to_collections.keys())

        number_of_unique_input_els = len(all_input_elements)

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
            except ValueError:
                break

            group_counts_array = session.run([frequent_groups_indices,
                                              frequent_groups_counts,
                                              next_possible_groups_indices])
            print(group_counts_array)
            gc["prev_group_indices"] = group_counts_array[2]
            gc["curr_bin_mask"] = next_mask
            gc["num_remaining_els"] = np.sum(next_mask)
            # TODO: convert group_counts_array elements to GroupCounts objects and collect in some list
        return group_counts_array, next_mask
