import unittest
import tensorflow as tf
import numpy as np

from apriori_groups import get_next_mask_and_groups


class TestAprioriGroups(unittest.TestCase):
    def test_get_next_mask_and_groups(self):
        original_input_els = ['A', 'B', 'C', 'D', 'E']
        vectorised_inputs_stack_2 = tf.Variable([[1, 1, 1, 1, 0],
                                                 [1, 1, 0, 1, 1],
                                                 [0, 1, 1, 1, 0]], tf.int32)
        frequent_bin_mask_2 = np.array([1, 1, 1, 1, 0], dtype=np.int32)
        prev_group_indices_2 = np.array([[0, 1], [1, 2], [2, 3], [1, 3], [0, 3]], dtype=np.int32)

        input_3_5 = {
            "current_N": 3,
            "prev_group_indices": prev_group_indices_2,
            "original_els": original_input_els,
            "num_original_els": 5,
            "input_rows": vectorised_inputs_stack_2,
            "num_input_rows": 3,
            "curr_bin_mask": frequent_bin_mask_2,
            "num_remaining_els": 4,
            "curr_group_count_totals": [],
            "min_support": 1
        }
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            frequent_groups_indices, frequent_groups_counts, next_possible_groups_indices, next_mask = \
                get_next_mask_and_groups(input_3_5)
            group_counts_array = session.run([frequent_groups_indices,
                                              frequent_groups_counts,
                                              next_possible_groups_indices])
        # TODO: equality
        print(group_counts_array, next_mask)
