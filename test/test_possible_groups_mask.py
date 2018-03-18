import unittest
import tensorflow as tf
import numpy as np
from apriori.possible_groups_mask import get_possible_groups_cube_and_mask


class TestPossibleGroupsMask(unittest.TestCase):

    def test_get_possible_groups_cube_and_mask(self):
        prev_group_indices_2d = np.array([[2, 3], [2, 4], [3, 4], [3, 5]])
        group_size = 3
        current_mask = np.array([0, 0, 1, 1, 1, 1])
        expected_groups_cube = np.array([[[0, 0, 0],
                                          [0, 0, 1],
                                          [0, 0, 0]],
                                         [[0, 0, 0],
                                          [0, 0, 0],
                                          [0, 0, 0]],
                                         [[0, 0, 0],
                                          [0, 0, 0],
                                          [0, 0, 0]]])
        expected_mask = np.array([0, 0, 1, 1, 1, 0])
        poss_groups, mask = get_possible_groups_cube_and_mask(prev_group_indices_2d,
                                                              group_size,
                                                              current_mask)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            result = session.run(poss_groups)
        self.assertTrue(np.array_equal(result, expected_groups_cube))
        self.assertTrue(np.array_equal(mask, expected_mask))
