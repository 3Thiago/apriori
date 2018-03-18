import unittest
import tensorflow as tf
import numpy as np
from apriori.reduce_inputs import get_sparse_mask_matrix, reduce_input_columns_with_current_mask


class TestReduceInputs(unittest.TestCase):

    def test_get_sparse_mask_matrix(self):
        test_current_mask = tf.constant([1, 0, 1, 0, 1], dtype=tf.int32)
        expected_mask_matrix = np.array([[1, 0, 0],
                                         [0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0],
                                         [0, 0, 1]])
        mask = get_sparse_mask_matrix(test_current_mask)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            result = session.run(mask)

        self.assertTrue(np.array_equal(result, expected_mask_matrix))

    def test_reduce_input_columns_with_current_mask(self):
        test_inputs = tf.constant([[1, 2, 1, 2, 1],
                                   [1, 2, 0, 2, 1]])
        test_current_mask = tf.constant([1, 0, 1, 0, 1], dtype=tf.int32)
        expected_reduced_inputs = np.array([[1, 1, 1],
                                            [1, 0, 1]])

        reduced_inputs = reduce_input_columns_with_current_mask(test_inputs, test_current_mask)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            result = session.run(reduced_inputs)

        self.assertTrue(np.array_equal(result, expected_reduced_inputs))


