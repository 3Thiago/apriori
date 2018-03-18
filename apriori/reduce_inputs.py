import tensorflow as tf
import numpy as np


def get_sparse_mask_matrix(current_mask):
    def create_indices(mask):
        positive_el_number = 0
        _indices = []
        for i, v in enumerate(mask):
            if v == 1:
                _indices.append([i, positive_el_number])
                positive_el_number += 1
        return np.array(_indices, dtype=np.int32)

    indices = tf.py_func(create_indices, [current_mask], tf.int32)
    number_of_remaining_elements = tf.cast(tf.divide(tf.size(indices), 2), tf.int32)
    values = tf.py_func(lambda _indices: np.ones(len(_indices), dtype=np.int32),
                        [indices], tf.int32)
    shape = tf.stack([tf.size(current_mask), number_of_remaining_elements])
    mask_matrix = tf.sparse_to_dense(sparse_indices=indices, sparse_values=values, output_shape=shape)
    return mask_matrix


def reduce_input_columns_with_current_mask(input_rows, current_mask):
    """
    Construct a sparse matrix containing the input_rows with only the 'allowed' columns
    """
    inputs_mask_matrix = get_sparse_mask_matrix(current_mask)
    return tf.matmul(input_rows, inputs_mask_matrix)
