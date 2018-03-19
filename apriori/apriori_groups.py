import tensorflow as tf
from reduce_inputs import reduce_input_columns_with_current_mask
from possible_groups_mask import get_possible_groups_cube_and_mask


def get_inputs_t_multiplied_by_transpose_permutations(input_groups, group_size):
    # We have the masked vectorised input rows as a N+1 dimensional tensor
    # Now multiply by N - 1 tensors whose axes have been permuted (on all but 0th axis)
    cross_multiplied_input_groups = input_groups
    dims_list = list(range(1, group_size + 1))
    num_dims = len(dims_list)
    for dim in dims_list[:-1]:
        perm = [0] + [((i + dim + num_dims - 1) % num_dims) + 1 for i in dims_list]
        print(f"perm{perm}")
#         perm = [0 if i == -1 else perm[i] for i in range(-1, len(perm))]
        input_groups_perm = tf.transpose(input_groups, perm=perm)
        # TODO: Test broadcasting always works when using same number of input rows as unique elements
        cross_multiplied_input_groups = tf.multiply(cross_multiplied_input_groups, input_groups_perm)
    return cross_multiplied_input_groups


def get_next_mask_and_groups(gc):
    """
    ----- The input elements axis becomes the new dimension of the
    """

    print(f"Get_next_mask_and_groups for group size {gc['current_N']}")

    inputs_reduced = reduce_input_columns_with_current_mask(gc["input_rows"],
                                                            gc["curr_bin_mask"])
    print(inputs_reduced)
    possible_groups_cube, mask = get_possible_groups_cube_and_mask(gc["prev_group_indices"],
                                                                   gc["current_N"],
                                                                   gc["curr_bin_mask"])

    shape = [gc["num_input_rows"]] + [gc["num_remaining_els"] for _ in range(0, gc["current_N"])]
    number_of_elements = gc["num_remaining_els"] ** (gc["current_N"] - 1)
    expanded_inputs_t = tf.reshape(tf.tile(inputs_reduced, [1, number_of_elements]), shape)

    transpose_multiplied_input_groups = get_inputs_t_multiplied_by_transpose_permutations(expanded_inputs_t,
                                                                                          gc["current_N"])
    print(possible_groups_cube)
    input_groups_t = tf.multiply(transpose_multiplied_input_groups, possible_groups_cube)

    group_counts_t = tf.reduce_sum(input_groups_t, axis=0)

    frequent_groups_indices = tf.where(group_counts_t > 0)
    frequent_groups_counts = tf.gather_nd(group_counts_t, frequent_groups_indices)
    next_possible_groups_indices = tf.where(group_counts_t >= gc["min_support"])

    return frequent_groups_indices, frequent_groups_counts, next_possible_groups_indices, mask
