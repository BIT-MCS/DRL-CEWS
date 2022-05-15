import torch
import numpy as np

"""
Create a normal used function lib
created by Yinuo Zhao
"""


def self_flatten(inputs):
    """
    a function helps to flatten a tensor
    :param inputs: a tensor that needed to be flatten
    :return: a one dim tensor that has been flatten
    """
    dim = inputs.dim()
    total_len = 1
    for i in range(dim):
        total_len *= inputs.size()[i]
    return inputs.view(total_len)


def smooth(input_array, sm_len=-1):
    """
    make a np.array to be more smooth
    :param input_array: np.array that needed to be smoothed, can be one dim or two dim
    :param sm_len: np.array that has been smoothed
    :return:
    """
    output_array = np.zeros_like(input_array)
    left_sum = np.zeros_like(input_array)
    if input_array.ndim == 1:
        len1 = np.shape(input_array)[0]
        left_sum[0] = input_array[0]
        for i in range(1, len1):
            left_sum[i] = left_sum[i - 1] + input_array[i]
        for i in range(len1):
            if sm_len == -1:
                output_array[i] = left_sum[i] / (i + 1)
            else:
                left_index = max(0, i - sm_len - 1)
                right_index = min(i + sm_len, len1 - 1)
                output_array[i] = (left_sum[right_index] - left_sum[left_index]) / (right_index - left_index)
    if input_array.ndim == 2:
        len1 = np.shape(input_array)[0]
        len2 = np.shape(input_array)[1]
        for k in range(len1):
            left_sum[k][0] = input_array[k][0]
            for i in range(1, len2):
                left_sum[k][i] = left_sum[k][i - 1] + input_array[k][i]
        for k in range(len1):
            for i in range(len2):
                if len == -1:
                    output_array[k][i] = left_sum[k][i] / (i + 1)
                else:
                    left_index = max(0, i - sm_len - 1)
                    right_index = min(i + sm_len, len2 - 1)
                    # right_repeat_time = i + sm_len - right_index
                    # right_repeat = input_array[k][len2 - 1] if right_repeat_time > 0 else 0
                    # left_repeat_time = sm_len - i
                    # left_repeat = input_array[k][0] if left_repeat_time > 0 else 0
                    # if right_repeat_time > 0:
                    #     print(i, ' ', sm_len, ' ', right_index)
                    #     print(right_repeat_time, ' ', right_repeat)
                    output_array[k][i] = (left_sum[k][right_index] - left_sum[k][left_index]) / (
                            right_index - left_index)
    # print(output_array)
    # print(np.shape(output_array))
    return output_array
