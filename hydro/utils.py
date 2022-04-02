import torch
import consts as CONSTS

def normalize_01_data(data):
    '''
    linearly scale data to [0-1] using constant min/max values from a priori knowledge
    '''
    for col_ind in range(data.shape[1]):
        min_bound, max_bound = CONSTS.BOUNDARIES[col_ind]
        data[:,col_ind] = (data[:,col_ind] - min_bound) / (max_bound - min_bound)
    return data

def normalize_log_output(output, from_original_data=True):
    '''
    logarithmic
    :param from_original_data: set to False to denormalize data into 0~188
    '''
    if not from_original_data:
        min_bound, max_bound = CONSTS.BOUNDARIES[-1]
        output = output * (max_bound - min_bound) + min_bound
    output = output.clip(1.0)
    output = torch.log(output)
    log_max_bound, log_min_bound = CONSTS.LOG_BOUNDARIES
    output = (output - log_min_bound) / (log_max_bound - log_min_bound)
    return output

def denormalize_output(output):
    '''
    de-logarithmic
    '''
    # output = torch.pow(output, 4.0)
    min_bound, max_bound = CONSTS.BOUNDARIES[-1]
    output = output * (max_bound - min_bound) + min_bound
    return output


if __name__ == '__main__':
    import numpy as np
    data = torch.from_numpy(np.load('norm_data.npy'))
    min_bound, max_bound = CONSTS.BOUNDARIES[-1]
    test_outputs = data[30:40,-1]
    norm_outp = normalize_log_output(test_outputs)
    print(norm_outp)