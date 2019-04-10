import numpy as np
from PIL import Image, ImageOps

def get_sparse_disp(disp_L, erase_ratio=0.7):
    '''set some non zero disparity to zero'''
    w, h = disp_L.shape
    valid_ind = np.where(disp_L!=0)
    valid_ind = np.array(zip(valid_ind[0], valid_ind[1]))
    # how many points to be set to 0
    sample_num = int(len(valid_ind) * erase_ratio)
    #print('origin: ', np.sum(disp_L!=0)/float(w*h))
    ind = np.random.choice(range(0, len(valid_ind)), replace=False,size=sample_num)
    sample_ind = valid_ind[ind]
    sparse_disp_L = np.copy(disp_L)
    y_ind = np.squeeze(sample_ind[:,0])
    x_ind = np.squeeze(sample_ind[:,1])
    sparse_disp_L[y_ind,x_ind] = 0
    #print('sparse:', np.sum(sparse_disp_L!=0)/float(w*h))
    # sparse_disp_L = np.expand_dims(sparse_disp_L, axis=0)
    return sparse_disp_L

def crop_np_matrix(dataL, target_h, target_w):
    w, h = dataL.shape
    # print('origin size: ', dataL.shape)
    dataL = np.pad(dataL, ((max(target_h-h, 0), 0), (max(target_w-w, 0), 0)), 'constant', constant_values=0)
    # print('after padding: ', dataL.shape)
    dataL = dataL[max(dataL.shape[0]-target_h, 0):dataL.shape[0], max(dataL.shape[1]-target_w, 0):dataL.shape[1]]
    # print('after crop: ', dataL.shape)
    return dataL
