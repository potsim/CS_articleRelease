import time
import numpy as np
from functools import partial
import multiprocessing

from scipy.sparse.linalg import LinearOperator

from spgl1 import spg_bpdn, spgl1



def findSolution_spgBPDN(Theta, y, tau=0.0, sigma=0.00000005, x0=None, iter_lim=100, verbosity=2, iscomplex=False, opt_tol=1e-4, m_ord_limit=None ):
    # s_est, resid, grad, info = spg_bpdn(Theta, y, sigma, iter_lim=iter_lim, verbosity=verbosity)
    s_est, resid, grad, info = spgl1(Theta, y, tau=tau, sigma=sigma, x0=x0, iter_lim=iter_lim, 
                                            verbosity=verbosity, iscomplex=iscomplex, opt_tol=opt_tol, m_ord_limit=m_ord_limit)
    if (info["stat"] in [1,2,3] ):
        print("    Found a solution in {} iterations. sol type:{}.".format(info["niters"], info["stat"] ))
    else:
        print("    WARNING !!!!!! BPDN EXIT not a root. stat:{}, niters:{}".format(info["stat"],info["niters"]))

    return {"s_est":s_est, "resid":resid, "grad":grad, "info":info }
    


def initThetaM(ori_vec_len, selected_indexes, toRAM=False, type='FFT', use_multiprocessing = True, nbThreads = 4 ):
    if use_multiprocessing == False:
        nbThreads =1

    if(type=='FFT'):
        start_time = time.time()
        if toRAM:
            #compute the entire Theta Matrix
            print('   The entire Theta matrix will stand in memory')
            print('   Creating the matrix... ',end ='',flush=True)      
            Theta = np.zeros((len(selected_indexes),ori_vec_len))
            
            # faster using the cos function (x10)
            cos_f = np.arange(ori_vec_len) * (2*np.pi/ ori_vec_len)
            for i, sample_instance in enumerate(selected_indexes):
                Theta[i,:] = np.cos(cos_f*sample_instance)
            # for i, sample_instance in enumerate(selected_indexes):
            #     tmp_vec =  np.zeros(ori_vec_len)
            #     tmp_vec[sample_instance] = 1
            #     Theta[i,:] = np.real(np.fft.fft(tmp_vec))
            #     #v2, the second version is faster (~x2). However, it requires x2 in memory space
            #     # Theta[i,sample_instance]=1
            # # Theta = np.real(scipyfft(Theta,overwrite_x=True, workers=4))

        else:
            #Use linear operators
            print('   The Theta matrix won be computed. Using linear operators')
            print('   Creating the linear Operator... ',end ='',flush=True)
            Theta = LinearOperator((len(selected_indexes),ori_vec_len), matvec=partial(partialFourier_matvec_mt, nbThreads, selected_indexes,ori_vec_len), 
                        rmatvec=partial(partialFourier_rmatvec_mt, nbThreads, selected_indexes,ori_vec_len))
        print('Done in {:.3e}s!'.format(time.time()-start_time))

    if(type=='FFTcomplex'):
        start_time = time.time()
        if toRAM:
            #compute the entire Theta Matrix
            print('   The entire Theta matrix will stand in memory')
            print('   Creating the matrix... ',end ='',flush=True)      
            Theta = np.zeros((len(selected_indexes),ori_vec_len),dtype=complex)
            
            # faster using the cos function (x10)
            cos_f = np.arange(ori_vec_len) * (2*np.pi/ ori_vec_len)
            for i, sample_instance in enumerate(selected_indexes):
                Theta[i,:] = np.cos(cos_f*sample_instance) - 1j*np.sin(cos_f*sample_instance)

        else:
            #Use linear operators
            print('   The Theta matrix won be computed. Using linear operators')
            print('   Creating the linear Operator... ',end ='',flush=True)
            Theta = LinearOperator((len(selected_indexes),ori_vec_len), matvec=partial(partialFourier_matvec_mt_complex, nbThreads, selected_indexes,ori_vec_len), 
                        rmatvec=partial(partialFourier_rmatvec_mt_complex, nbThreads, selected_indexes,ori_vec_len))
        print('Done in {:.3e}s!'.format(time.time()-start_time))

    return Theta


# # function used to build the linear operator
# def partialFourier_matvec(idx,n,x):
#     """
#     Matrix to vector operation Ax (or Theta s)
#     Theta is the reduced fft matrix, s is the sparse spectrum
#     Theta is of dimension (len(idx),n)
#     s is of dimension (n)
#     output is of dimension (len(idx))
#     Multiply Theta row with s column, sum points -> give on element. Repeate to get the entire vector x
#     """
    
#     vec = np.zeros(int(len(idx)))
#     #---v1--- 10 times slower
#     # for i, sample_instance in enumerate(idx):
#     #     p_vec = np.zeros(n)
#     #     p_vec[sample_instance] = 1
#     #     vec[i] = np.sum(np.real(np.fft.fft(p_vec))*x)
#     #---v1---
#     cos_f = np.arange(n) * (2*np.pi/ n)
#     for i, sample_instance in enumerate(idx):
#         vec[i] = np.sum(np.cos(cos_f*sample_instance)*x)   

#     return vec

# # function used to build the linear operator the function resolve 
# def partialFourier_rmatvec(idx,n,x):
#     """
#     Matrix to vector operation A^H * x (or Theta^H * y)
#     Theta^H is the reduced fft matrix conjugate transposed, y is the sub sampled igm
#     Theta^H is of dimension (n,len(idx))
#     y is of dimension (len(idx))
#     output is of dimension (n)
#     Multiply Theta^H row with y column (column of y is one element), give a vector dimension(n)
#     Repeate for all rows and sum all output vector of dimension (n)
#     """
#     vec = np.zeros(n)
#     #----v1 ---10 times slower
#     # for i, sample_instance in enumerate(idx):
#     #     if x[i]==0: #speedup
#     #         continue
#     #     p_vec = np.zeros(n)
#     #     p_vec[sample_instance] = 1
#     #     vec = vec + np.real(np.fft.fft(p_vec))*x[i]
#     #----v1 ---
#     cos_f = np.arange(n) * (2*np.pi/ n)
#     for i, sample_instance in enumerate(idx):
#         if x[i]==0: #speedup
#             continue
#         vec = vec + np.cos(cos_f*sample_instance)*x[i]   

#     return vec

    # function used to build the linear operator
def partialFourier_rmatvec_mt(nbThreads,idx,n,x): #multithreaded version
    # merge array
    merge_array = np.stack((x,idx), axis=1)

    if nbThreads > 1:
        # split operation in number of thread
        split_list  = np.array_split(merge_array, nbThreads, axis=0)

        pool = multiprocessing.Pool(nbThreads)
        vec_list = pool.map(partial(partialFourier_rmatvec, n), split_list)
        vec = np.sum(vec_list,axis=0)
    else:
        vec = partialFourier_rmatvec(n,merge_array)
    return vec

def partialFourier_rmatvec(n,x_idx): #x_idx is a two-column array [x, idx], patch to ease the multithreaded implementation
    """
    Matrix to vector operation A^H * x (or Theta^H * y)
    Theta^H is the reduced fft matrix conjugate transposed, y is the sub sampled igm
    Theta^H is of dimension (n,len(idx))
    y is of dimension (len(idx))
    output is of dimension (n)
    Multiply Theta^H row with y column (column of y is one element), give a vector dimension(n)
    Repeate for all rows and sum all output vector of dimension (n)
    """
    vec = np.zeros(n)
    cos_f = np.arange(n) * (2*np.pi/ n)
    for x_, sample_instance in x_idx:
        if x_==0: #speedup
            continue
        vec = vec + np.cos(cos_f*sample_instance)*x_   
    return vec


    # function used to build the linear operator
def partialFourier_matvec_mt(nbThreads,idx,n,x): #multithreaded version
    if nbThreads > 1:
        # split operation in number of thread
        idx_list = np.array_split(idx, nbThreads)

        pool = multiprocessing.Pool(nbThreads)
        vec_list = pool.map(partial(partialFourier_matvec, x,n), idx_list)
        vec = np.concatenate(vec_list, axis=0)
    else:
        vec = partialFourier_matvec(x,n,idx)
    return vec

def partialFourier_matvec(x,n,idx):
    """
    Matrix to vector operation Ax (or Theta s)
    Theta is the reduced fft matrix, s is the sparse spectrum
    Theta is of dimension (len(idx),n)
    s is of dimension (n)
    output is of dimension (len(idx))
    Multiply Theta row with s column, sum points -> give on element. Repeate to get the entire vector x
    """
    vec = np.zeros(int(len(idx)))
    cos_f = np.arange(n) * (2*np.pi/ n)
    for i, sample_instance in enumerate(idx):
        vec[i] = np.sum(np.cos(cos_f*sample_instance)*x)   
    return vec


#debug matvec et rmatvec
# T = np.array(([0,0,0,0,1,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,1,0]))
# Tfft = np.zeros(T.shape)
# for i in range(T.shape[0]):
#     Tfft[i,:] = np.real(np.fft.fft(T[i,:]))
# print(Tfft)
# LO=aslinearoperator(Tfft)
# LO.matvec(np.array([0,0,0,0,1,0,0]))
# LO.rmatvec(np.array([1,1,0]))

def partialFourier_rmatvec_mt_complex(nbThreads,idx,n,x): #multithreaded version
    # merge array
    merge_array = np.stack((x,idx), axis=1)

    if nbThreads > 1:
        # split operation in number of thread
        split_list  = np.array_split(merge_array, nbThreads, axis=0)

        pool = multiprocessing.Pool(nbThreads)
        vec_list = pool.map(partial(partialFourier_rmatvec_complex, n), split_list)
        vec = np.sum(vec_list,axis=0)
    else:
        vec = partialFourier_rmatvec_complex(n,merge_array)
    return vec

def partialFourier_rmatvec_complex(n,x_idx): #x_idx is a two-column array [x, idx], patch to ease the multithreaded implementation
    """
    Matrix to vector operation A^H * x (or Theta^H * y)
    Theta^H is the reduced fft matrix conjugate transposed, y is the sub sampled igm
    Theta^H is of dimension (n,len(idx))
    y is of dimension (len(idx))
    output is of dimension (n)
    Multiply Theta^H row with y column (column of y is one element), give a vector dimension(n)
    Repeate for all rows and sum all output vector of dimension (n)
    """
#     vec = np.zeros(n)
    vec = np.zeros(n, dtype=complex)
    cos_f = np.arange(n) * (2*np.pi/ n)
    for x_, sample_instance in x_idx:
        if x_==0: #speedup
            continue
#         vec = vec + np.cos(cos_f*sample_instance)*x_  
        vec = vec + ( ( np.cos(cos_f*sample_instance)+ 1j*np.sin(cos_f*sample_instance) ) *x_) #watch out complex conjugate
    return vec


    # function used to build the linear operator
def partialFourier_matvec_mt_complex(nbThreads,idx,n,x): #multithreaded version
    if nbThreads > 1:
        # split operation in number of thread
        idx_list = np.array_split(idx, nbThreads)

        pool = multiprocessing.Pool(nbThreads)
        vec_list = pool.map(partial(partialFourier_matvec_complex, x,n), idx_list)
        vec = np.concatenate(vec_list, axis=0)
    else:
        vec = partialFourier_matvec_complex(x,n,idx)
    return vec

def partialFourier_matvec_complex(x,n,idx):
    """
    Matrix to vector operation Ax (or Theta s)
    Theta is the reduced fft matrix, s is the sparse spectrum
    Theta is of dimension (len(idx),n)
    s is of dimension (n)
    output is of dimension (len(idx))
    Multiply Theta row with s column, sum points -> give on element. Repeate to get the entire vector x
    """
#     vec = np.zeros(int(len(idx)))
    vec = np.zeros(int(len(idx)), dtype=complex)
    cos_f = np.arange(n) * (2*np.pi/ n)
    for i, sample_instance in enumerate(idx):
#         vec[i] = np.sum(np.cos(cos_f*sample_instance)*x)
        vec[i] = np.sum( (np.cos(cos_f*sample_instance)- 1j*np.sin(cos_f*sample_instance) ) *x)  
    return vec
