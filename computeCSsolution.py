import os
import time
import argparse
import pickle
import warnings
from pathlib import Path
import multiprocessing
from functools import partial
import glob
import re
import gc

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt

from APOvsCSmain import IgmsCharac
from CSwithSPGL1 import findSolution_spgBPDN, initThetaM

from demodTools import getTriggerWindows, demodSignal

SPEED_OF_LIGHT = 299792458 # (m/s)

use_multiprocessing = True
nbThreads = 4 
# maxMemoryThetaMatrixGB = 30.0

filename_saveresults = "cs_solutions"
processVerbose = 0 # 0-nothingm, 1-save figures in folder, 2 slow -  a lot of figures live



#demodulation parameters
afterTriggerMaskPoints  = 12
beforeTriggerMaskPoints = 12
fIF                     = 50.0e3  # Intermediate frequency; Hz
fS = 216e3 # ADC Sample frequency; Hz

#spgl1 parameter
opt_tol = 1.0e-4
spgl1verbosity = 0
iter_lim = 200



def flip_CSgrid(CS_est):
    if (len(CS_est) % 2) == 0:    
        return np.flipud(np.append(CS_est[1:], CS_est[0]))
    else:
        return np.flipud(CS_est)

def split_baselineSamples(ModelIGM, x_opt_axis):
    idx_baseline = np.where(ModelIGM.apodize(ModelIGM.baseline_mask) == False)[0]
    CS_x_opt_axis_noBL_idx = []
    CS_x_opt_axis_BL_idx = []
    samples_axis_noBL_idx = []
    samples_BL_idx = []
    for i, val in enumerate(x_opt_axis):
        idx = (np.abs(ModelIGM.APO_x_opt_axis - val)).argmin()
            
        if idx not in idx_baseline:
            CS_x_opt_axis_noBL_idx.append(idx)
            samples_axis_noBL_idx.append(i)
        else:
            samples_BL_idx.append(i)
            CS_x_opt_axis_BL_idx.append(idx)

    CS_x_opt_axis_noBL_idx = np.array(CS_x_opt_axis_noBL_idx)
    CS_x_opt_axis_BL_idx = np.array(CS_x_opt_axis_BL_idx)    
    return CS_x_opt_axis_noBL_idx, CS_x_opt_axis_BL_idx, samples_axis_noBL_idx, samples_BL_idx


def loadIGMcharacteristics(path_fname):
    with open(path_fname,'rb') as f:
        igmpool_obj = pickle.load(f)   
    igmpool = igmpool_obj['igmpool']
    # cellUsedIgmpool = igmpool_obj['cell']
    # spc_parameters = igmpool_obj['spc_parameters']
    ModelIGM  = IgmsCharac(igmpool_obj)
    return ModelIGM

def processCSfiles(processingfolder, experiment_files_sig, cs_pattern_fname, pickle_fname,verbose=1):

    nb_of_files = len(experiment_files_sig)
    print('{} measurements to process'.format(nb_of_files))

    #load pickle file - IGMcharacteristics
    igmpool_file = "igmSCPool_July15_30fs.pickle"
    ModelIGM = loadIGMcharacteristics(Path(processingfolder.parent,pickle_fname))

    APO_factor = 0.0
    re_s = re.search('cs([0-9]{2})_([0-9]{2})kpnts', processingfolder.name)
    CS_nbSamples = int(re_s[1])*1000 + int(re_s[2])*10

    ModelIGM.setApodizationAndCSParams(APO_factor,CS_nbSamples)
    if verbose == 2:
        print('IGM characteristics: ')
        print('  APO factor is {}'.format(APO_factor))
        print('  IGM length after APO is {} pts. Number of samples for CS is {} pts. The compressive factor is : {}'.format(ModelIGM.APO_igm_len,
                                                                                                                 CS_nbSamples,
                                                                                                                 ModelIGM.APO_igm_len/CS_nbSamples))

    #load CS_pattern
    df = pd.read_csv(Path(processingfolder.parent,cs_pattern_fname), header=None, engine='python')
    x_opt_axis = df[0].to_numpy()
    nbSamples = len(x_opt_axis)
    if verbose == 2:
        print('  Sampling pattern length, {} pts'.format(nbSamples))


    # Process files
    theta = None
    for i, exp_file in enumerate(experiment_files_sig):
        file_id=re.search(exp_file.parent.name +'([0-9]{4})sig.txt', exp_file.name)[1]
        print(' Processing file id {}'.format(file_id),end="")
        print('. File {} of {}'.format(i,nb_of_files))
        sig = np.loadtxt(exp_file)
        trig_file_name = exp_file.name[0:-7] + 'trig.txt'
        trig = np.loadtxt(Path(exp_file.parent,trig_file_name))
        if not len(sig) == len(trig):
            print('  !!Warning, Trig and sig files dont fit. Processing skipped')
            continue

        startPoints, endPoints = getTriggerWindows(trig,beforeTriggerMaskPoints,afterTriggerMaskPoints, nbSamples)
        baseBandIGM = demodSignal(sig, startPoints, endPoints, fS, fIF)

        if verbose == 2:
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(np.real(baseBandIGM))
            ax.plot(np.imag(baseBandIGM))
            ax.plot(np.abs(baseBandIGM),'-k')
            plt.show()
            plt.close(fig)


        #check point -1- Max signal at 0 OPD
        # is igm maximum at 0.0 OPD?
        # Big problem if not the case, the grid has changed
        # Won't be possible to reuse the theta matrix
        if not (x_opt_axis[np.argmax(np.abs(baseBandIGM))] == 0.0):
            x_opt_axis = x_opt_axis - x_opt_axis[np.argmax(np.abs(baseBandIGM))]
            if not (x_opt_axis[np.argmax(np.abs(baseBandIGM))] == 0.0):
                warnings.warn('IGM center not at 0 OPD, check point failed')
                continue


        #check point -2-  Equidistant sampling grid
        step_size = ModelIGM.APO_x_opt_axis[1] - ModelIGM.APO_x_opt_axis[0]
        # print('Step size is {} s'.format(step_size))
        new_CS_x_opt_axis = [ModelIGM.APO_x_opt_axis[(np.abs(ModelIGM.APO_x_opt_axis - val)).argmin()] for val in x_opt_axis]
        if not (new_CS_x_opt_axis-x_opt_axis < (step_size / 100)).all():
            warnings.warn('Equidistance OPD mapping, check point failed')
            continue

        if verbose == 2:
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(new_CS_x_opt_axis-x_opt_axis)
            ax.set_ylabel('Error [s]')
            ax.set_xlabel('Sample number')
            ax.set_title('Step scan sampling jitter \n Samples OPD vs the closest point on the equidistant grid')
            plt.show()
            plt.close(fig)


        CS_x_opt_axis_noBL_idx, CS_x_opt_axis_BL_idx, samples_axis_noBL_idx, samples_BL_idx = split_baselineSamples(ModelIGM, x_opt_axis)
        samples_x_opt_axis_noBL = ModelIGM.APO_x_opt_axis[CS_x_opt_axis_noBL_idx]
        samples_x_opt_axis_BL = ModelIGM.APO_x_opt_axis[CS_x_opt_axis_BL_idx]
        CS_igm_noBL = baseBandIGM[samples_axis_noBL_idx]
        CS_igm_BL = baseBandIGM[samples_BL_idx]
        if verbose >= 1:
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(samples_x_opt_axis_noBL,np.abs(CS_igm_noBL),'bx' ,label='Samples used for CS')
            ax.plot(samples_x_opt_axis_BL,np.abs(CS_igm_BL),'-xk' ,label='Samples used for Baseline')
            ax.set_xlabel('OPD [s]')
            ax.set_title('Baseband IGM - analytical signal')
            ax.legend()
            if verbose == 1:
                fig.savefig(Path(Path(processingfolder,filename_saveresults), 'fig_samples_' + file_id + '.png'))
            else:
                plt.show()
            plt.close(fig)

        # ##### Patch to change the number of points
        # if i == 0:
        #     DS_samples = 50 #940,625,525,425,300, 200, 100,50,10
        #     sel_ind = np.random.choice(len(CS_x_opt_axis_noBL_idx),size=DS_samples,replace=False)

        #     # with open(Path(Path('C:/Users/SimPot/Documents/projets/NIST/20220906StepScan/cs10_00kpnts'),'list__selected_results_with_index.pickle'), 'rb') as f:
        #     #     results_list = pickle.load(f)
        #     # res_= results_list['selected_list_of_results'][2]
        #     # DS_samples = res_['DS_samples']
        #     # sel_ind = res_['sel_ind']

        #     print('New number of points for CS is: {}'.format(DS_samples))

             

        # CS_x_opt_axis_noBL_idx = CS_x_opt_axis_noBL_idx[sel_ind]
        # samples_axis_noBL_idx = np.array(samples_axis_noBL_idx)[sel_ind]

        # samples_x_opt_axis_noBL = ModelIGM.APO_x_opt_axis[CS_x_opt_axis_noBL_idx]
        # samples_x_opt_axis_BL = ModelIGM.APO_x_opt_axis[CS_x_opt_axis_BL_idx]
        # CS_igm_noBL = baseBandIGM[samples_axis_noBL_idx]
        # CS_igm_BL = baseBandIGM[samples_BL_idx]
        # #############################

        selected_indexes = CS_x_opt_axis_noBL_idx
        ori_vec_len = ModelIGM.APO_igm_len
        samples = -1.0*CS_igm_noBL


        if verbose == 2:
            ThetaSizeGB = 8*len(selected_indexes) /1000 * ModelIGM.APO_igm_len/1000 * 1e-3 *2 #[GB]
            print('  Apodized igms have {} pts. With CS, the compressive_factor is {:.4}.'.format( 
                        ModelIGM.APO_igm_len, ModelIGM.APO_igm_len/len(selected_indexes)))
            print('  Theta matrix would take {} GB in memory'.format(ThetaSizeGB))




        if theta is None:
            theta = initThetaM(ori_vec_len, selected_indexes, toRAM=True, type='FFTcomplex',
                                use_multiprocessing = use_multiprocessing, nbThreads = nbThreads )


        CS_solution = findSolution_spgBPDN(theta,samples,iter_lim=iter_lim,verbosity=spgl1verbosity,iscomplex=True, opt_tol=opt_tol)
        s_est_tmp = np.fft.fftshift(CS_solution['s_est']) * ModelIGM.APO_igm_len
        s_est = flip_CSgrid(s_est_tmp)

        igm_est = np.fft.ifft(np.fft.ifftshift(s_est))
        igm_baseline_mask = ModelIGM.apodize(ModelIGM.baseline_mask)
        tmp_vec = np.zeros(len(igm_est),dtype=np.complex_)
        tmp_vec[CS_x_opt_axis_BL_idx] = baseBandIGM[samples_BL_idx]
        new_igm_est = -1.0*(igm_est*igm_baseline_mask + -1.0*tmp_vec)

        if verbose >= 1:
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(np.abs(s_est),'-g',label='Estimated spc - raw axis')
            transmit_spc_est = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(new_igm_est))))
            ax.plot(transmit_spc_est,label='Reconstructed transmittance')
            ax.legend()
            if verbose == 1:
                fig.savefig(Path(Path(processingfolder,filename_saveresults), 'fig_transmittance_' + file_id + '.png'))
            else:
                plt.show()
            plt.close(fig)

        #save results
        # first save igm axis if doesnt exist
        if not os.path.exists(Path(Path(processingfolder,filename_saveresults),'CS_x_opt_axis.txt')):
            np.savetxt(Path(Path(processingfolder,filename_saveresults),'CS_x_opt_axis.txt'), ModelIGM.APO_x_opt_axis, fmt='%.18e')
        # next, save cs solution
        np.savetxt(Path(Path(processingfolder,filename_saveresults),exp_file.name[0:-7] + 'CSigm.txt'), new_igm_est, fmt='%.18e %.18e')
        np.savetxt(Path(Path(processingfolder,filename_saveresults),exp_file.name[0:-7] + 'CSraw.txt'), CS_solution['s_est'] * ModelIGM.APO_igm_len, fmt='%.18e %.18e')
        
        
        print(' Processing file id {} completed'.format(re.search(exp_file.parent.name +'([0-9]{4})sig.txt', exp_file.name)[1]))

        gc.collect()

    return

########
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-datapath", "--folderpath", type=str, help="working folder", required=False)
    parser.add_argument("-folder", "--foldername", type=str, help="folder to process", required=False)
    parser.add_argument("-cs_pattern", "--cs_patternfname", type=str, help="csv filename", required=False)
    parser.add_argument("-pickle", "--pickle_fname", type=str, help="pickle filename", required=False)
    args = parser.parse_args()

    workfolderpath = Path(args.folderpath)
    folder2process = args.foldername
    cs_pattern_fname = args.cs_patternfname
    pickle_fname = args.pickle_fname

    processingfolder = Path(workfolderpath,folder2process) 
    # Does the folder to process exist?
    if not os.path.exists(processingfolder):
        raise Exception('The folder ' + folder2process + ' already exists in ' + str(workfolderpath) )

    # Does the cs_pattern file exist?
    if not os.path.exists(Path(workfolderpath,cs_pattern_fname)):
        raise Exception('CS_pattern file name ' + cs_pattern_fname + ' does not exist in ' + str(workfolderpath) )

    # Does the pickle file exist?
    if not os.path.exists(Path(workfolderpath, pickle_fname)):
        raise Exception('CS_pattern file name ' + pickle_fname + ' does not exist in ' + str(workfolderpath) )

    # generate the list to process
    experiment_files = list(processingfolder.glob(folder2process + "*sig.txt"))
    # sig_ids = [re.search(folder2process +'([0-9]{4})sig.txt', fname.name)[1] for fname in experiment_files ]
    
    ### Make results dir, if exists, remove from the list the files already processed
    if os.path.exists(Path(processingfolder,filename_saveresults)):
        print('The folder {} already exist. Processing will be resume. No files will be overwritten.'.format(filename_saveresults))
        solution_files = list(Path(processingfolder,filename_saveresults).glob(folder2process + "*CSigm.txt"))
        solution_ids = [re.search(folder2process +'([0-9]{4})CSigm.txt', fname.name)[1] for fname in solution_files ]
        for fname in experiment_files.copy():
            file_id = re.search(folder2process +'([0-9]{4})sig.txt', fname.name)[1]
            if file_id in solution_ids:
                experiment_files.remove(Path(processingfolder,folder2process + file_id + 'sig.txt'))
    else:
        Path(processingfolder,filename_saveresults).mkdir()

    
    processCSfiles(processingfolder, experiment_files, cs_pattern_fname, pickle_fname,verbose=processVerbose)

    print(' Processing of folder {} completed!!'.format(processingfolder))