import os
import time
import argparse
import pickle
from pathlib import Path
import multiprocessing
from functools import partial
import numpy as np
import glob
import re
import gc

import yaml

import matplotlib.pyplot as plt

from CSwithSPGL1 import initThetaM, findSolution_spgBPDN

use_multiprocessing = True
nbThreads = 4 
maxMemoryThetaMatrixGB = 30.0

savedConfigFilename = "iparams.yaml"

SPGL1verbosity = -1

SPEED_OF_LIGHT = 299792458 # (m/s)


class IgmsCharac():
    def __init__(self, igmpool_obj):
        #self.igm = igmpool_obj['igmpool'][id]
        self.igm_len = len(igmpool_obj['igmpool'][0])
        self.molefraction = igmpool_obj['molefractions']
        self.baseline = igmpool_obj['baseline']
        self.id=id

        #Optical frequency axis
        self.f_axis_opt_wn = igmpool_obj['f_wavenumber']
        self.f_axis_opt_lambda = 1/(self.f_axis_opt_wn*100)
        self.f_opt_axis = SPEED_OF_LIGHT * (self.f_axis_opt_wn*100)
        self.f_opt_offset = self.f_opt_axis[0]
        self.f_opt_axis_offset = self.f_opt_axis - self.f_opt_offset
        self.f_res= self.f_opt_axis[1] - self.f_opt_axis[0]
        self.wn_res = self.f_axis_opt_wn[1] - self.f_axis_opt_wn[0]

        #RF axis
        if igmpool_obj['spc_parameters']['move2baseband']:
            if "basebandfinal_opt2RF" in igmpool_obj['spc_parameters']:
                self.opt2RF =  igmpool_obj['spc_parameters']['basebandfinal_opt2RF']
            else:
                self.opt2RF = igmpool_obj['opt2RF']

            self.analytical_igm = True
        else:
            self.opt2RF = igmpool_obj['spc_parameters']['opt2RF']
            self.analytical_igm = False

        self.f_RF_axis = self.f_opt_axis / self.opt2RF
        self.f_RF_offset = self.f_RF_axis[0]
        self.f_RF_axis_offset = self.f_RF_axis - self.f_RF_offset

        self.RF_carrier = self.f_RF_axis_offset[np.argmax(self.baseline)]
        # self.opt_carrier=self.f_opt_axis[np.argmax(self.baseline)]
        self.opt_carrier = (self.RF_carrier+self.f_RF_offset)*self.opt2RF
        
        #igm axis
        self.x_opt_axis = igmpool_obj['x_opt_axis']
        self.x_RF_axis = igmpool_obj['x_axis']
        self.delta_t = self.x_RF_axis[1] - self.x_RF_axis[0]

        if igmpool_obj['spc_parameters']['move2baseband']:
            self.f_RF_axis_offset = np.fft.fftshift(np.fft.fftfreq(self.igm_len,d=self.delta_t))
            

        #baseline
        self.baseline_mask = self.computeBaselineMask()
        # plt.figure()
        # plt.plot(self.x_opt_axis,self.baseline_mask)
        # plt.plot(self.x_opt_axis,igmpool_obj['igmpool'][0])
        # plt.show()
        

        #generate sampling PDF
        self.samplingPDF = self.generateSamplingPDF(igmpool_obj['igmpool'][0])
        # plt.figure()
        # plt.plot(self.x_opt_axis,np.abs(igmpool_obj['igmpool'][0]) / np.max(np.abs(igmpool_obj['igmpool'][0])))
        # plt.plot(self.x_opt_axis,self.samplingPDF/ np.max(self.samplingPDF),'-r')
        # plt.title("pdf shape")
        # plt.show()
        
    
    def computeBaselineMask(self,fudge_factor=4):
        ### identify FWTM: full width at tenth of maximum
        ind = np.where(self.baseline > 0.1*np.max(self.baseline))[0] 
        self.Optspc_BW = self.f_opt_axis_offset[ind[-1]] - self.f_opt_axis_offset[ind[0]]
        #print('Opt spectrum bandwidth is: {} Hz'.format(self.Optspc_BW))

        gaus_std  = self.Optspc_BW/4.29193
        self.Optigm_BW = 4.29193 * (1/(2*np.pi*gaus_std))
        #add a margin to the IGM central burst width
        self.Optigm_BW= self.Optigm_BW*2*fudge_factor

        #return bool window
        return ~((self.x_opt_axis >= -self.Optigm_BW/2) & (self.x_opt_axis<= self.Optigm_BW/2))

    def generateSamplingPDF(self, igm, kernel_size=1000):
        pdf = np.abs(igm)
        kernel = np.ones(kernel_size) / kernel_size

        ###Do this before and after smooth to avoid strange pdf (effect of the centerburst)
        #we don't want to sample the center point -> DC in the spectrum
        pdf[int(np.argmax(np.abs(igm)))] = 0
        # Avoid sampling points in the baseline
        pdf = pdf*self.baseline_mask

        #Smooth pdf
        pdf_smoothed = np.convolve(pdf, kernel, mode='same')
        # add DC to avoid problems with noiseless igms 
        pdf_smoothed = pdf_smoothed + np.max(pdf_smoothed)*0.005 # 0.5% of max amplitude 
        # #we don't want to sample the center point -> DC in the spectrum
        # pdf_smoothed[int(np.argmax(pdf))] = 0
        # Avoid sampling points in the baseline
        pdf_smoothed = pdf_smoothed*self.baseline_mask
        # sum needs to be == 1 for the random sampling function
        pdf_smoothed =  pdf_smoothed/ np.sum(pdf_smoothed)

        return pdf_smoothed

    
    def setApodizationAndCSParams(self,APO_factor, CS_nbSamples):
        self.APO_factor = APO_factor
        x_center_ind = np.where(self.x_opt_axis==0)[0][0]
        nb_half_cut = int(self.igm_len*(1-self.APO_factor) /2)

        self.start_ind = x_center_ind-nb_half_cut
        self.stop_ind = x_center_ind+nb_half_cut
        
        self.APO_x_opt_axis = self.x_opt_axis[self.start_ind:self.stop_ind]
        self.APO_igm_len = len(self.APO_x_opt_axis)

        if self.analytical_igm:
                self.APO_f_opt_axis = np.fft.fftfreq(self.APO_igm_len,d=(self.APO_x_opt_axis[1] - self.APO_x_opt_axis[0]))
                self.APO_f_opt_axis = np.fft.fftshift(self.APO_f_opt_axis)
                self.APO_f_opt_axis = self.APO_f_opt_axis + - self.APO_f_opt_axis[0] + self.f_opt_offset + self.RF_carrier + self.f_RF_offset
        else:
            self.APO_f_opt_axis = np.fft.rfftfreq(self.APO_igm_len,d=(self.APO_x_opt_axis[1] - self.APO_x_opt_axis[0])) + self.f_opt_offset 
        self.APO_f_res = self.APO_f_opt_axis[1]-self.APO_f_opt_axis[0]
        self.APO_wn_opt_axis = self.APO_f_opt_axis/(SPEED_OF_LIGHT*100)
        self.APO_wn_res = self.APO_f_res/(SPEED_OF_LIGHT*100)

        self.setCSParams(CS_nbSamples)
        return

    def setCSParams(self, CS_nbSamples):
        self.CS_nbSamples = CS_nbSamples
        apo_samplingPDF = self.apodize(self.samplingPDF)
        apo_samplingPDF = apo_samplingPDF/ np.sum(apo_samplingPDF)

        self.sampledIndexes = np.random.choice(self.APO_igm_len,size=self.CS_nbSamples,replace=False,p=apo_samplingPDF)
        return

    def apodize(self,vec):
        return vec[self.start_ind:self.stop_ind]

    def apodizeAndRandomSampling(self,igm):
        return self.apodize(igm)[self.sampledIndexes]
    
    def printIgmCharac(self):
        print('  ---Optical parametres---')
        print('  Optical carrier {:.5e} Hz'.format(self.opt_carrier))
        print('  Optical sampling period T is: {:.2e} s'.format( self.x_opt_axis[1] - self.x_opt_axis[0]))
        print('  Optical f_res is: {:.5e}'.format(self.f_res))
        print('  Baseline width is: {:.5e} s'.format( self.Optigm_BW ) )

        print('  ---RF parametres---')
        print('  Compression ratio opt2RF of {:.2e}'.format(self.opt2RF))
        print('  RF sampling period T is: {:.2e} s'.format( self.delta_t))
        print('  RF fs is: {:.3e}'.format(1/self.delta_t))
        print('  RF Igm length {} points, {:.5} s'.format(self.igm_len,self.igm_len*(self.x_RF_axis[1] - self.x_RF_axis[0])))
        return


def printIgmPoolSpecs(data_obj):
    print("Igm pool characteristics")
    print("  {} igms of {} points".format(len(data_obj['igmpool']), len(data_obj['igmpool'][0])))
    print("  Gas cell is {}, T:{}K, P:{}atm, path_length:{}cm, mole_Frac_range {} ".format(data_obj['cell']['db_name'], data_obj['cell']['ti_kelvin'],
            data_obj['cell']['p_atm'],data_obj['cell']['path_cm'],data_obj['spc_parameters']['molfrac_range']))
    if "noise" in data_obj['spc_parameters']:
        print("  Noise model: timing jitter {} rad RMS, phase noise {} rad RMS, additive noise IGM sigma {}".format(
            data_obj['spc_parameters']['noise']['timing_RMS_rad'],data_obj['spc_parameters']['noise']['phase_RMS_rad'],
        data_obj['spc_parameters']['noise']['additive_IGMsig'] ))
    else:
        print("  No noise model")

    if data_obj['spc_parameters']['move2baseband']:
        print("  Analytical IGM signal, shifted to baseband.")

    return


def startCSExperiments(data_obj,config):
    #print pool characteristics
    printIgmPoolSpecs(data_obj)

    #select igms
    ModelIGM  = IgmsCharac(data_obj)
    ModelIGM.printIgmCharac() #print igm characteristics
    igmIDlist = config['searchParams']['igm_IDS']
    print("Number of igms to process per experiment is {}.".format(len(igmIDlist)))

    #select an experiment
    experiment_files = list(Path(config['folderpath'],config['foldername']).glob("ToDo*.txt"))
    #force experiment to the easiest one
    # experiment_files= [experiment_files[0]]
    for experiment_file in experiment_files:
        # mark file as selected
        Path(experiment_file.parent,experiment_file.name[4:] )
        filepathname_proccesing = Path(experiment_file.parent, "Processing" + experiment_file.name[4:])
        os.rename(experiment_file, filepathname_proccesing ) # Rename : name Processing instead of ToDo
        # pattern matching
        filename_parsed = re.search('ToDo_APO_([0-9]{1}.[0-9]{3})__CS_([0-9]{4})k', experiment_file.stem)
        APOandCS = (float(filename_parsed[1]), int(float(filename_parsed[2])*1e3))
        print("Selected APO is {} and number of samples for CS is {}".format(APOandCS[0],APOandCS[1]))

        ### PROCESS
        imgpool = data_obj['igmpool']
        #Apodization and sampling
        ModelIGM.setApodizationAndCSParams(APOandCS[0],APOandCS[1])
        ### igm*(-1) to get absorption igms instead of transmittance (true since we do not sample ZPD)
        igm_list_apodized_sampled= [ModelIGM.apodizeAndRandomSampling(-1*imgpool[igm_ID]) for igm_ID in igmIDlist ] 
        
        #Matrix size
        ThetaSizeGB = 8*ModelIGM.CS_nbSamples /1000 * ModelIGM.APO_igm_len/1000 * 1e-3 #[GB]
        print('  Apodized igms have {} pts. With CS, the compressive_factor is {:.4}.'.format( 
            ModelIGM.APO_igm_len, ModelIGM.APO_igm_len/ModelIGM.CS_nbSamples))
        print('  Theta matrix would take {} GB in memory'.format(ThetaSizeGB))

        thetaM2Mem = False
        if(ThetaSizeGB <= maxMemoryThetaMatrixGB):
                thetaM2Mem = True

        # Use the same theta Matrix to save computational time
        # because of this, all processed igms in this APO and CS config have the same "random" sampling pattern
        print("  Using the same theta matrix for all igms with the same APO and CS config")
        theta = initThetaM(ModelIGM.APO_igm_len, ModelIGM.sampledIndexes, toRAM=thetaM2Mem, type='FFT',
                            use_multiprocessing = use_multiprocessing, nbThreads = nbThreads )
        
        start_time = time.time()
        spc_absorption_est_list = []
        print("   Processing IGM {} of {} ".format(1,len(igmIDlist)))
        spc_absorption_est_list.append(findSolution_spgBPDN(theta,igm_list_apodized_sampled[0],verbosity=SPGL1verbosity))
        tau = 0.0
        x0 = None
        for i, (igm_ID, igm) in enumerate(zip(igmIDlist[1:],igm_list_apodized_sampled[1:])):
            print("   Processing IGM {} of {} ".format(i+2,len(igmIDlist)))
            #start at the last know solution if a root was founded
            if tau == 0.0:
                if spc_absorption_est_list[i]["info"]["stat"] == 1:
                    tau = spc_absorption_est_list[i]["info"]["tau"]*0.5
                    x0 = spc_absorption_est_list[i]["s_est"]
                    print("   Using the last solution as a starting point for BPDN. Same one for the next igms")
            spc_absorption_est_list.append(findSolution_spgBPDN(theta,igm, tau=tau, x0=x0, verbosity=SPGL1verbosity)) # start CS from a known point
            
            # break
            # plt.figure()
            # plt.plot(-1.0*(np.abs(spc_absorption_est_list[0]/100)*ModelIGM.APO_igm_len - 1) )
            # plt.title("SPGL1 result")
            # plt.show()
        print("   All igms processed in {:.3e}s".format(time.time()-start_time))

        #save results
        #TODO perform baseline estimation and save it
        results_obj = {     "spc_absorption_est_list" : spc_absorption_est_list,
                            "igmIDlist" : igmIDlist,
                            "ModelIGM" : ModelIGM,
                        }
        with open(Path(experiment_file.parent,"results" + experiment_file.name[4:-4] + '.pickle'), 'wb') as f:
            pickle.dump(results_obj, f)
        print("   Results saved!!")
        
        #delete task file
        if os.path.isfile(filepathname_proccesing):
            os.remove(filepathname_proccesing)

        #release memory when looping
        del theta 
        gc.collect()


    return

def generateSearchGrid_txtFiles(config,max_pts):
    APO_factors = config['searchParams']['APO_factors']
    CS_sampling_pts = config['searchParams']['CS_sampling_pts']
    for APO_f in APO_factors:
        for CS_pts in CS_sampling_pts:
            if(CS_pts*1e3 > max_pts*(1-APO_f)): # impossible config, asking too many points
                continue
            tmp_filename = get_txtFile_str(APO_f,CS_pts)
            with open(Path(Path(config['folderpath'],config['foldername']),tmp_filename), mode='a'): pass

    return

def get_txtFile_str(APO_f,CS_pts):
    return "ToDo_APO_{:.3f}__CS_{:04d}k.txt".format(APO_f,CS_pts)


########
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-iparams", "--igmpool", type=str, help="yaml config file", required=False)
    parser.add_argument("-resume", "--folderPath", type=str, help="yaml config file", required=False)
    args = parser.parse_args()

    # Start new simulation or resume one
    if args.folderPath == None:
        # Open config file
        with open(Path(args.igmpool)) as f:
            config = yaml.safe_load(f)
        print("Start a new APO vs CS simulation")
        print(" Folder: {}".format(config['foldername']))
        print(" Folder location: {}".format(config['folderpath']))
        print(" Igmpool file used: {}".format(config['igmpool_file']))

        # Does the imgPool file exist?
        if not os.path.exists(Path(config['folderpath'],config['igmpool_file'])):
            raise Exception('The igmPool file ' + config['igmpool_file'] + ' does not exist in ' + config['folderpath'] )
        else:
            with open(Path(config['folderpath'],config['igmpool_file']),'rb') as f:
                data_obj = pickle.load(f)
                
        # Make Simulation dir
        if os.path.exists(Path(config['folderpath'],config['foldername'])):
            raise Exception('The folder ' + config['foldername'] + ' already exists in ' + config['folderpath'] )
        else:
            Path(config['folderpath'],config['foldername']).mkdir()
        
        #Save config file
        with open(Path(Path(config['folderpath'],config['foldername']), savedConfigFilename),'w') as f:
            yaml.dump(config, f)
        
        generateSearchGrid_txtFiles(config, len(data_obj['igmpool'][0]))

    else:
        #Does the folder exist
        if not Path(args.folderPath).is_dir():
            raise Exception("Try to resume experiment from a folder that does not exist")

        print("APO vs CS simulation resume from folder .... ")
        # Open config file
        if os.path.exists(Path(args.folderPath,savedConfigFilename)):
            with open(Path(args.folderPath,savedConfigFilename)) as f:
                config = yaml.safe_load(f)
        print(" Folder: {}".format(config['foldername']))
        print(" Folder location: {}".format(config['folderpath']))
        print(" Igmpool file used: {}".format(config['igmpool_file']))
        
        # Does the imgPool file exist?
        if not os.path.exists(Path(config['folderpath'],config['igmpool_file'])):
            raise Exception('The igmPool file ' + config['igmpool_file'] + ' does not exist in ' + config['folderpath'] )
        else:
            with open(Path(config['folderpath'],config['igmpool_file']),'rb') as f:
                data_obj = pickle.load(f)

    
    startCSExperiments(data_obj,config)

    print("Done!!")