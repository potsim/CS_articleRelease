
import numpy as np
from scipy import signal
import warnings

def demodSignal(sig, startPoints, endPoints, fs, fIF):
    # get the average number of points per window to limit the number of points
    dwell_times_tmp = endPoints - startPoints
    mean_nbpts = np.ceil(np.mean(dwell_times_tmp[int(1* len(startPoints)/4): int(3* len(startPoints)/4)]))
    print('Using a fix number of points per bin, {} points'.format(mean_nbpts))

    sigAnalytic = signal.hilbert(sig)    
    sigAnalytic *= np.exp(-1j*2*np.pi*fIF/fs*np.arange(len(sigAnalytic)))
    
    baseBandIGM  = np.empty(len(dwell_times_tmp)).astype(np.cdouble)
    for indx, (startIndx, endIndx) in enumerate(zip(startPoints, endPoints)):
        #Patch to limit the number of points, might be +-1 pt
        if endIndx-startIndx > (mean_nbpts + 1):
            tmp=int((endIndx+startIndx)/2)
            startIndx = tmp-int(mean_nbpts/2)
            endIndx = tmp+int(mean_nbpts/2)
#         print('new_length is ',endIndx-startIndx)    
        baseBandIGM[indx]  = np.mean(sigAnalytic[startIndx:endIndx])  
        
    return baseBandIGM


def getTriggerOccurence(signal, trigger_val, type="Rising"):
    if type == "Rising":
        preTriggerIndex = np.flatnonzero((signal[:-1] < trigger_val) & (signal[1:] > trigger_val))
    elif type == "Falling":
        preTriggerIndex = np.flatnonzero((signal[:-1] > trigger_val) & (signal[1:] < trigger_val))
    elif type == "both":
        preTriggerIndex = (np.flatnonzero((signal[:-1] < trigger_val) & (signal[1:] > trigger_val)) or
                   np.flatnonzero((signal[:-1] > trigger_val) & (signal[1:] < trigger_val)))
    slope = signal[preTriggerIndex+1] - signal[preTriggerIndex]
    return preTriggerIndex + (trigger_val-signal[preTriggerIndex])/slope


def getTriggerWindows(trig, beforeTriggerMaskPoints, afterTriggerMaskPoints, nbSamples ):
    triggerCrossings = (np.ceil(getTriggerOccurence(trig, 0.2, type="Rising"))).astype(int) 
    if triggerCrossings[0] == 1: # remove strange behavior of a early trig 
        triggerCrossings = triggerCrossings[1:]
    startPoints      = triggerCrossings[:-1] + afterTriggerMaskPoints
    endPoints        = triggerCrossings[1:] - beforeTriggerMaskPoints
    pointsPerMeas    = np.floor(np.mean(endPoints - startPoints)).astype(int)
    startPoints      = np.concatenate(([triggerCrossings[0]-beforeTriggerMaskPoints-pointsPerMeas], startPoints))
    endPoints        = np.concatenate(([triggerCrossings[0]-beforeTriggerMaskPoints], endPoints))
    nMeas            = len(startPoints)

    if not len(startPoints) == len(endPoints):
        warnings.warn('trigger problem')
    
    # print('First trig indexes: ',end="")
    # print(startPoints[0:5],endPoints[0:5])
    # print('Last trig indexes: ',end="")
    # print(startPoints[-5:],endPoints[-5:])

    # do we have the same number of points 
    if not nMeas == nbSamples: #find point to delete
        #case where we have one point more than expected
        if nMeas - nbSamples == 1:
            dwell_times_tmp = endPoints - startPoints
            if dwell_times_tmp[0] > dwell_times_tmp[-1]:
                startPoints= startPoints[1:]
                endPoints= endPoints[1:]
                # print('remove first igm point')
            else:
                startPoints= startPoints[:-1]
                endPoints= endPoints[:-1]
                # print('remove last igm point')
        else:
            warnings.warn('Problem with the trigger signal, need attention')

        # print('First trig indexes: ',end="")
        # print(startPoints[0:5],endPoints[0:5])
        # print('Last trig indexes: ',end="")
        # print(startPoints[-5:],endPoints[-5:])

    return startPoints, endPoints