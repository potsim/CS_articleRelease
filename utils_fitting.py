"""
Fit fonctions
Based on Nate's code used for lookup-table fitting of CO2/H2O around 1600nm

Created on Spetember 10, 2021

@author: SPO
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, interpn

import hapi

SPEED_OF_LIGHT = 299792458 # (m/s)

def spc_lookup2(xx, lookup_table_wvn_axis, lookup_table, molefraction, path_cm, shift ):
    
    ### original algo
    # y0 = np.squeeze(lookup_table(np.array(([molefraction])))) * path_cm
    # y_model_td = np.fft.irfft(xc.xc_shift(y0, xx, xc_x0, xc_dx, shift))

    ## quadratic interp for resampling the x axis. Linear for the lookup table 
    y0 = np.squeeze(lookup_table(np.array(([molefraction])))) * path_cm
    set_interp = interp1d(lookup_table_wvn_axis, y0, kind='quadratic')  #linear, quadratic, cubic
    new_y = set_interp(xx+shift)
    y_model_td = np.fft.irfft(new_y)

    # # Need to modify what is passed to the function one single spline for every thing
    # y_model_td = interpn((np.linspace(.001,.05,31),lookup_table_wvn_axis), lookup_table, (molefraction,xx+shift), 
    #                         method='splinef2d') * path_cm
    # y_model_td = np.fft.irfft(y_model_td)

    # # Compute directly from Hitran. Time consuming.
    # pi_atm = pi_mbar / 1013.25
    # y_model_td = calc_spectrum_fd(xx+shift, mol_id, iso, molefraction, pi_atm, ti_kelvin, path_cm, 0, 'co2', 
    #                               sdvoigt = False, LineMixingRosen = False)
    # y_model_td = np.fft.irfft(y_model_td)

    return y_model_td


def generateLookuptbl(specs):
    xi = np.linspace(specs['molfrac_range'][0],specs['molfrac_range'][1],specs['nb_sampled_pts'])
    lookuptbl, wvn_axis = calc_lookup_table( specs['ti_kelvin'], specs['p_atm'], xi, band_fit = specs['band_fit'], 
                            dx = specs['spc_resolution'], name = specs['db_name'], path_cm = 1, 
                            mol_id = specs['mol_id'], iso = specs['iso'],
                            sdvoigt = False, linemixingrosen = False)

    return lookuptbl, wvn_axis


def calc_lookup_table(ti_kelvin, p_atm, xi, band_fit = [6140,6300], 
                      dx = .005, name = 'co2', path_cm = 1, mol_id = 2, iso = 1,
                      save_name = '', sdvoigt = False, linemixingrosen = False):
    '''
    Make H5 file of HITRAN absorption model at different conditions
    INPUTS:
        ti_kelvin = np.arange(280,310,5)
        pi_mbar = np.arange(800,850,10)
        xi = np.linspace(.001,.05,10)
    '''
    x_wvn = np.arange(band_fit[0], band_fit[1], dx)
    n_x = len(xi)
    absorption = np.zeros((n_x, len(x_wvn)))
    for k, xk in enumerate(xi):
            absorption[k,:] = calc_spectrum_fd(x_wvn, mol_id, iso,
                          xk, p_atm, ti_kelvin, path_cm, 0, name, sdvoigt = sdvoigt,
                          LineMixingRosen = linemixingrosen)
    # save to interpolator # but how to get array rather than scalar out?
    interp_func = RegularGridInterpolator((xi,),absorption)

    # # Write to H5 file for later recall
    # if len(save_name) == 0:
    #     save_name = name
    # look.lookup_to_h5(save_name, absorption, x_wvn, ti_kelvin, pi_mbar, xi)
    
    return interp_func, x_wvn




def calc_spectrum_fd(xx, mol_id, iso, molefraction, pressure, temperature, 
                     pathlength, shift, db_file_name='H2O', sdvoigt = False,
                     LineMixingRosen=False):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        db_file_name -> string name (no extension) of Hitran linelist file
    
    TODO: how to pull molecule name from prefix of call?
    '''
    if sdvoigt:
        nu, coef = hapi.absorptionCoefficient_SDVoigt(((int(mol_id), int(iso), molefraction),),
                db_file_name, HITRAN_units=False,
                OmegaGrid = xx + shift,
                Environment={'p':pressure,'T':temperature},
                Diluent={'self':molefraction,'air':(1-molefraction)},
                LineMixingRosen=LineMixingRosen) 
    else:
        nu, coef = hapi.absorptionCoefficient_Voigt(((int(mol_id), int(iso), molefraction),),
                db_file_name, HITRAN_units=False,
                OmegaGrid = xx + shift,
                Environment={'p':pressure,'T':temperature},
                Diluent={'self':molefraction,'air':(1-molefraction)})
#     absorp = coef * pathlength * hapi.abundance(mol_id, iso)
    absorp = coef * pathlength
    return absorp






def largest_prime_factor(n):
    '''
    Want 2 * (x_stop - x_start - 1) to have small largest_prime_factor
    '''
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def bandwidth_select_td(x_array, band_fit, max_prime_factor = 500):
    '''
    Tweak bandwidth selection for swift time-domain fitting.
    
    Time-domain fit does inverse FFT for each nonlinear least-squares iteration,
    and speed of FFT goes with maximum prime factor.
    
    INPUTS:
        x_array = x-axis for measurement transmission spectrum
        band_fit = [start_frequency, stop_frequency]
    '''
    x_start = np.argmin(np.abs(x_array - band_fit[0]))
    x_stop = np.argmin(np.abs(x_array - band_fit[1]))
    if x_stop == len(x_array)-1:
        x_stop += 1
       
    len_td = 2 * (np.abs(x_stop - x_start) - 1) # np.fft.irfft operation
    prime_factor = largest_prime_factor(len_td)
    while prime_factor > max_prime_factor:
        x_stop -= 1
        len_td = 2 * (np.abs(x_stop - x_start) - 1)
        prime_factor = largest_prime_factor(len_td)
    return x_start, x_stop

def weight_func(len_fd, bl, etalons = []):
    '''
    Time-domain weighting function, set to 0 over selected baseline, etalon range
    INPUTS:
        len_fd = length of frequency-domain spectrum
        bl = number of points at beginning to attribute to baseline
        etalons = list of [start_point, stop_point] time-domain points for etalon spikes
    '''
    weight = np.ones(2*(len_fd-1))
    weight[:bl] = 0; weight[-bl+1:] = 0
    for et in etalons:
        weight[et[0]:et[1]] = 0
        weight[-et[1]+1:-et[0]+1] = 0
    return weight

def weight_func_ps(x_wvn, bl_ps, etalons_ps = []):
    '''
    Weighting function, scaled by effective time (picoseconds) not index
    '''
    len_fd = len(x_wvn)
    ti_ps = cepstrum_xaxis(x_wvn)[:len_fd]
    bl = np.argmin(np.abs(bl_ps - ti_ps))
    etalons = []
    for et in etalons_ps:
        st = np.argmin(np.abs(et[0] - ti_ps))
        en = np.argmin(np.abs(et[1] - ti_ps))
        etalons.append([st,en])
    weight = weight_func(len_fd, bl, etalons)
    return weight


#### It doesn't work for both side of the spectrum ....
def cepstrum_xaxis(x_data):
    '''
    Calculate x-axis scaling of time-domain cepstrum.
    In same units as x_data (the frequency-axis array)
    center data point should be 1/frep, first data point is DC
    INPUT:
        x_data (cm-1) array of frequencies
    OUTPUT:
        ti_ps (ps) array of cepstrum times (picoseconds)
    TODO:
        make sure this isn't totally wrong
    '''
    WVN2HZ = 100 * SPEED_OF_LIGHT
    dx_hz = (x_data[1] - x_data[0]) * WVN2HZ
#    delta = x_data[-1] - x_data[0]
    nx = len(x_data)
    # what is cosine-wave period corresponding to each point?
    ti_ps = np.linspace(0,1,nx) / dx_hz * 1e12
    # wrap-around the symmetric second half of cepstrum
    ti_ps = np.concatenate((ti_ps, ti_ps[-2:0:-1]))
#    cosi = 2 * delta / ti
    
    # turnaround point, y_td[nx + bar] = y_td[nx - bar - 2]
    # so turnaround at y_td[nx-1]
    
    return ti_ps

    