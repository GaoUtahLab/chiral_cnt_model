# libraries
import numpy as np # numpy
import scipy.constants as spcs # scipy constants
from scipy.integrate import quad
from numpy.linalg import inv

# useful constants
eps_0 = spcs.epsilon_0
mu_0 = spcs.mu_0
c_0 = spcs.c
eta_0 = np.sqrt(mu_0/eps_0)
qe = spcs.e
eV = spcs.eV
kb = spcs.k
hbar = spcs.hbar


def chiral_molecules(exp_wl, env_eps, const_kappa):

    '''generation of a constant dielectric function vector for dielectrics

    Parameters
    ----------
    'exp_wl' = wavelength range (um)
    'const_eps' = constant dielectrics (number, unit 1)

    Returns
    -------
    'eps' = a constant dielectric function vector
    'kappa' = a zero vector
    
    '''

    eps = env_eps*np.ones(np.size(exp_wl),dtype = np.complex128)
    kappa = const_kappa*np.ones(np.size(exp_wl),dtype = np.complex128)

    return eps, kappa

def eps_chiral_CNT(exp_wl, hand='left'):

    '''computation of CNT dielectric constant and chiral parameter using one-resonance model

    Parameters
    ----------
    'exp_wl' = wavelength list (unit: um)

    Returns
    -------
    'kappa' = chiral parameter in exp_wl range
    'eps_R' = dielectric constant in exp_wl range
    
    '''

    m_kappa = np.zeros(np.size(exp_wl), dtype = np.complex128)
    m_eps_R = np.zeros(np.size(exp_wl), dtype = np.complex128)

    ### convert wavelength to angular frequency
    wl0_list = exp_wl*1e-6
    k0_list =  2*np.pi/wl0_list
    w0_list = c_0 * k0_list

    if hand == 'left':
        param_list = np.array([1.0117888 ,  1.36400176,  1.43500172,  1.23899827,  1.19369612, 0.65827978, -0.00497601]) ##  (6, 5), left
    elif hand == 'right':
        param_list = np.array([ 1.0117888 ,  1.36400176,  1.43500172,  1.23899827,  3.26404541,  0.64251043, -0.0070093]) ## (11, -5), right

    else:
        print('Illegal handedness value')
    
    wl0_res = param_list[0]*1e-6 ## 1st parameter: resonance wavelength, um
    k_res = 2*np.pi/wl0_res
    w0_res = c_0 * k_res

    eps_inf = param_list[1]
    eps_wp = param_list[2]*1e15
    eps_gamma = param_list[3]*1e14

    kappa_R = param_list[4]*1e12 ## 4th
    kappa_left_gamma = param_list[5]*1e14

    delta_res = param_list[6]
    kappa_w0_res = w0_res*(1+delta_res)

    for wv_idx, w0 in enumerate(w0_list):

        if hand == 'left':
            m_kappa[wv_idx] = kappa_R*w0/(kappa_w0_res**2 - w0**2 - 1j*w0*kappa_left_gamma)
        elif hand =='right':
            m_kappa[wv_idx] = -kappa_R*w0/(kappa_w0_res**2 - w0**2 - 1j*w0*kappa_left_gamma)
        else:
            print('Illegal handedness value')
        
        m_eps_R[wv_idx] = eps_inf + eps_wp**2/(w0_res**2 - w0**2 - 1j*w0*eps_gamma)

    return m_kappa, m_eps_R
