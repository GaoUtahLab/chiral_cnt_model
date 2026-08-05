import torch
import torch.linalg as tln
import scipy.constants as spcs  # For constants like epsilon_0, mu_0, etc.

# useful constants
eps_0 = torch.tensor(spcs.epsilon_0, dtype=torch.float64)  # epsilon_0 in F/m
mu_0 = torch.tensor(spcs.mu_0, dtype=torch.float64)  # mu_0 in H/m
c_0 = torch.tensor(spcs.c, dtype=torch.float64)  # speed of light in m/s
eta_0 = torch.sqrt(mu_0 / eps_0)

def calc_Pmat(eps, mu, xi, zeta, w, q, phi, device='cuda:0'):
    """
    Calculate the P matrix of a single layer in a wavelength (w) range

    Args:   'eps' = w * 3*3 complex tensor: permittivity dyadic
            'mu' = w * 3*3 complex tensor: permeability dyadic
            'xi' = w * 3*3 complex tensor: magnetoelectric dyadic, D = eps*E + xi*H
            'zeta' = w * 3*3 complex tensor: magnetoelectric dyadic, B = zeta*E + mu*H
            'w' = w * 1 real number tensor: angular frequency of light
            'q' = w * 1 complex number tensor: parallel wavenumber
            'phi' = real number [0, 2pi): incident azimuthal angle in radians.
    Returns:  'P' = w * 4*4 complex tensor: propagation matrix

    """
    # calculate in-plane wave vectors
    qx = q * torch.cos(phi).to(device)
    qy = q * torch.sin(phi).to(device)

    # calculate coefficients
    denominator = eps[:, 2, 2] * mu[:, 2, 2] - xi[:, 2, 2] * zeta[:, 2, 2]

    eezx = -(mu[:, 2, 2] * eps[:, 2, 0] - xi[:, 2, 2] * (zeta[:, 2, 0] + q / w * torch.sin(phi).to(device))) / denominator
    eezy = -(mu[:, 2, 2] * eps[:, 2, 1] - xi[:, 2, 2] * (zeta[:, 2, 1] - q / w * torch.cos(phi).to(device))) / denominator
    ehzx = (xi[:, 2, 2] * mu[:, 2, 0] - mu[:, 2, 2] * (xi[:, 2, 0] - q / w * torch.sin(phi).to(device))) / denominator
    ehzy = (xi[:, 2, 2] * mu[:, 2, 1] - mu[:, 2, 2] * (xi[:, 2, 1] + q / w * torch.cos(phi).to(device))) / denominator
    hezx = (zeta[:, 2, 2] * eps[:, 2, 0] - eps[:, 2, 2] * (zeta[:, 2, 0] + q / w * torch.sin(phi).to(device))) / denominator
    hezy = (zeta[:, 2, 2] * eps[:, 2, 1] - eps[:, 2, 2] * (zeta[:, 2, 1] - q / w * torch.cos(phi).to(device))) / denominator
    hhzx = -(eps[:, 2, 2] * mu[:, 2, 0] - zeta[:, 2, 2] * (xi[:, 2, 0] - q / w * torch.sin(phi).to(device))) / denominator
    hhzy = -(eps[:, 2, 2] * mu[:, 2, 1] - zeta[:, 2, 2] * (xi[:, 2, 1] + q / w * torch.cos(phi).to(device))) / denominator

    # calculate tensors
    m1 = torch.stack((zeta[:, 1, 0], zeta[:, 1, 1], mu[:, 1, 0], mu[:, 1, 1],
                       -zeta[:, 0, 0], -zeta[:, 0, 1], -mu[:, 0, 0], -mu[:, 0, 1],
                       -eps[:, 1, 0], -eps[:,1, 1], -xi[:, 1, 0], -xi[:, 1, 1],
                       eps[:, 0, 0], eps[:,0, 1], xi[:, 0, 0], xi[:, 0, 1]), dim=1).reshape((len(w),4,4))

    mJ = torch.ones((len(w), 4, 4), dtype=torch.complex128, device=device)

    m21 = torch.diag_embed(torch.stack([zeta[:, 1, 2] + q / w * torch.cos(phi), -zeta[:, 0, 2] + q / w * torch.sin(phi),
                                   -eps[:, 1, 2], eps[:, 0, 2]], dim=1))
    m22 = torch.diag_embed(torch.stack([eezx, eezy, ehzx, ehzy], dim=1))

    m2 = torch.matmul(torch.matmul(m21, mJ), m22)

    m31 = torch.diag_embed(torch.stack([mu[:, 1, 2],-mu[:, 0, 2],-xi[:, 1, 2] + q / w * torch.cos(phi),
                                   xi[:, 0, 2] + q / w * torch.sin(phi)], dim=1))
    m32 = torch.diag_embed(torch.stack([hezx, hezy, hhzx, hhzy], dim=1))

    m3 = torch.matmul(torch.matmul(m31, mJ), m32)

    # finally calculate P matrix
    P = w.reshape(-1,1,1) * (m1 + m2 + m3)
    return P


def calc_Mmat_from_Pmat(P, d, device='cuda:0'):
    """
    Calculate the M matrix of a single layer with P matrix and thickness d in a wavelength (w) range.

    Args:   'P' = w*4*4 complex tensor: P matrix
            'd' = positive real number: thickness

    Returns:  'M' = w*4*4 complex tensor: transfer matrix

    """
    # print(P,'P')
    # print(d,'d')
    M = torch.matrix_exp(1j * P * d).to(device)
    # print(M)
    return M


def calc_Mmat(eps, mu, xi, zeta, w, q, phi, d, device='cuda:0'):
    """
    Calculate the M matrix of a single layer in a wavelength (w) range.

    Args:   'eps' = w*3*3 complex tensor: permittivity dyadic
            'mu' = w*3*3 complex tensor: permeability dyadic
            'xi' = w*3*3 complex tensor: magnetoelectric dyadic, D = eps*E + xi*H
            'zeta' = w*3*3 complex tensor: magnetoelectric dyadic, B = zeta*E + mu*H
            'w' = w * 1 real number tensor: angular frequency of light
            'q' = w * 1 complex number tensor: parallel wavenumber
            'phi' = real number [0, 2pi): incident azimuthal angle in radians
            'd' = real number: thickness
    Returns:  'M' = w*4*4 complex tensor: transfer matrix

    """
    P = calc_Pmat(eps, mu, xi, zeta, w, q, phi, device)
    M = calc_Mmat_from_Pmat(P, d, device)
    return M


def calc_Mmat_all(eps_list, mu_list, xi_list, zeta_list, w, q, phi, d_list, device='cuda:0'):
    """
    Calculate the M matrix of multiple layers in a wavelength (w) range.

    Args:   'eps_list' = w*n*3*3 complex tensor: permittivity dyadic
            'mu_list' = w*n*3*3 complex tensor: permeability dyadic
            'xi_list' = w*n*3*3 complex tensor: magnetoelectric dyadic, D = eps*E + xi*H
            'zeta_list' = w*n*3*3 complex tensor: magnetoelectric dyadic, B = zeta*E + mu*H
            'w' = w * 1 real number tensor: angular frequency of light
            'q' = w * 1 complex number tensor: parallel wavenumber
            'phi' = real number [0, 2pi): incident azimuthal angle in radians (float or tensor)
            'd_list' = n real tensor: thickness (1D tensor)

    Returns:  'M_total' = w*4x4 complex tensor: total transfer matrix
              'M_list' = w*n*4*4 complex tensor: individual transfer matrices
    """
    if d_list.size(0) == eps_list.size(1) == mu_list.size(1) == xi_list.size(1) == zeta_list.size(1):
        M_list = torch.zeros((len(w), d_list.size(0), 4, 4), dtype=torch.complex128, device=device)
        # M_total = torch.eye(4, dtype=torch.complex128, device=device)
        M_total = torch.eye(4, dtype=torch.complex128, device=device).repeat(len(w), 1, 1)
        # Loop through each layer and calculate its M matrix
        for i_d, d in enumerate(d_list):
            eps_tmp = eps_list[:, i_d, :, :]
            mu_tmp = mu_list[:, i_d, :, :]
            xi_tmp = xi_list[:, i_d, :, :]
            zeta_tmp = zeta_list[:, i_d, :, :]
            M_tmp = calc_Mmat(eps_tmp, mu_tmp, xi_tmp, zeta_tmp, w, q, phi, d, device)
            M_list[:, i_d, :, :] = M_tmp
            M_total = torch.matmul(M_tmp, M_total)
        # print(M_total)
        return M_total, M_list

    else:
        raise ValueError("d_list, eps_list, mu_list, xi_list, zeta_list should have the same dimension along the first axis")

def calc_Kmat(phi, theta, n, device='cuda:0'):
    """
    Calculate the K matrix using PyTorch.

    Args:
        'phi' (torch.Tensor): azimuthal angle (scalar)
        'theta' (torch.Tensor): polar angle, which can be different for incident/transmission side (scalar)
        'n' (torch.Tensor): refractive index of the superstrate/substrate (scalar)

    Returns:
        'K' (torch.Tensor): 4x4 complex matrix
    """
    K = torch.tensor([
        [-torch.sin(phi), -torch.cos(phi) * torch.cos(theta), -torch.sin(phi), torch.cos(phi) * torch.cos(theta)],
        [torch.cos(phi), -torch.sin(phi) * torch.cos(theta), torch.cos(phi), torch.sin(phi) * torch.cos(theta)],
        [-n / eta_0 * torch.cos(phi) * torch.cos(theta), n / eta_0 * torch.sin(phi),
         n / eta_0 * torch.cos(phi) * torch.cos(theta), n / eta_0 * torch.sin(phi)],
        [-n / eta_0 * torch.sin(phi) * torch.cos(theta), -n / eta_0 * torch.cos(phi),
         n / eta_0 * torch.sin(phi) * torch.cos(theta), -n / eta_0 * torch.cos(phi)]
    ], dtype=torch.complex128, device=device)  # Ensure the matrix is complex

    return K

def calc_Kinc(phi, theta_inc, n_inc, device='cuda:0'):
    """
    Calculate the K matrix of the incident side using PyTorch.

    Args:
        'phi' (torch.Tensor): azimuthal angle (scalar)
        'theta_inc' (torch.Tensor): incident polar angle (scalar)
        'n_inc' (torch.Tensor): refractive index of the incident side (scalar)
        'eta_0' (torch.Tensor): wave impedance in vacuum (predefined constant)

    Returns:
        'Kinc' (torch.Tensor): 4x4 complex matrix (K matrix at the incident side)
    """
    Kinc = calc_Kmat(phi, theta_inc, n_inc, device)

    return Kinc

def calc_Ktr(phi, theta_inc, n_inc, n_tr, device='cuda:0'):
    """
    Calculate the K matrix of the transmission side using PyTorch.

    Args:
        'phi': azimuthal angle
        'theta_inc': incident polar angle
        'n_inc': refractive index of the incident side
        'n_tr': refractive index of the transmission side

    Returns:
        'Ktr': 4x4 complex matrix (K matrix at the transmission side)
    """
    theta_tr = correct_theta_forward(torch.asin(n_inc / n_tr * torch.sin(theta_inc) + 0j).to(device))  # transmission angles
    # Caution! np.conjugat() is required so that Im(kz)~Im(np.cos(theta_tr))>0, exp(ikz*z) is decaying by propagation along +z direction. This requires Im(theta_tr)<0 in python

    Ktr = calc_Kmat(phi, theta_tr, n_tr, device)

    return Ktr


def correct_theta_forward(theta):
    """
    Correct complex theta for the forward propagating waves. For waves exp(ikz*z), forward propagating waves requires Im(kz)>0, which requires Im(theta)<0 in python.

    Args:  'theta' = 1 complex ndarray: uncorrected complex theta for forward propagating waves.
           'theta_correct' = 1 complex ndarray: corrected complex theta for forward propagating waves.
    """
    theta_correct = theta
    if theta.numel() == 1:  # single theta
        if theta.imag > 0:
            theta_correct = torch.conj(theta)
    elif theta.numel() > 1:
        for i_theta, theta in enumerate(theta):  # theta array
            if theta.imag > 0:
                theta_correct[i_theta] = torch.conjugate(theta)
        return theta_correct
    else:
        raise ValueError('theta should be a complex number or list.')

    return theta_correct


def calc_Mat_KMK(M_total, K_inc, K_tr):
    """
    Calculate the total matrix of the problem

    Args:
        'M_total' (torch.Tensor): 4x4 complex tensor: total M matrix for all layers.
        'K_inc' (torch.Tensor): 4x4 complex tensor: incident K matrix
        'K_tr' (torch.Tensor): 4x4 complex tensor: transmission K matrix

    Returns:
        'Mat' (torch.Tensor): The result of the matrix multiplication and inversion.
    """
    Mat = tln.inv(K_tr) @ M_total @ K_inc  # Matrix multiplication
    return Mat


def create_Mat(d_list, eps_list, mu_list, xi_list, zeta_list, n1, n2, w, theta, phi, device='cuda:0'):
    """
    Create the matrix for the transfer matrix method problem

    Args:
        'd_list' (torch.Tensor): n real tensor: thickness
        'eps_list' (torch.Tensor): w*n*3*3 complex tensor: permittivity dyadic
        'mu_list' (torch.Tensor): w*n*3*3 complex tensor: permeability dyadic
        'xi_list' (torch.Tensor): w*n*3*3 complex tensor: magnetoelectric dyadic, D = eps*E + xi*H
        'zeta_list' (torch.Tensor): w*n*3*3 complex tensor: magnetoelectric dyadic, B = zeta*E + mu*H
        'n1' (torch.Tensor): 1 real tensor: refractive index of the incident side
        'n2' (torch.Tensor): 1 real tensor: refractive index of the transmission side
        'w' (torch.Tensor): 1 real tensor: angular frequency of light
        'theta' (torch.Tensor): 1 real tensor: incident polar angle
        'phi' (torch.Tensor): real tensor [0, 2pi): incident azimuthal angle in radians

    Returns:
        'Mat' (torch.Tensor): 4*4 complex tensor: complete transfer matrix K_tr^(-1) @ M @ K_inc
    """
    k0 = w / c_0  # vacuum wavenumber
    q = n1 * k0 * torch.sin(theta).to(device)  # parallel wavevectors

    M_total, M_list = calc_Mmat_all(eps_list, mu_list, xi_list, zeta_list, w, q, phi, d_list, device)
    K_inc = calc_Kinc(phi, theta, n1, device)
    K_tr = calc_Ktr(phi, theta, n1, n2, device)
    Mat = calc_Mat_KMK(M_total, K_inc, K_tr)

    return Mat


def solve_Mat(Mat):
    """
    Solve the transfer matrix problem: (ts, tp, 0, 0)^T = Mat @ (as, ap, rs, rp)^T
    using the direct solving method. (fast)

    Args:    'Mat' = w*4*4 complex tensor: complete transfer matrix K_tr^(-1)@ M @ K_inc

    Returns: 'r_mat' = w*2*2 complex tensor: reflection matrix {rss, rsp; rps, rpp}
             't_mat' = w*2*2 complex tensor: transmission matrix {tss, tsp; tps, tpp}
    """
    M = Mat

    # Calculate the denominator
    deno_M = M[:, 2, 2] * M[:, 3, 3] - M[:, 2, 3] * M[:, 3, 2]

    # Calculate reflection coefficients
    rss = (-M[:, 2, 0] * M[:, 3, 3] + M[:, 2, 3] * M[:, 3, 0]) / deno_M
    rps = (-M[:, 2, 2] * M[:, 3, 0] + M[:, 3, 2] * M[:, 2, 0]) / deno_M
    rsp = (-M[:, 2, 1] * M[:, 3, 3] + M[:, 2, 3] * M[:, 3, 1]) / deno_M
    rpp = (-M[:, 2, 2] * M[:, 3, 1] + M[:, 3, 2] * M[:, 2, 1]) / deno_M

    # Calculate transmission coefficients
    tss = M[:, 0, 0] + M[:, 0, 2] * rss + M[:, 0, 3] * rps
    tps = M[:, 1, 0] + M[:, 1, 2] * rss + M[:, 1, 3] * rps
    tsp = M[:, 0, 1] + M[:, 0, 2] * rsp + M[:, 0, 3] * rpp
    tpp = M[:, 1, 1] + M[:, 1, 2] * rsp + M[:, 1, 3] * rpp
    # print(tss,tsp,tps,tpp)
    # Store the reflection and transmission matrices
    r_mat = torch.stack((rss, rsp, rps, rpp),dim=1).reshape((M.shape[0], 2, 2))
    t_mat = torch.stack((tss, tsp, tps, tpp),dim=1).reshape((M.shape[0], 2, 2))

    return r_mat, t_mat


def tmm_rt_circular(d_list, eps_list, mu_list, xi_list, zeta_list, n1, n2, w, theta, phi, device='cuda:0'):
    """
    Solve the reflection and transmission matrix using TMM.

    Args:   'd_list' = n real ndarray: thickness
            'eps_list' = w*n*3*3 complex ndarray: permittivity dyadic
            'mu_list' = w*n*3*3 complex ndarray: permeability dyadic
            'xi_list' = w*n*3*3 complex ndarray: magnetoelectric dyadic, D = eps*E + xi*H
            'zeta_list' = w*n*3*3 complex ndarray: magnetoelectric dyadic, B = zeta*E + mu*H
            'n1' = 1 real number: refractive index of the incident side
            'n2' = 1 real number: refractive inex of the transmission side
            'w' = 1 real number: angular frequency of light
            'theta' = 1 real number: incident polar angle
            'phi' = real number [0, 2pi): incident azimuthal angle in radians

    Returns: 'r_mat' = w*2*2 complex ndarray: reflection matrix {rss, rsp; rps, rpp}
             't_mat' = w*2*2 complex ndarray: transmission matrix {tss, tsp; tps, tpp}
             'CD' = w*1 real ndarray: circular dichroism 
             'ABS' = w*1 real ndarray: average absorption

    """
    Mat = create_Mat(d_list, eps_list, mu_list, xi_list, zeta_list, n1, n2, w, theta, phi, device)
    r_mat, t_mat = solve_Mat(Mat)

    CD, ABS = calc_RT_circular(r_mat, t_mat, n1, n2, theta)
    # print(CD)
    return r_mat, t_mat, CD, ABS


def calc_RT_circular(r_mat, t_mat, n1, n2, theta_inc):
    """
    Calculate the power reflection and transmission from the field reflection and transmission coefficients.
    Note rsp means p-polarized incident and s-polarized reflected

    Args:   'r_mat' = w*2*2 complex tensor: reflection coefficient matrix {rss, rsp; rps, rpp}
            't_mat' = w*2*2 complex tensor: transmission coefficient matrix {tss, tsp; tps, tpp}
            'n1' = 1 real number: refractive index of the incident side
            'n2' = 1 real number: refractive inex of the transmission side
            'theta_inc' = 1 number: incident angles

    Returns: 'CD' = w*1 real ndarray: circular dichroism 
             'ABS' = w*1 real ndarray: average absorption

    """
    # Calculate transmission angles
    theta_tr = correct_theta_forward(torch.asin(n1 / n2 * torch.sin(theta_inc) + 0j))  # transmission angles

    # Field coefficients
    rss = r_mat[:, 0, 0]
    rsp = r_mat[:, 0, 1]
    rps = r_mat[:, 1, 0]
    rpp = r_mat[:, 1, 1]

    tss = t_mat[:, 0, 0]
    tsp = t_mat[:, 0, 1]
    tps = t_mat[:, 1, 0]
    tpp = t_mat[:, 1, 1]


    # Circular field coefficients
    rLL = -((rss + rpp) + 1j * (rsp - rps)) / 2
    rLR = ((rss - rpp) - 1j * (rsp + rps)) / 2
    rRL = ((rss - rpp) + 1j * (rsp + rps)) / 2
    rRR = -((rss + rpp) - 1j * (rsp - rps)) / 2
    tLL = ((tss + tpp) + 1j * (tsp - tps)) / 2
    tLR = -((tss - tpp) - 1j * (tsp + tps)) / 2
    tRL = -((tss - tpp) + 1j * (tsp + tps)) / 2
    tRR = ((tss + tpp) - 1j * (tsp - tps)) / 2

    # Calculate circular power transmission/reflection
    RLL = torch.abs(rLL) ** 2
    RLR = torch.abs(rLR) ** 2
    RRL = torch.abs(rRL) ** 2
    RRR = torch.abs(rRR) ** 2

    coef_T = n2 / n1 * torch.real(torch.cos(theta_tr)) / torch.cos(theta_inc)
    TLL = coef_T * torch.abs(tLL) ** 2
    TLR = coef_T * torch.abs(tLR) ** 2
    TRL = coef_T * torch.abs(tRL) ** 2
    TRR = coef_T * torch.abs(tRR) ** 2

    AL = -torch.log10(TLL + TRL)
    AR = -torch.log10(TLR + TRR)
    CD = AL - AR
    Avg_ABS = (AL + AR) / 2
    return CD, Avg_ABS