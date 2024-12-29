from multilayer_torch import chiral_multistack
import numpy as np
import matplotlib.pyplot as plt
import mat
import torch

## load experimental data
exp_wl = np.linspace(0.9, 1.2, 100, endpoint=True)

theta_angle = 0.0
cnt_thickness1 = 55.0
cnt_thickness2 = 55.0
phi_angle = 0

## create material parameters
chiral_cnt_kappa_left, chiral_cnt_eps_left = mat.eps_chiral_CNT(exp_wl,hand='left')
chiral_cnt_kappa_right, chiral_cnt_eps_right = mat.eps_chiral_CNT(exp_wl,hand='right')

eps_lib = {'cnt_left': chiral_cnt_eps_left, 'cnt_right': chiral_cnt_eps_right}
kappa_lib = {'cnt_left': chiral_cnt_kappa_left, 'cnt_right': chiral_cnt_kappa_right}

## create chiral heterostructures
name_list = ['cnt_left']
name_list2 = ['cnt_right']

mind = np.array([cnt_thickness1])
maxd = mind

mind2 = np.array([cnt_thickness2])
maxd2 = mind2

def tensor_to_np(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy() if arr.is_cuda else arr.detach().numpy()
    return arr

model = chiral_multistack(wl_range=exp_wl, stack_list_c=name_list,stack_list_a=name_list, mind=mind, maxd=maxd, eps_lib=eps_lib, kappa_lib=kappa_lib, n_in=1,
                        n_out=1.45, chiral_purity=0.87, theta=theta_angle, phi=phi_angle, device='cpu')


model2 = chiral_multistack(wl_range=exp_wl, stack_list_c=name_list2,stack_list_a=name_list2, mind = mind2, maxd=maxd2, eps_lib=eps_lib, kappa_lib=kappa_lib, n_in=1,
                        n_out=1.45, chiral_purity=0.87, theta=theta_angle, phi=phi_angle, device='cpu')


wl = exp_wl*1e3  #unit: nm
CD_left, _, ABS_left, _ = model()
CD_left = tensor_to_np(CD_left)
ABS_left = tensor_to_np(ABS_left)

CD_right, _, ABS_right, _ = model2()
CD_right = tensor_to_np(CD_right)
ABS_right = tensor_to_np(ABS_right)

plt.figure(figsize=(7.0, 5.0))
plt.plot(wl, CD_left, 'b-')
plt.plot(wl, CD_right, 'r-')
plt.xlabel('Wavelength (nm)')
plt.ylabel('CD (mdeg)')
plt.show()

plt.figure(figsize=(7.0, 5.0))
plt.plot(wl, ABS_left, 'b-')
plt.plot(wl, ABS_right, 'r-')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Attenuation')
plt.show()

