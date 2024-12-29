import torch
import core_CD_torch as cd_core
import numpy as np

class chiral_multistack(torch.nn.Module):


    def __init__(self, wl_range=None, stack_list_a=None, stack_list_c=None, mind=None, maxd=None, eps_lib=None, kappa_lib=None, n_in=1, n_out=1.45, chiral_purity=0.87, theta=0, phi=0,
                 device='cuda:0', **kwargs):
        super().__init__()

        # Define constants as PyTorch tensors
        self.eps_0 = torch.tensor(8.8541878128e-12, dtype=torch.float32)
        self.mu_0 = torch.tensor(1.25663706212e-06, dtype=torch.float32)
        self.c_0 = torch.tensor(299792458.0, dtype=torch.float32)
        self.eta_0 = torch.tensor(376.73031366686166, dtype=torch.float32)

        self.wl_range = torch.from_numpy(wl_range.astype(np.float64)).to(device)  # simulation wavelength range (um)
        self.k0_range = 2 * 3.1415926 / (self.wl_range * 1e-6)
        self.w0_range = self.c_0 * self.k0_range
        self.stack_list_a = stack_list_a  # string list
        self.stack_list_c = stack_list_c  # string list
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.purity = chiral_purity
        self.eps_lib = eps_lib
        self.kappa_lib = kappa_lib
        self.mind = torch.tensor(mind).to(device)
        self.maxd = torch.tensor(maxd).to(device)


        # Parameter to be optimized
        self.thickP = torch.nn.Parameter(torch.randn(len(stack_list_a), dtype=torch.float32, device=device)*2)
        self.thick_list = (torch.sigmoid(self.thickP) * (self.maxd - self.mind) + self.mind) * 1e-9

        # Normal incidence for now
        self.phi_0 = torch.tensor(np.deg2rad(phi), dtype=torch.float32, device=device)
        self.theta_0 = torch.tensor(np.deg2rad(theta), dtype=torch.float32, device=device)

        # Result tensors to store results
        self.m_r_mat = torch.zeros((len(self.w0_range), 2, 2), dtype=torch.cfloat, device=device)
        self.m_t_mat = torch.zeros((len(self.w0_range), 2, 2), dtype=torch.cfloat, device=device)
        self.CD = torch.zeros(len(self.w0_range), dtype=torch.float32, device=device)
        self.ABS = torch.zeros_like(self.CD)

        #initialize CD and ABS
        self.CD_c = torch.zeros(len(self.w0_range), dtype=torch.float32, device=self.device)
        self.ABS_c = torch.zeros_like(self.CD_c)

        self.CD_a = torch.zeros(len(self.w0_range), dtype=torch.float32, device=self.device)
        self.ABS_a = torch.zeros_like(self.CD_a)

        ### create material list 
        # Material tensors
        # initialize
        #initialize material properties for PCM in crystalline state case
        self.eps_list_c = torch.zeros((len(self.wl_range), self.thick_list.size(0), 3, 3), dtype=torch.cfloat, device=self.device)
        self.mu_list_c = torch.zeros_like(self.eps_list_c)
        self.xi_list_c = torch.zeros_like(self.eps_list_c)
        self.zeta_list_c = torch.zeros_like(self.eps_list_c)

        #initialize material properties for PCM in amorphous state case
        self.eps_list_a = torch.zeros((len(self.wl_range), self.thick_list.size(0), 3, 3), dtype=torch.cfloat, device=self.device)
        self.mu_list_a = torch.zeros_like(self.eps_list_a)
        self.xi_list_a = torch.zeros_like(self.eps_list_a)
        self.zeta_list_a = torch.zeros_like(self.eps_list_a)
        
        for layer_idx in range(self.thick_list.size(0)):
            
            eps_temp_c = torch.tensor(self.eps_lib[self.stack_list_c[layer_idx]], device=self.device)
            kappa_temp_c = torch.tensor(self.kappa_lib[self.stack_list_c[layer_idx]], device=self.device)
            self.eps_list_c[:,layer_idx,0,0] = eps_temp_c*self.eps_0
            self.eps_list_c[:,layer_idx,1,1] = eps_temp_c*self.eps_0
            if self.stack_list_c[layer_idx] in ['cnt_left', 'cnt_right']: ### chiral cnt materials
                self.eps_list_c[:,layer_idx,2,2] = 3.4225*self.eps_0
            else:
                self.eps_list_c[:,layer_idx,2,2] = eps_temp_c*self.eps_0
            
            self.mu_list_c[:, layer_idx, 0, 0] = self.mu_0
            self.mu_list_c[:, layer_idx, 1, 1] = self.mu_0
            self.mu_list_c[:, layer_idx, 2, 2] = self.mu_0

            self.xi_list_c[:, layer_idx, 0, 0] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_c)
            self.xi_list_c[:, layer_idx, 1, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_c)
            self.xi_list_c[:, layer_idx, 2, 2] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_c)

            self.zeta_list_c[:, layer_idx, 0, 0] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_c)
            self.zeta_list_c[:, layer_idx, 1, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_c)
            self.zeta_list_c[:, layer_idx, 2, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_c)
            
            eps_temp_a = torch.tensor(self.eps_lib[self.stack_list_a[layer_idx]], device=self.device)
            kappa_temp_a = torch.tensor(self.kappa_lib[self.stack_list_a[layer_idx]], device=self.device)
            self.eps_list_a[:,layer_idx,0,0] = eps_temp_a*self.eps_0
            self.eps_list_a[:,layer_idx,1,1] = eps_temp_a*self.eps_0
            if self.stack_list_a[layer_idx] in ['cnt_left', 'cnt_right']: ### chiral cnt materials
                self.eps_list_a[:,layer_idx,2,2] = 3.4225*self.eps_0
            else:
                self.eps_list_a[:,layer_idx,2,2] = eps_temp_a*self.eps_0
            
            self.mu_list_a[:, layer_idx, 0, 0] = self.mu_0
            self.mu_list_a[:, layer_idx, 1, 1] = self.mu_0
            self.mu_list_a[:, layer_idx, 2, 2] = self.mu_0

            self.xi_list_a[:, layer_idx, 0, 0] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_a)
            self.xi_list_a[:, layer_idx, 1, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_a)
            self.xi_list_a[:, layer_idx, 2, 2] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_a)

            self.zeta_list_a[:, layer_idx, 0, 0] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_a)
            self.zeta_list_a[:, layer_idx, 1, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_a)
            self.zeta_list_a[:, layer_idx, 2, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_a)
            


    def forward(self):
        #for updating thicklist in forward function
        self.thick_list = (torch.sigmoid(self.thickP) * (self.maxd - self.mind) + self.mind) * 1e-9

        _, _, CD_val_c, ABS_val_c = cd_core.tmm_rt_circular(self.thick_list, self.eps_list_c, self.mu_list_c, self.xi_list_c, self.zeta_list_c,
                                                                    self.n_in, self.n_out, self.w0_range, self.theta_0, self.phi_0, device=self.device)

        _, _, CD_val_a, ABS_val_a = cd_core.tmm_rt_circular(self.thick_list, self.eps_list_a, self.mu_list_a, self.xi_list_a, self.zeta_list_a,
                                                                    self.n_in, self.n_out, self.w0_range, self.theta_0,self.phi_0, device=self.device)

        self.CD_c = CD_val_c* (2 * self.purity - 1) * 32980
        self.ABS_c = ABS_val_c
        self.CD_a = CD_val_a * (2 * self.purity - 1) * 32980
        self.ABS_a = ABS_val_a

        return self.CD_c, self.CD_a, self.ABS_c, self.ABS_a