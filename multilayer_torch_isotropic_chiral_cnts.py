import torch
import core_CD_torch as cd_core
import numpy as np

class chiral_multistack(torch.nn.Module):

    """Calculate CD of a multilayer stack with chiral materials and PCM under two phases, 
    which are described by two different material lists, ``stack_list_a`` and ``stack_list_c``.

    Extension to anisotropic materials
    ------------------------------
    The core solver already supports full-wavelength dependent 3 x 3 tensor for each layer.
    Hence, to extend to anisotropic materials, the material library file can provide complete
    dielectric and chiral parameter tensors with a shape of ``(N_wavelengths, 3, 3)``. 
    The full tensors can be directly loaded into the selected layer instead of only assigning 
    the diagonal elements. 

    Four-configuration measurement expansion
    ----------------------------------------
    The current script evaluates one illumination direction and one sample
    rotation.  The used four-configuration procedure requires four separate solver calls: 
    two azimuths separated by 90 degrees for front illumination, followed by the same two azimuths 
    for illumination from the opposite side.  Opposite-side illumination requires reversing the 
    layer and thickness order, swapping the incident and exit refractive indices, and
    using the physically mirrored laboratory-frame tensors for oriented anisotropic layers.  
    The intrinsic sign of a material's chirality is not changed merely because the sample is 
    physically flipped. The four calculated CD spectra can be averaged to obtain the 
    four-configuration CD response. Average absorbance is obtained by averaging the four
    absorbance spectra. 

    The opposite-side illumination calculations use the reversed layer and thickness order, swap 
    the incident and exit refractive indices, and physically flips every constitutive tensor by a 
    180-degree rotation about the laboratory y-axis.  This changes orientation-dependent off-diagonal 
    elements, but it does not reverse intrinsic chirality. 
    """

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


        # Four-configuration note:
        # This implementation stores only the forward layer lists. Starting from 
        # the present front calculation, four-configuration processing requires 
        # three additional solver calls: front at 90 deg rotation, reverse at 
        # current orientation, and reverse at 90 deg rotation. The reverse calculations
        # use the reverse lists by reversing the forward order and mapping each 
        # oriented material to a separately prepared mirror entry. Scalar isotropic 
        # layers keep the same material entry because a physical rotation does not 
        # alter an isotropic tensor. 
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


        # Thickness parameterization for optimization. If mind = maxd, the thickness is fixed. 
        # Reversing the stack later uses torch.flip, which preserves the gradient path.
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
        # initialize material properties for PCM in crystalline state case
        self.eps_list_c = torch.zeros((len(self.wl_range), self.thick_list.size(0), 3, 3), dtype=torch.cfloat, device=self.device)
        self.mu_list_c = torch.zeros_like(self.eps_list_c)
        self.xi_list_c = torch.zeros_like(self.eps_list_c)
        self.zeta_list_c = torch.zeros_like(self.eps_list_c)

        # initialize material properties for PCM in amorphous state case
        self.eps_list_a = torch.zeros((len(self.wl_range), self.thick_list.size(0), 3, 3), dtype=torch.cfloat, device=self.device)
        self.mu_list_a = torch.zeros_like(self.eps_list_a)
        self.xi_list_a = torch.zeros_like(self.eps_list_a)
        self.zeta_list_a = torch.zeros_like(self.eps_list_a)

        # This loop implements the scalar material interface. 
        # For an anisotropic extension, replace the diagonal-only assignments 
        # with a full-tensor assignment. 

        # For a four-configuration measurement, a separate reverse tensor
        # set would be constructed from the reversed material list.  Oriented 
        # anisotropic layers would be read from their mirror library entries. 
        # Those entries can be generated at the material-model level with the orientation 
        # changed from theta to pi - theta.  The reverse epsilon and kappa
        # both come from the reverse material entry.
        
        for layer_idx in range(self.thick_list.size(0)):

            # Crystalline/state-c branch
            eps_temp_c = torch.tensor(self.eps_lib[self.stack_list_c[layer_idx]], device=self.device)
            kappa_temp_c = torch.tensor(self.kappa_lib[self.stack_list_c[layer_idx]], device=self.device)

            # Relative permittivity eps_temp_c can be a full 3 x 3 tensor 
            # and be directly assigned to the full eps list.
            self.eps_list_c[:,layer_idx,0,0] = eps_temp_c*self.eps_0
            self.eps_list_c[:,layer_idx,1,1] = eps_temp_c*self.eps_0
            if self.stack_list_c[layer_idx] in ['cnt_left', 'cnt_right']: ### chiral cnt materials
                self.eps_list_c[:,layer_idx,2,2] = 3.4225*self.eps_0
            else:
                self.eps_list_c[:,layer_idx,2,2] = eps_temp_c*self.eps_0
            
            self.mu_list_c[:, layer_idx, 0, 0] = self.mu_0
            self.mu_list_c[:, layer_idx, 1, 1] = self.mu_0
            self.mu_list_c[:, layer_idx, 2, 2] = self.mu_0

            # For anisotropic chirality, kappa_temp_c would be a full 3 x 3 tensor 
            # and the entire xi/zeta layer would be assigned.
            self.xi_list_c[:, layer_idx, 0, 0] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_c)
            self.xi_list_c[:, layer_idx, 1, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_c)
            self.xi_list_c[:, layer_idx, 2, 2] = torch.sqrt(self.eps_0 * self.mu_0) * (-1j * kappa_temp_c)

            self.zeta_list_c[:, layer_idx, 0, 0] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_c)
            self.zeta_list_c[:, layer_idx, 1, 1] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_c)
            self.zeta_list_c[:, layer_idx, 2, 2] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_c)

            # Amorphous/state-a branch
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
            self.zeta_list_a[:, layer_idx, 2, 2] = torch.sqrt(self.eps_0 * self.mu_0) * (1j * kappa_temp_a)



    def forward(self):
        # updating thicklist in forward function
        self.thick_list = (torch.sigmoid(self.thickP) * (self.maxd - self.mind) + self.mind) * 1e-9

        # If four-configuration measurement is used, for each PCM state, four CD and ABS spectra are calculated: (1)
        # front illumination, azimuth 0; (2) front illumination, azimuth 90; (3) reverse illumination, azimuth 0; 
        # (4) reverse illumination, azimuth 90. The output CD and ABS spectra are averaged over four configurations. 
        _, _, CD_val_c, ABS_val_c = cd_core.tmm_rt_circular(self.thick_list, self.eps_list_c, self.mu_list_c, self.xi_list_c, self.zeta_list_c,
                                                                    self.n_in, self.n_out, self.w0_range, self.theta_0, self.phi_0, device=self.device)

        _, _, CD_val_a, ABS_val_a = cd_core.tmm_rt_circular(self.thick_list, self.eps_list_a, self.mu_list_a, self.xi_list_a, self.zeta_list_a,
                                                                    self.n_in, self.n_out, self.w0_range, self.theta_0,self.phi_0, device=self.device)


        # CD from the core is differential absorbance.  The factor 32980
        # converts it to millidegrees, and (2*purity-1) applies the selected
        # enantiomeric excess scaling used by this example.
        self.CD_c = CD_val_c* (2 * self.purity - 1) * 32980
        self.ABS_c = ABS_val_c
        self.CD_a = CD_val_a * (2 * self.purity - 1) * 32980
        self.ABS_a = ABS_val_a

        return self.CD_c, self.CD_a, self.ABS_c, self.ABS_a