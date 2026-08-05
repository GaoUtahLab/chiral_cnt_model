# Calculating circular dichroism involving layers of bi-anisotropic materials
# and reconfigurable photonic materials

### Main file info:
There are 4 python file:

```bash
1. "core_CD_torch.py" ## PyTorch-implemented core transfer matrix solver code for calculating CD of multilayers of general bi-anisotropic materials
2. "mat.py" ## Material model library
3. "multilayer_torch_isotropic_chiral_cnts.py" ## Example model describing multiple layers consisting of isotropic chiral CNT films
4. "example_code_isotropic_chiral_cnts.py" ## Example code to plot CD and absorption of isotropic chiral CNT films
```
Notes in mat.py and multilayer_torch_isotropic_chiral_cnts.py files describe how to extend to general bi-anisotropic materials.

### Environment and system info:

```bash
1. python 3.11.9
2. pytorch 2.2.2
3. numpy   1.26.4
4. matplotlib  3.8.4
5. scipy   1.12.0
```

### Demo code running

```bash
% python example_code_isotropic_chiral_cnts.py
```
The demo code plots two figures displaying calculated CD spectra and absorption spectra for (6,5) and (11,-5) CNTs, respectively. 

