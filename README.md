# Chiral CNT Code

### Main file info:
There are 4 python file:

```bash
1. "core_CD_torch.py" ## Core transfer matrix code for calculating CD, implemented using PyTorch
2. "mat.py" ## Chiral CNT model file
3. "multilayer_torch.py" ## Torch model describing multiple layers consisting of chiral CNTs
4. "sample_code.py" ## An example code to plot CD and absorption of chiral CNT films
```

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
% python sample_code.py
```
It will plot two figures displaying calculated CD spectra and absorption spectra for (6,5) and (11,-5) CNTs, respectively. 

