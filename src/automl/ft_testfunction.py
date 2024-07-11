import numpy as np

def global_1D(x):
    return (1 / (1 + np.exp(-(0.1*x**3 - 0.5*x**2 + 0.2*x )))).flatten()

def global_xD(x):
    if x.shape[0] == 1:
        return global_1D(x)
    else:
        raise NotImplementedError("Only 1D data is supported")

def local_1D(x,f,noise=0.1):
    return np.round((global_1D(x)+(1-global_1D(x))*np.exp(-0.1*f)) + np.random.normal(0,noise,f.shape),2)#*0.5*np.sqrt(f)

def local_xD(x,f,noise=0.1):
    if x.shape[0] == 1:
        return local_1D(x,f,noise)
    else:
        raise NotImplementedError("Only 1D data is supported")