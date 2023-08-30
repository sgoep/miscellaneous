'''
This is an adaption of the Shearlet transform of the alpha-transform package (https://pypi.org/project/alpha-transform/). In particular
the shearlet transform below can be applied to Tensors on the GPU.
'''

import numpy as np
import torch
from alpha_transform import AlphaShearletTransform as AST

class my_shearlet():
    def __init__(self, M, N, is_subsamp, nscales, to_device):
        # super(my_shearlet, self).__init__()
        
        self.TYPE = 'shear'
        self.M = M
        self.N = N
        if is_subsamp:
            _T = AST(M, N, [0.5]*nscales, subsampled=True, parseval=True, real=False, periodization=True) 
        else:
            _T = AST(M, N, [0.5]*nscales, subsampled=False, parseval=True, real=False, periodization=True) 
        
        self.to_device = to_device
        if to_device:
            self.spectrograms_tensor = [torch.Tensor(spect).to(config.device) for spect in _T.spectrograms]
        else:
            self.spectrograms_tensor = [torch.Tensor(spect) for spect in _T.spectrograms]
        
        self.is_subsampled       = _T.is_subsampled
        self.scales              = [_T.indices[i][0] for i in range(1, len(_T.indices))]
        self.normalization       = _T._AlphaShearletTransform__normalization
        self.num_coeffs          = len(_T.spectrograms)
        self.num_scales          = _T.num_scales
        self.indices             = [idx for idx in _T.indices]
        self.shearlets           = [shear for shear in _T.shearlets]
        self.grid                = _T._grid()
        if _T.is_subsampled:
            self._wrapped_to_index   = [widx for widx in _T._wrapped_to_index]
        self.zero_tensor         = torch.Tensor([0]).to(config.device)
        self.space_norms         = [sn for sn in _T.space_norms]
        # self.vis_idx             = []
        self.thresholds          = torch.Tensor([0.0003*2**(-3*(j[0]+1)/4) for j in _T.indices[1:]]).to(config.device)
        
    def set_idx(self, idx, invis):
        if invis == 'vis':
            self.vis_idx = idx
        else:
            self.inv_idx = idx
    
    def _add_wrapped_to_matrix(self, i, source, target):
        if i != 0:
            indices = tuple(self._wrapped_to_index[i][::-1])
            target[indices[0], indices[1]] += source
        else:
            (x_min, x_max) = self._wrapped_to_index[0][0]
            (y_min, y_max) = self._wrapped_to_index[0][1]
            target[y_min: y_max + 1, x_min: x_max + 1] += source

    def my_fft_shift_tensor(self, A):
        return torch.fft.fftshift(A).flip(0)

    def my_ifft_shift_tensor(self, A):
        return torch.fft.ifftshift(A.flip(0))

    def fft2_tensor(self, F):
        return torch.fft.fft2(F, norm='ortho')

    def ifft2_tensor(self, F):
        return torch.fft.ifft2(F, norm='ortho')
    
    def hard_thresholding(self, coeffs, alpha):

        if len(alpha) > 1 and len(alpha) != self.num_scales:
            raise Exception("Number of parameters and scales do not match!")
        elif len(alpha) == 1:
            c = []
            for ck in coeffs:
                ctmp = ck
                ck[torch.abs(ctmp) <= alpha[0]] = 0
                c.append(ck)
            c[0] = coeffs[0]
        else:
            c = [coeffs[0]]
            for k in range(1, len(coeffs)):
                ctmp  = coeffs[k]
                ctmp[torch.abs(ctmp) <= alpha[self.indices[k][0]]] = 0
                c.append(ctmp)
        
        return c
    
    def thresholding(self, coeffs, alpha):

        # if len(alpha) > 1 and len(alpha) != self.num_scales:
            # raise Exception("Number of parameters and scales do not match!")
        if len(alpha) == 1:
            c = [torch.sgn(ck)*torch.maximum(self.zero_tensor, torch.abs(ck)-alpha[0]) for ck in coeffs]
            c[0] = coeffs[0]
        else:
            c = [coeffs[0]]
            for k in range(1, len(coeffs)):
                ctmp = coeffs[k]
                j    = self.indices[k][0]+1
                tc   = torch.sgn(ctmp)*torch.maximum(self.zero_tensor, torch.abs(ctmp)-alpha[self.indices[k][0]]*2**(-3*j/4))
                c.append(tc)
        
        return c
    
    
    def proj_onto(self, c, idx, invis):
        ' Projection of an image x onto the shearlet coefficients defined in idx '
        if invis == 'vis':
            cproj = [c[0]]
        else:
            if self.to_device:
                cproj = [torch.zeros(c[0].shape).to(config.device)]
            else:
                cproj = [torch.zeros(c[0].shape)]
        for k in range(1, len(c)):
            # print(c[k].is_cuda)
            # print(idx[k-1].is_cuda)
            cproj.append( c[k] * idx[k-1])
        return cproj
    
    
    def transform(self, x):
        if self.is_subsampled:
            xhat = self.my_fft_shift_tensor(self.fft2_tensor(x.squeeze()))
            c_list = []
            (x_min, x_max) = self._wrapped_to_index[0][0]
            (y_min, y_max) = self._wrapped_to_index[0][1]
            c_list.append(self.ifft2_tensor(self.my_ifft_shift_tensor(
                            xhat[y_min:y_max+1,x_min:x_max+1] * self.spectrograms_tensor[0] / self.normalization[0]
                        )))
            
            for i, (spect, indices) in enumerate(zip(self.spectrograms_tensor[1:], self._wrapped_to_index[1:])):
                c = self.ifft2_tensor(self.my_ifft_shift_tensor(
                            xhat[indices[1], indices[0]] * spect / self.normalization[i+1]
                        ))
                c_list.append(c)
        else:
            xhat = self.fft2_tensor(x.squeeze())
            c_list = []
            for (i, spect) in enumerate(self.spectrograms_tensor):
                c  = self.ifft2_tensor(self.my_ifft_shift_tensor(
                            xhat * spect / self.normalization
                        ))
                c_list.append(c)
                    
        return c_list
    
    def adjoint(self, coeffs):
        if self.to_device:
            out_im = torch.zeros([self.N, self.M], dtype=torch.complex128).to(config.device)
        else:
            out_im = torch.zeros([self.N, self.M], dtype=torch.complex128)
            
        if self.is_subsampled:
            for j, (c, spect) in enumerate(zip(coeffs, self.spectrograms_tensor)):
                adj_coeff = self.my_fft_shift_tensor(self.fft2_tensor(c))
                self._add_wrapped_to_matrix(j, adj_coeff * spect * self.normalization[j], out_im)
        
            rec = torch.real(self.ifft2_tensor(self.my_ifft_shift_tensor(out_im)))

        else:
            for i, (c, spect) in enumerate(zip(coeffs, self.spectrograms_tensor)):
                adj_coeff = self.my_fft_shift_tensor(self.fft2_tensor(c))
                out_im += self.ifft2_tensor(adj_coeff * spect * self.normalization)
            rec = torch.real(out_im)
            
        return rec