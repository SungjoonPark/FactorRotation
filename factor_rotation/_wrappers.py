from __future__ import division
from __future__ import print_function

import factor_rotation._gpa_rotation as gr
import torch

def rotate_factors(A, method, dtype=torch.float64, device=torch.device("cuda"), *method_args, **algorithm_kwargs):
    assert 'rotation_method' not in algorithm_kwargs, 'rotation_method cannot be provided as keyword argument'
    L = None
    T = None
    ff = None
    vgQ = None
    A = torch.tensor(A, dtype=dtype, device=device)
    p, k = A.size()

    #set ff or vgQ to appropriate objective function, compute solution using recursion or analytically compute solution
    if method == 'CF':
        assert len(method_args)==2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method=method_args[1]
        assert rotation_method in ['orthogonal','oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        kappa = method_args[0]
        vgQ = lambda L=None, A=None, T=None: gr.CF_objective(L=L, A=A, T=T, kappa=kappa,
                                                           rotation_method=rotation_method,
                                                           return_gradient=True,
                                                           dtype=dtype, device=device)

    elif method == 'quartimax':
        return rotate_factors(A, 'CF', dtype, device, 0.0, 'orthogonal', **algorithm_kwargs)
    elif method == 'varimax':
        return rotate_factors(A, 'CF', dtype, device, 1.0/p, 'orthogonal', **algorithm_kwargs)
    elif method == 'parsimax':
        return rotate_factors(A, 'CF', dtype, device, float(k-1)/(p+k-2), 'orthogonal', **algorithm_kwargs)
    elif method == 'parsimony':
        return rotate_factors(A, 'CF', dtype, device, 1.0, 'orthogonal', **algorithm_kwargs)

    elif method == 'quartimin':
        return rotate_factors(A, 'CF', dtype, device, 0.0, 'oblique', **algorithm_kwargs)
    elif method == 'covarimin':
        return rotate_factors(A, 'CF', dtype, device, 1.0/p, 'oblique', **algorithm_kwargs)
    elif method == 'parsimax_oblique':
        return rotate_factors(A, 'CF', dtype, device, float(k-1)/(p+k-2), 'oblique', **algorithm_kwargs)
    elif method == 'parsimony_oblique':
        return rotate_factors(A, 'CF', dtype, device, 1.0, 'oblique', **algorithm_kwargs)

    else:
        raise ValueError('Invalid method')

    #compute L and T if not already done
    L, phi, T, table = gr.GPA(A, vgQ=vgQ, T=None, rotation_method=rotation_method, tol=1e-5, dtype=dtype, device=device)

    #return
    return L, T
