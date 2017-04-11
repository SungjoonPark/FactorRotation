# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import factor_rotation_gpu._gpa_rotation as gr
import numpy as np
import tensorflow as tf

__all__=[]

def rotate_factors(A, method, dtype, *method_args, **algorithm_kwargs):
    if 'algorithm' in algorithm_kwargs:
        algorithm = algorithm_kwargs['algorithm']
        algorithm_kwargs.pop('algorithm')
    else:
        algorithm = 'gpa'
    assert not 'rotation_method' in algorithm_kwargs, 'rotation_method cannot be provided as keyword argument'
    L=None
    T=None
    ff=None
    vgQ=None
    p,k = A.shape
    #set ff or vgQ to appropriate objective function, compute solution using recursion or analytically compute solution
    if method == 'CF':
        assert len(method_args)==2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method=method_args[1]
        assert rotation_method in ['orthogonal','oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        kappa = method_args[0]
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: gr.CF_objective(L=L,A=A,T=T,
                                                               kappa=kappa,
                                                               rotation_method=rotation_method,
                                                               return_gradient=True, 
                                                               dtype=dtype)
    elif method == 'oblimin':
        assert len(method_args)==2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method=method_args[1]
        assert rotation_method in ['orthogonal','oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        gamma = method_args[0]
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: gr.oblimin_objective(L=L,A=A,T=T,
                                                                    gamma=gamma,
                                                                    return_gradient=True,
                                                                    dtype=dtype)   
    elif method == 'quartimax':
        return rotate_factors(A, 'CF', dtype, 0, 'orthogonal', **algorithm_kwargs)
    elif method == 'varimax':
        return rotate_factors(A, 'CF', dtype, 1./p, 'orthogonal', **algorithm_kwargs)
    elif method == 'parsimax':
        return rotate_factors(A, 'CF', dtype, float(k-1)/(p+k-2), 'orthogonal', **algorithm_kwargs)    
    elif method == 'parsimony':
        return rotate_factors(A, 'CF', dtype, 1, 'orthogonal', **algorithm_kwargs)    
    
    elif method == 'quartimin':
        return rotate_factors(A, 'CF', dtype, 0, 'oblique', **algorithm_kwargs)
    elif method == 'covarimin':
        return rotate_factors(A, 'CF', dtype, 1./p, 'oblique', **algorithm_kwargs) 
    elif method == 'parsimax_oblique':
        return rotate_factors(A, 'CF', dtype, float(k-1)/(p+k-2), 'oblique', **algorithm_kwargs)    
    elif method == 'parsimony_oblique':
        return rotate_factors(A, 'CF', dtype, 1, 'oblique', **algorithm_kwargs)    

    #compute L and T if not already done
    if T is None:
        with tf.Graph().as_default():
            with tf.Session() as sess:

                    A = tf.Variable(A, dtype=dtype);
                    sess.run(tf.global_variables_initializer())
                    L, phi, T, s, f = sess.run(gr.GPA(A, dtype, vgQ=vgQ, ff=ff, rotation_method=rotation_method, **algorithm_kwargs))
    if L is None:
        assert T is not None, 'Cannot compute L without T'
        L=rotateA(A,T,rotation_method=rotation_method)

    return L, T, s, f

