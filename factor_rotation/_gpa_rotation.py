from __future__ import division
import tensorflow as tf


def GPA(A, dtype, ff=None, vgQ=None, T=None, max_tries=10000, rotation_method='orthogonal', tol=1e-09):

    # pre processing
    tol = tf.cast(tol, dtype=dtype)

    rotation_method = tf.constant(rotation_method)
    method_assert = tf.Assert(tf.logical_or(tf.equal(rotation_method, "orthogonal"),
                                            tf.equal(rotation_method, "oblique")), [rotation_method])
    with tf.control_dependencies([method_assert]):
        rotation_method = tf.identity(rotation_method)

    p, k = A.get_shape().as_list()
    # p = tf.cast(p, 'int32')
    # k = tf.cast(k, 'int32')
    if T is None:
        T = tf.eye(k, dtype=dtype)

    # pre processing for iteration
    def pre_ortho(A, T):
        Ti = tf.transpose(T)
        L = tf.matmul(A, T)
        f, Gq = vgQ(L=L, A=A, T=T)
        G = tf.matmul(tf.transpose(A), Gq)
        return Ti, L, f, Gq, G, L

    def pre_obli(A, T):
        Ti = tf.matrix_inverse(T)
        L = tf.matmul(A, tf.transpose(Ti))
        f, Gq = vgQ(L=L, A=A, T=T)
        G = -1. * tf.transpose(tf.matmul(tf.matmul(tf.transpose(L), Gq), Ti))
        return Ti, L, f, Gq, G, L

    Ti, L, f, Gq, G, L = tf.cond(tf.equal(rotation_method, 'orthogonal'),
                                 lambda: pre_ortho(A, T),
                                 lambda: pre_obli(A, T))
    # outer iteration
    al = tf.constant(.01, dtype=dtype)

    i_try = 0
    Gp = tf.cond(tf.equal(rotation_method, 'orthogonal'),
                 lambda: G - tf.matmul(T, ((tf.matmul(tf.transpose(T), G)) + tf.transpose(tf.matmul(tf.transpose(T), G))) / 2.),
                 lambda: G - tf.matmul(T, tf.diag(tf.reduce_sum(T*G, 0))))

    s = tf.sqrt(tf.reduce_sum(Gp**2, [0, 1]))
    Tt = T
    ft = f

    # outer_loop
    def outer_cond(i_try, Gp, al, s, f, T, Ti, Gq, G, L, ft, Tt):
        return tf.logical_and(tf.logical_and( s > tol, i_try < max_tries), al >= 1e-16)

    def outer_body(i_try, Gp, al, s, f, T, Ti, Gq, G, L, ft, Tt):
        i_try += 1
        Gp = tf.cond(tf.equal(rotation_method, 'orthogonal'),
                     lambda: G - tf.matmul(T, ((tf.matmul(tf.transpose(T), G)) + tf.transpose(tf.matmul(tf.transpose(T), G))) / 2.),
                     lambda: G - tf.matmul(T, tf.diag(tf.reduce_sum(T*G, 0))))
        s = tf.sqrt(tf.reduce_sum(Gp**2, [0, 1]))

        # inner loop
        j_try = 0
        al *= 2
        ft = f
        al_old = al

        def inner_cond(j_try, Gp, al, al_old, s, f, T, ft, Ti, Gq, G, Tt, L):
            return tf.logical_and(ft >= f-.5*s**2*al_old , j_try < 100)

        def inner_body(j_try, Gp, al, al_old, s, f, T, ft, Ti, Gq, G, Tt, L):
            j_try += 1
            X = T - al * Gp

            def inner_orth(X):
                D, U, V = tf.svd(X, full_matrices=True)
                Tt = tf.matmul(U, tf.transpose(V))
                Ti = tf.transpose(Tt)
                L = tf.matmul(A, Tt)
                ft, Gq = vgQ(L=L, A=A, T=T)
                return ft, Gq, Tt, Ti, L

            def inner_obli(X):
                v = 1. / tf.sqrt(tf.reduce_sum(X**2, 0))
                Tt = tf.matmul(X, tf.diag(v))
                Ti = tf.matrix_inverse(Tt)
                L = tf.matmul(A, tf.transpose(Ti))
                ft, Gq = vgQ(L=L, A=A, T=T)
                return ft, Gq, Tt, Ti, L

            ft, Gq, Tt, Ti, L = tf.cond(tf.equal(rotation_method, 'orthogonal'),
                                        lambda: inner_orth(X),
                                        lambda: inner_obli(X))

            ft = tf.Print(ft, [s, i_try, j_try, al, ft, f-.5*s**2*al], summarize=100)
            al_old = al
            al = tf.cond(ft < f-.5*s**2*al, lambda: tf.identity(al), lambda: al / 2)

            return j_try, Gp, al, al_old, s, f, T, ft, Ti, Gq, G, Tt, L

        j_try, Gp, al, al_old, s, f, T, ft, Ti, Gq, G, Tt, L \
            = tf.while_loop(inner_cond, inner_body, [j_try, Gp, al, al_old, s, f, T, ft, Ti, Gq, G, Tt, L])
        # inner_loop
        T = Tt
        f = ft
        G = tf.cond(tf.equal(rotation_method, 'orthogonal'),
                    lambda: tf.matmul(tf.transpose(A), Gq),
                    lambda: -1. * tf.transpose(tf.matmul(tf.matmul(tf.transpose(L), Gq), Ti)))

        return i_try, Gp, al, s, f, T, Ti, Gq, G, L, ft, Tt

    i_try, Gp, al, s, f, T, Ti, Gq, G, L, ft, Tt \
        = tf.while_loop(outer_cond, outer_body, [i_try, Gp, al, s, f, T, Ti, Gq, G, L, ft, Tt])
    # outer_loop

    return rotateA(A, T, rotation_method=rotation_method), tf.matmul(tf.transpose(T), T), T, s, f


def rotateA(A, T, rotation_method='orthogonal'):
    L = tf.case({
        tf.equal(rotation_method, "orthogonal"):
            lambda: tf.matmul(A, T),
        tf.equal(rotation_method, "oblique"):
            lambda: tf.matmul(A, tf.matrix_inverse(tf.transpose(T)))}, default=lambda: tf.matmul(A, T), exclusive=True)

    return L


def CF_objective(L=None, A=None, T=None, kappa=0,
                 rotation_method='orthogonal', dtype='float32'):

    kappa = tf.cast(kappa, dtype)
    zero = tf.cast(0, dtype)
    one = tf.cast(1, dtype)
    cls_zero8 = tf.cast(1e-08, dtype)
    cls_zero5 = tf.cast(1e-05, dtype)

    if L is None:
        L = rotateA(A, T, rotation_method=rotation_method)

    p, k = L.get_shape().as_list()
    p = tf.cast(p, 'int32')
    k = tf.cast(k, 'int32')
    L2 = L**2

    kap_1 = tf.less_equal(tf.abs(kappa - one), tf.add(cls_zero8, cls_zero5*tf.abs(one)))
    X = tf.cond(kap_1,
                lambda: tf.zeros([p, k], dtype=dtype),
                lambda: (1.-kappa) * tf.matmul(L2, tf.ones([k, k], dtype=dtype) - tf.eye(k, dtype=dtype)))

    kap_0 = tf.less_equal(tf.abs(kappa - zero), tf.add(cls_zero8, cls_zero5*tf.abs(zero)))
    X += tf.cond(kap_0,
                 lambda: tf.zeros([p, k], dtype=dtype),
                 lambda: kappa * (tf.tile(tf.reshape(tf.reduce_sum(L2, 0), (1, k)), (p, 1)) - L2))

    phi = tf.reduce_sum(L2 * X, [0, 1])/4.
    Gphi = L*X

    return phi, Gphi


def oblimin_objective(L=None, A=None, T=None, gamma=0,
                      rotation_method='orthogonal', dtype='float32'):

    gamma = tf.cast(gamma, dtype)
    zero = tf.cast(0, dtype)
    one = tf.cast(1, dtype)
    cls_zero8 = tf.cast(1e-08, dtype)
    cls_zero5 = tf.cast(1e-05, dtype)

    if L is None:
        L = rotateA(A, T, rotation_method=rotation_method)

    p, k = L.get_shape().as_list()
    p = tf.cast(p, 'int32')
    k = tf.cast(k, 'int32')
    p_fl = tf.cast(p, dtype)
    L2 = L**2

    N = tf.ones((k, k), dtype=dtype) - tf.eye(k, dtype=dtype)

    kap_0 = tf.less_equal(tf.abs(gamma - zero), tf.add(cls_zero8, cls_zero5*tf.abs(zero)))
    X = tf.cond(kap_0,
                lambda: tf.matmul(L2, N),
                lambda: tf.tile(tf.reshape((-gamma/p_fl) *
                                           tf.reduce_sum(tf.matmul(L2, N), 0), (1, k)), (p, 1)) + tf.matmul(L2, N))

    phi = tf.reduce_sum(L2 * X, [0, 1]) / 4.
    Gphi = L*X

    return phi, Gphi
