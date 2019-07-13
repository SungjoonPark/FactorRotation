import torch

def GPA(A, vgQ=None, T=None, max_tries=30000, rotation_method='orthogonal', tol=1e-9,
        dtype=torch.float64, device=torch.device("cuda")):
    if rotation_method not in ['orthogonal', 'oblique']:
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    if vgQ is None:
        raise ValueError('vgQ should be provided')
    if T is None:
        T = torch.eye(A.size()[1], dtype=dtype, device=device)

    #pre processing for iteration: initialize f and G
    al = torch.tensor(.01, dtype=dtype, device=device)
    table = [] # for logging
    if rotation_method == 'orthogonal':
        L = torch.matmul(A, T)
        f, Gq = vgQ(L=L)
        G = torch.matmul(torch.transpose(A, 0, 1), Gq)
    else: #i.e. rotation_method == 'oblique'
        Ti = torch.inverse(T)
        L = torch.matmul(A, torch.transpose(Ti, 0, 1))
        f, Gq = vgQ(L=L)
        G = -torch.transpose(torch.matmul(torch.matmul(torch.transpose(L, 0, 1), Gq), Ti), 0, 1)

    #iteration
    for i_try in range(0, max_tries):

        #determine Gp
        if rotation_method == 'orthogonal':
            M = torch.matmul(torch.transpose(T, 0, 1), G)
            S = (M + torch.transpose(M, 0, 1)) / 2
            Gp = G - torch.matmul(T, S)
        else: #i.e. if rotation_method == 'oblique':
            Gp = G - torch.matmul(T, torch.diag(torch.sum(T * G, 0)))
        s = torch.norm(Gp, p='fro')

        # for logging
        table.append([i_try, s, al])
        print('iter', i_try, '::\ts:', round(s.item(), 7), '; alpha:', al.item())

        #if we are close stop
        if s < tol or al <= 1e-16: break

        #update T
        al = 2*al
        for i in range(100):
            #determine Tt
            X = T - al*Gp
            if rotation_method == 'orthogonal':
                U, D, V = torch.svd(X)
                Tt = torch.matmul(U, torch.transpose(V, 0, 1))
            else: #i.e. if rotation_method == 'oblique':
                v = 1 / torch.sqrt(torch.sum(X**2, 0))
                Tt = torch.matmul(X, torch.diag(v))

            #calculate objective using Tt
            if rotation_method == 'orthogonal':
                L = torch.matmul(A, Tt)
                ft, Gq = vgQ(L=L)
            else: #i.e. rotation_method == 'oblique'
                Ti = torch.inverse(Tt)
                L = torch.matmul(A, torch.transpose(Ti, 0, 1))
                ft, Gq = vgQ(L=L)

            #if sufficient improvement in objective -> use this T
            if ft < f - .5*s**2*al: break
            al = al/2

        #post processing for next iteration
        T = Tt
        f = ft
        if rotation_method == 'orthogonal':
            G = torch.matmul(torch.transpose(A, 0, 1), Gq)
        else: #i.e. rotation_method == 'oblique'
            G = -torch.transpose(torch.matmul(torch.matmul(torch.transpose(L, 0, 1), Gq), Ti), 0, 1)

    #post processing
    Th = T
    Lh = rotateA(A, T, rotation_method=rotation_method)
    Phi = torch.matmul(torch.transpose(T, 0, 1), T)

    return Lh, Phi, Th, table


def rotateA(A, T, rotation_method='orthogonal'):
    if rotation_method == 'orthogonal':
        L = torch.matmul(A, T)
    elif rotation_method == 'oblique':
        L = torch.matmul(A, torch.inverse(torch.transpose(T, 0, 1)))
    else: #i.e. if rotation_method == 'oblique':
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    return L


def CF_objective(L=None, A=None, T=None, kappa=0, rotation_method='orthogonal',
                 return_gradient=True, dtype=torch.float64, device='cuda'):
    assert 0<=kappa<=1, "Kappa should be between 0 and 1"
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A, T, rotation_method=rotation_method)
    p, k = L.size()
    L2=L**2
    X=None
    if not torch.abs(torch.tensor(kappa - 1, dtype=dtype, device=device)) <= (1e-08 + 1e-05*torch.abs(torch.tensor(1, dtype=dtype, device=device))):
        N = torch.ones((k,k), dtype=dtype, device=device) - torch.eye(k, dtype=dtype, device=device)
        X = (1 - kappa) * torch.matmul(L2, N)
    if not torch.abs(torch.tensor(kappa - 0, dtype=dtype, device=device)) <= (1e-08 + 1e-05*torch.abs(torch.tensor(0, dtype=dtype, device=device))):
        if X is None:
            X = kappa * (torch.sum(L2, dim=0, keepdim=True).repeat(p, 1) - L2)
        else:
            X += kappa * (torch.sum(L2, dim=0, keepdim=True).repeat(p, 1) - L2)
    phi = torch.sum(L2 * X)/4
    if return_gradient:
        Gphi = L * X
        return phi, Gphi
    else:
        return phi
