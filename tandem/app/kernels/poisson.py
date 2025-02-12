#!/usr/bin/env python3

from yateto import *

def add(generator, dim, nbf, Nbf, nq, Nq, petsc_alignment):
    J = Tensor('J', (Nq,))
    G = Tensor('G', (dim, dim, Nq))
    K = Tensor('K', (Nbf,))
    K_Q = Tensor('K_Q', (Nq,))
    W = Tensor('W', (Nq,))
    E = Tensor('E', (Nbf, Nq))
    Em = Tensor('Em', (Nq, Nbf))
    D_xi = Tensor('D_xi', (Nbf, dim, Nq))
    D_x = Tensor('D_x', D_xi.shape())
    A = Tensor('A', (Nbf, Nbf))
    matMinv = Tensor('matMinv', (Nbf, Nbf))

    generator.add('project_K', K['p'] <= matMinv['pk'] * K_Q['q'] * Em['qk'] * W['q'])

    generator.add('assembleVolume', [ 
        D_x['kiq'] <= G['eiq'] * D_xi['keq'],
        A['kl'] <= J['q'] * W['q'] * K['m'] * Em['qm'] * D_x['kiq'] * D_x['liq']
    ])

    g = [Tensor('g({})'.format(x), (dim, dim, nq)) for x in range(2)]
    n = Tensor('n', (dim, nq))
    n_unit_q = Tensor('n_unit_q', (dim, nq))
    nl = Tensor('nl', (nq,))
    w = Tensor('w', (nq,))
    e = [Tensor('e({})'.format(x), (Nbf, nq)) for x in range(2)]
    em = [Tensor('em({})'.format(x), (nq, Nbf)) for x in range(2)]
    d_xi = [Tensor('d_xi({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    d_x = [Tensor('d_x({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    a = [[Tensor('a({},{})'.format(x, y), (Nbf, Nbf)) for y in range(2)] for x in range(2)]
    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]
    c1 = [Scalar('c1{}'.format(x)) for x in range(2)]
    c2 = [Scalar('c2{}'.format(x)) for x in range(2)]

    def surface(x, y):
        return a[x][y]['kl'] <= c0[y] * w['q'] * d_x[x]['kiq'] * n['iq'] * e[y]['lq'] + \
                                c1[x] * w['q'] * d_x[y]['liq'] * n['iq'] * e[x]['kq'] + \
                                c2[abs(y-x)] * w['q'] * e[x]['kq'] * e[y]['lq'] * nl['q']

    surfaceKernelsLocal = [
        d_x[0]['kiq'] <= K['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        surface(0, 0)
    ]
    surfaceKernelsNeighbour = [
        d_x[1]['kiq'] <= K['m'] * em[1]['qm'] * g[1]['eiq'] * d_xi[1]['keq'],
        surface(0, 1),
        surface(1, 0),
        surface(1, 1)]
    generator.add('assembleFacetLocal', surfaceKernelsLocal)
    generator.add('assembleFacetNeighbour', surfaceKernelsNeighbour)

    b = Tensor('b', (Nbf,), alignStride=petsc_alignment)
    F_Q = Tensor('F_Q', (Nq,))
    generator.add('rhsVolume', b['k'] <= b['k'] + J['q'] * W['q'] * E['kq'] * F_Q['q'])

    f_q = Tensor('f_q', (nq,))
    generator.add('rhsFacet',
        b['k'] <= b['k'] + \
            c1[0] * w['q'] * K['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'] * n['iq'] * f_q['q'] + \
            c2[0] * w['q'] * e[0]['kq'] * nl['q'] * f_q['q'])

    # traction
    u = [Tensor('u({})'.format(x), (Nbf,), alignStride=petsc_alignment) for x in range(2)]
    k = [Tensor('k({})'.format(x), (Nbf,)) for x in range(2)]
    grad_u = Tensor('grad_u', (dim, nq))
    generator.add('grad_u', [
        d_x[0]['kiq'] <= k[0]['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        d_x[1]['kiq'] <= k[1]['m'] * em[1]['qm'] * g[1]['eiq'] * d_xi[1]['keq'],
        grad_u['pq'] <= 0.5 * (d_x[0]['lpq'] * u[0]['l'] + d_x[1]['lpq'] * u[1]['l']) +
                       c0[0] * (e[0]['lq'] * u[0]['l'] - e[1]['lq'] * u[1]['l'] - f_q['q']) * n_unit_q['pq']    ])

    generator.add('grad_u_bnd', [
        d_x[0]['kiq'] <= k[0]['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        grad_u['pq'] <= d_x[0]['lpq'] * u[0]['l'] +
                        c0[0] * (e[0]['lq'] * u[0]['l'] - f_q['q']) * n_unit_q['pq']
    ])

    # derivative of the gradient
    Dgrad_u_Du = Tensor('Dgrad_u_Du', (Nbf, dim, nq))
    generator.add('Dgrad_u_Du', [
        d_x[0]['kiq'] <= k[0]['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        d_x[1]['kiq'] <= k[1]['m'] * em[1]['qm'] * g[1]['eiq'] * d_xi[1]['keq'],
        Dgrad_u_Du['kpq'] <= 0.5 * (d_x[0]['kpq'] + d_x[1]['kpq']) +
                       c0[0] * (e[0]['kq'] - e[1]['kq']) * n_unit_q['pq']
    ])

    generator.add('Dgrad_u_Du_bnd', [
        d_x[0]['kiq'] <= k[0]['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        Dgrad_u_Du['kpq'] <= d_x[0]['kpq'] +
                       c0[0] * e[0]['kq'] * n_unit_q['pq']
    ])
   
 


