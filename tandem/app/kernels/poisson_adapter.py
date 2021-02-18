from yateto import *

def add(generator, dim, nbf_fault, Nbf_element, nq):
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq,))

    slip = Tensor('slip', (nbf_fault,))
    slip_q = Tensor('slip_q', (nq,))

    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]

    generator.add('evaluate_slip', slip_q['q'] <= e_q_T['qp'] * slip['p'])

    grad_u = Tensor('grad_u', (dim, nq))
    n_unit_q = Tensor('n_unit_q', (dim, nq))
    traction = Tensor('traction', (nbf_fault,))
    generator.add('evaluate_traction', traction['p'] <= minv['rp'] * e_q_T['qr'] * w['q'] * \
                                                        grad_u['kq'] * n_unit_q['kq'])


    dtau_du = Tensor('dtau_du', (nbf_fault, Nbf_element))
    Dgrad_u_Du = Tensor('Dgrad_u_Du', (Nbf_element, dim, nq))
    generator.add('evaluate_derivative_traction_dU', dtau_du['pl'] <= minv['rp'] * e_q_T['qr'] * w['q'] * \
                                                        Dgrad_u_Du['lkq'] * n_unit_q['kq'])


    dtau_dS = Tensor('dtau_dS', (nbf_fault, nbf_fault))
    generator.add('evaluate_derivative_traction_dS', dtau_dS['pj'] <= minv['rp'] * e_q_T['qr'] * w['q'] * \
                                                         c0[0] * e_q_T['qj'] * n_unit_q['kq'] * n_unit_q['kq'])
