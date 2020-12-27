from yateto import *
from yateto.memory import CSCMemoryLayout


def add(generator, dim, nbf_fault, nq):
    e_q = Tensor('e_q', (nbf_fault, nq))
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    fault_basis_q = Tensor('fault_basis_q', (dim, dim, nq))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq, ))

    copy_slip = Tensor('copy_slip', (dim - 1, dim),
                       spp={(d - 1, d): '1.0'
                            for d in range(1, dim)},
                       memoryLayoutClass=CSCMemoryLayout)
    slip = Tensor('slip', (dim - 1, nbf_fault))
    slip_q = Tensor('slip_q', (dim, nq))

    generator.add(
        'evaluate_slip', slip_q['pq'] <=
        e_q['lq'] * fault_basis_q['poq'] * slip['nl'] * copy_slip['no'])

    traction_q = Tensor('traction_q', (dim, nq))
    traction = Tensor('traction', (nbf_fault, dim))
    generator.add('evaluate_traction', traction['kp'] <= minv['lk'] * e_q_T['ql'] * w['q'] * \
                                                         traction_q['oq'] * fault_basis_q['opq'])

    n_unit_q = Tensor('n_unit_q', (dim, nq))
    dtau_du = Tensor('dtau_du', (nbf_fault, 2*nbf_fault))
    Dgrad_u_Du = Tensor('Dgrad_u_Du', (dim, nq, 2*nbf_fault))
    generator.add('evaluate_derivative_traction', dtau_du['pl'] <= minv['rp'] * e_q_T['qr'] * w['q'] * \
                                                        Dgrad_u_Du['kql'] * n_unit_q['kq'])

