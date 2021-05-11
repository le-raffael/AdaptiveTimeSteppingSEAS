from yateto import *
from yateto.memory import CSCMemoryLayout


def add(generator, dim, nbf_fault, Nbf_element, nq):
    e_q = Tensor('e_q', (nbf_fault, nq))
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    fault_basis_q = Tensor('fault_basis_q', (dim, dim, nq))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq, ))

    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]

    copy_slip = Tensor('copy_slip', (dim - 1, dim),
                       spp={(d - 1, d): '1.0'
                            for d in range(1, dim)})
#                       memoryLayoutClass=CSCMemoryLayout)
    slip = Tensor('slip', (nbf_fault, dim - 1))
    slip_q = Tensor('slip_q', (dim, nq))

    generator.add(
        'evaluate_slip', slip_q['pq'] <=
        -e_q['lq'] * fault_basis_q['poq'] * slip['ln'] * copy_slip['no'])

    traction_q = Tensor('traction_q', (dim, nq))
    traction = Tensor('traction', (nbf_fault, dim))
    generator.add('evaluate_traction', traction['kp'] <= minv['lk'] * e_q_T['ql'] * w['q'] * \
                                                         traction_q['oq'] * fault_basis_q['opq'])

    dtau_du = Tensor('dtau_du', (nbf_fault, dim, Nbf_element, dim))
    D_traction_q_Du = Tensor('D_traction_q_Du', (Nbf_element, dim, dim, nq))
    generator.add('evaluate_derivative_traction_dU', dtau_du['kprs'] <= minv['lk'] * e_q_T['ql'] * w['q'] * \
                                                         D_traction_q_Du['rsoq'] * fault_basis_q['opq'])

    dtau_dS = Tensor('dtau_dS', (nbf_fault, dim, nbf_fault, dim-1))
    generator.add('evaluate_derivative_traction_dS', dtau_dS['kpln'] <= 
        c0[0] * minv['lk'] * e_q_T['ql'] * w['q'] * 
        e_q['lq'] * fault_basis_q['otq'] * copy_slip['nt'] * 
        fault_basis_q['opq'])
