# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import numpy as np

import ase.units as u
from ase.parallel import rank, parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap


class Albrecht(ResonantRaman):
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        all from ResonantRaman.__init__
        combinations: int
            Combinations to consider for multiple excitations.
            Default is 1, possible 2
        skip: int
            Number of first transitions to exclude. Default 0,
            recommended: 5 for linear molecules, 6 for other molecules
        """
        self.combinations = kwargs.pop('combinations', 1)
        self.skip = kwargs.pop('skip', 0)
        ResonantRaman.__init__(self, *args, **kwargs)
        
    def read(self, method='standard', direction='central'):
        ResonantRaman.read(self, method, direction)

        # single transitions and their occupation
        om_Q = self.om_Q[self.skip:]
        om_v = om_Q
        ndof = len(om_Q)
        n_vQ = np.eye(ndof, dtype=int)
        if self.combinations > 1:
            # double transitions and their occupation
            om_v = list(om_v)
            n_vQ = list(n_vQ)
            for i in range(ndof):
                n_Q = np.zeros(ndof, dtype=int)
                n_Q[i] = 1
                for j in range(i, ndof):
                    n_vQ.append(n_Q.copy())
                    n_vQ[-1][j] += 1
                    om_v.append(np.dot(n_vQ[-1], om_Q))
            om_v = np.array(om_v)
            n_vQ = np.array(n_vQ)
            
        self.om_v = om_v
##        print('self.om_v', self.om_v)
        self.n_vQ = n_vQ
##        print('self.n_vQ', self.n_vQ)

    def Huang_Rhys_factors(self, forces_r):
        """Evaluate Huang-Rhys factors derived from forces."""
        self.timer.start('Huang-Rhys')
        assert(len(forces_r.flat) == self.ndof)

        # solve the matrix equation for the equilibrium displacements
        X_q = np.linalg.solve(self.im[:, None] * self.H * self.im,
                              forces_r.flat * self.im)
        d_Q = np.dot(self.modes, X_q)

        # Huang-Rhys factors S
        s = 1.e-20 / u.kg / u.C / u._hbar**2  # SI units
        self.timer.stop('Huang-Rhys')
        return s * d_Q**2 * self.om_Q / 2.

    def meA(self, omega, gamma=0.1, ml=range(16)):
        """Evaluate Albrecht A term.

        Returns
        -------
        Full Albrecht A matrix element. Unit: e^2 Angstrom^2 / eV
        """
        self.read()

        self.timer.start('AlbrechtA')

        if not hasattr(self, 'fco'):
            self.fco = FranckCondonOverlap()

        om = omega + 1j * gamma
        # excited state forces
        F_pr = self.exF_rp.T

        m_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        for p, energy in enumerate(self.ex0E_p):
            S_Q = self.Huang_Rhys_factors(F_pr[p])
            energy_Q = energy - self.om_Q * S_Q
            me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())

            wm_Q = np.zeros((self.ndof), dtype=complex)
            wp_Q = np.zeros((self.ndof), dtype=complex)
            for m in ml:
                self.timer.start('0mm1')
                fco_Q = self.fco.direct0mm1(m, S_Q)
                self.timer.stop('0mm1')
                
                self.timer.start('weight_Q')
                wm_Q += fco_Q / (energy_Q + m * self.om_Q - om)
                wp_Q += fco_Q / (energy_Q + (m - 1) * self.om_Q + om)
                self.timer.stop('weight_Q')
            self.timer.start('einsum')
            m_Qcc += np.einsum('a,bc->abc', wm_Q, me_cc)
            m_Qcc += np.einsum('a,bc->abc', wp_Q, me_cc.T.conj())
            self.timer.stop('einsum')
                
        self.timer.stop('AlbrechtA')
        return m_Qcc

    def meA_2(self, omega, gamma=0.1, ml=range(16)):
        """Evaluate Albrecht A term.

        Returns
        -------
        Full Albrecht A matrix element. Unit: e^2 Angstrom^2 / eV
        """
        self.read()

        self.timer.start('AlbrechtA')

        if not hasattr(self, 'fco'):
            self.fco = FranckCondonOverlap()

        om = omega + 1j * gamma
        # excited state forces
        F_pr = self.exF_rp.T

        m_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        for p, energy in enumerate(self.ex0E_p):
            S_Q = self.get_Huang_Rhys_factors(F_pr[p])
            # scattered light
            omS_v = om - self.om_v
            # relaxed excited state energy
            n_vQ = np.where(self.n_vQ > 0, 1, 0)
            energy_v = energy - n_vQ.dot(self.om_Q * S_Q)
            
            me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())

            wm_Q = np.zeros((self.ndof), dtype=complex)
            wp_Q = np.zeros((self.ndof), dtype=complex)
            for m in ml:
                self.timer.start('0mm1/2')
                fco1_Q = self.fco.direct0mm2(m, S_Q)
                if (self.n_vQ > 1).any():
                    fco2_Q = self.fco.direct0mm2(m, S_Q)
                self.timer.stop('0mm1/2')
                
                self.timer.start('weight_Q')
                wm_Q += fco_Q / (energy_Q + m * self.om_Q - om)
                wp_Q += fco_Q / (energy_Q + (m - 1) * self.om_Q + om)
                self.timer.stop('weight_Q')
            self.timer.start('einsum')
            m_Qcc += np.einsum('a,bc->abc', wm_Q, me_cc)
            m_Qcc += np.einsum('a,bc->abc', wp_Q, me_cc.T.conj())
            self.timer.stop('einsum')
                
        self.timer.stop('AlbrechtA')
        return m_Qcc

    def meBC(self, omega, gamma=0.1, ml=[1],
             term='BC'):
        """Evaluate Albrecht B and/or C term(s)."""
        self.read()
        # we need the overlaps
        assert(self.overlap)

        self.timer.start('AlbrechtBC')

        if not hasattr(self, 'fco'):
            self.fco = FranckCondonOverlap()

        # excited state forces
        F_pr = self.exF_rp.T

        m_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        for p, energy in enumerate(self.ex0E_p):
            S_Q = self.get_Huang_Rhys_factors(F_pr[p])

            for m in ml:
                self.timer.start('Franck-Condon overlaps')
                fc1mm1_Q = self.fco.direct(1, m, S_Q)
                fc0mm02_Q = self.fco.direct(0, m, S_Q)
                fc0mm02_Q += np.sqrt(2) * self.fco.direct0mm2(m, S_Q)
##                print(m, fc1mm1_r[-1], fc0mm02_r[-1])
                self.timer.stop('Franck-Condon overlaps')

                self.timer.start('me dervivatives')
                dm_rc = []
                r = 0
                for a in self.indices:
                    for i in 'xyz':
                        dm_rc.append(
                            (self.expm_rpc[r, p] - self.exmm_rpc[r, p]) *
                            self.im[r])
                        print('pm=', self.expm_rpc[r, p], self.exmm_rpc[r, p])
                        r += 1
                dm_rc = np.array(dm_rc) / (2 * self.delta)
                self.timer.stop('me dervivatives')

                self.timer.start('map to modes')
                # print('dm_rc[2], dm_rc[5]', dm_rc[2], dm_rc[5])
                print('dm_rc=', dm_rc)
                dm_rc = np.dot(dm_rc.T, self.modes.T).T
                print('dm_rc[-1][2]', dm_rc[-1][2])
                self.timer.stop('map to modes')

                self.timer.start('multiply')
                # me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())
                for r in range(self.ndof):
                    if 'B' in term:
                        # XXXX
                        denom = (1. /
                                 (energy + m * 0 * self.om_Q[r] -
                                  omega - 1j * gamma))
                        # ok print('denom=', denom)
                        m_rcc[r] += (np.outer(dm_rc[r],
                                              self.ex0m_pc[p].conj()) *
                                     fc1mm1_r[r] * denom)
                        if r == 5:
                            print('m_rcc[r]=', m_rcc[r][2, 2])
                        m_rcc[r] += (np.outer(self.ex0m_pc[p],
                                              dm_rc[r].conj()) *
                                     fc0mm02_r[r] * denom)
                    if 'C' in term:
                        denom = (1. /
                                 (energy + (m - 1) * self.om_Q[r] +
                                  omega + 1j * gamma))
                        m_rcc[r] += (np.outer(self.ex0m_pc[p],
                                              dm_rc[r].conj()) *
                                     fc1mm1_r[r] * denom)
                        m_rcc[r] += (np.outer(dm_rc[r],
                                              self.ex0m_pc[p].conj()) *
                                     fc0mm02_r[r] * denom)
                self.timer.stop('multiply')
        print('m_rcc[-1]=', m_rcc[-1][2, 2])

        self.timer.start('pre_r')
        with np.errstate(divide='ignore'):
            pre_r = np.where(self.om_Q > 0,
                             np.sqrt(u._hbar**2 / 2. / self.om_Q), 0)
            # print('BC: pre_r=', pre_r)
        for r, p in enumerate(pre_r):
            m_rcc[r] *= p
        self.timer.stop('pre_r')
        self.timer.stop('AlbrechtBC')
        return m_rcc

    def electronic_me_Qcc(self, omega, gamma):
        """Evaluate an electronic matric element."""
        if self.approximation.lower() == 'albrecht a':
            Vel_Qcc = self.meA(omega, gamma)
        elif self.approximation.lower() == 'albrecht bc':
            Vel_Qcc = self.meBC(omega, gamma)
        elif self.approximation.lower() == 'albrecht b':
            Vel_Qcc = self.meBC(omega, gamma, term='B')
        elif self.approximation.lower() == 'albrecht c':
            Vel_Qcc = self.meBC(omega, gamma, term='C')
        else:
            raise NotImplementedError(
                'Approximation {0} not implemented. '.format(
                    self.approximation) +
                'Please use "Albrecht A/B/C".')

        Vel_Qcc *= u.Hartree * u.Bohr  # e^2Angstrom^2 / eV -> Angstrom^3
        # divide through pre-factor
        with np.errstate(divide='ignore'):
            Vel_Qcc *= np.where(self.vib01_Q > 0,
                                1. / self.vib01_Q, 0)[:, None, None]

        return Vel_Qcc

    def matrix_element(self, omega, gamma):
        self.read()
        V_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        if self.approximation.lower() in ['profeta', 'placzek',
                                          'p-p', 'placzekalpha']:
            me_Qcc = self.electronic_me_Qcc(omega, gamma)
            for Q, vib01 in enumerate(self.vib01_Q):
                V_Qcc[Q] = me_Qcc[Q] * vib01
        elif self.approximation.lower() == 'albrecht a':
            V_Qcc += self.me_AlbrechtA(omega, gamma)
        elif self.approximation.lower() == 'albrecht b':
            V_Qcc += self.me_AlbrechtBC(omega, gamma, term='B')
        elif self.approximation.lower() == 'albrecht c':
            V_Qcc += self.me_AlbrechtBC(omega, gamma, term='C')
        elif self.approximation.lower() == 'albrecht bc':
            V_Qcc += self.me_AlbrechtBC(omega, gamma)
        elif self.approximation.lower() == 'albrecht':
            V_Qcc += self.me_AlbrechtA(omega, gamma)
            V_Qcc += self.me_AlbrechtBC(omega, gamma)
        elif self.approximation.lower() == 'albrecht+profeta':
            raise NotImplementedError('not useful')
            V_Qcc += self.get_matrix_element_AlbrechtA(omega, gamma)
            V_Qcc += self.get_matrix_element_Profeta(omega, gamma)
        else:
            raise NotImplementedError(
                'Approximation {0} not implemented. '.format(
                    self.approximation) +
                'Please use "Profeta", "Placzek", "Albrecht A/B/C/BC", ' +
                'or "Albrecht".')

        return V_Qcc
    
    def summary(self, omega=0, gamma=0,
                method='standard', direction='central',
                log=sys.stdout):
        """Print summary for given omega [eV]"""
        hnu = self.get_energies(method, direction)
        intensities = self.absolute_intensity(omega, gamma)

        if isinstance(log, str):
            log = paropen(log, 'a')

        parprint('-------------------------------------', file=log)
        parprint(' excitation at ' + str(omega) + ' eV', file=log)
        parprint(' gamma ' + str(gamma) + ' eV', file=log)
        parprint(' approximation:', self.approximation, file=log)
        parprint(' Mode    Frequency        Intensity', file=log)
        parprint('  #    meV     cm^-1      [A^4/amu]', file=log)
        parprint('-------------------------------------', file=log)
        for n, e in enumerate(self.om_v):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            parprint('%3d %6.1f%s  %7.1f%s  %9.1f' %
                     (n, 1000 * e, c, e / u.invcm, c, intensities[n]),
                     file=log)
        parprint('-------------------------------------', file=log)
        parprint('Zero-point energy: %.3f eV' % self.get_zero_point_energy(),
                 file=log)
