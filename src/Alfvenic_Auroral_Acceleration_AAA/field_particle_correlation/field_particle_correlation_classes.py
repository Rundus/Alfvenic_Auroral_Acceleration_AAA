
import numpy as np
import scipy.integrate
from scipy.integrate import simpson
import spaceToolsLib as stl

class FPCClasses:

    def to_EeV(self, Vel):
        return (0.5 * stl.m_e * np.square(Vel)) / stl.q0

    def to_Vel(self,energy_eV, mass=stl.m_e):
        """Particle speed (m/s) from kinetic energy (eV), non-relativistic."""
        return np.sqrt(2.0 * np.asarray(energy_eV) * stl.q0 / mass)

    def to_vpar_vperp(self, pitch_deg, energy_eV, mass=stl.m_e):
        """Velocity coordinates of each bin centre.

        Returns v_par, v_perp in m/s, each shaped (n_pitch, n_energy).
        """
        a = np.radians(np.asarray(pitch_deg))[:, None]
        v = self.to_Vel(energy_eV, mass)[None, :]
        return v * np.cos(a), v * np.sin(a)

    def instant_FPC_detector_coords(self,energies,pitch_angle,distribution, Epara, mass=stl.m_e):
        '''
        param energies [energy]
        param pitch_angle [pitch]
        param distribution [time, pitch, energy]
        param Epara - time series of E_parallel, downsampled onto ESA timebase  [time]
        Returns C_E_parallel(E,pitch,t)
        '''

        # --- Adjust dimensions of inputs ---
        alpha = np.radians(pitch_angle[None, :, None])
        Engy = energies[None, None, :]
        E_para = Epara[:, None, None]

        # --- Calculate the derivatives ---
        Vmag = np.sqrt(2*Engy*stl.q0/mass)
        df_dE = np.gradient(distribution, energies, axis=2, edge_order=2)
        df_dalpha = np.gradient(distribution, np.radians(pitch_angle), axis=1, edge_order=2)
        df_dvpara = - (1/Vmag) * np.sin(alpha) * df_dalpha + stl.m_e * Vmag * np.cos(alpha) * df_dE

        # --- Calculate the instantaneous FPC (detector coordinates) ---
        vparas = Vmag*np.cos(alpha)
        FPC_instant =  -1*(-1*stl.q0)*(vparas**2/2)*df_dvpara*E_para
        return FPC_instant

    def instant_alternate_FPC_detector_coords(self,energies,pitch_angle,distribution, Epara, mass=stl.m_e):
        '''
                param energies [energy]
                param pitch_angle [pitch]
                param distribution [time, pitch, energy]
                param Epara - time series of E_parallel, downsampled onto ESA timebase  [time]
                Returns C_E_parallel(E,pitch,t)
                '''

        # --- Adjust dimensions of inputs ---
        alpha = np.radians(pitch_angle[None, :, None])
        Engy = energies[None, None, :]
        E_para = Epara[:, None, None]

        # --- Calculate the Vmags ---
        Vmag = np.sqrt(2 * Engy * stl.q0 / mass)

        # --- Calculate the instantaneous FPC (detector coordinates) ---
        vparas = Vmag * np.cos(alpha)
        FPC_alternate_instant = (-1 * stl.q0) * vparas * distribution * E_para
        return FPC_alternate_instant

    def FPC_detector_coords(self, FPC_instant, time_obs):
        '''
            param FPC_instant - [time, pitch, energy] - instanteous FPC value over time
            param time_obs - [time] - observation times of the detector
        '''

        # Set the correlation window
        low_idx = np.abs(time_obs - 2.25).argmin()
        high_idx = np.abs(time_obs - 3.5).argmin()
        # low_idx = np.abs(time_obs - 0).argmin()
        # high_idx = np.abs(time_obs - time_obs[-1]).argmin()

        # Calculate the total FPC (detector coordinates). NOTE: assumes the deltaT between points is constant, DeltaT = constant, thus we only divided by 1/num_points
        n = high_idx - low_idx
        FPC = np.asarray((1 / n) * np.sum(FPC_instant[low_idx:high_idx + 1], axis=0))  # note: the 2*pi comes from integrating over cylindrical velocity space

        return FPC



    def _lin_edges(self, c):
        """Edges halfway between linearly spaced centres."""
        c = np.asarray(c, dtype=float)
        mid = 0.5 * (c[1:] + c[:-1])
        return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])

    def _log_edges(self, c):
        """Edges at geometric midpoints between log-spaced centres."""
        lc = np.log10(np.asarray(c, dtype=float))
        mid = 0.5 * (lc[1:] + lc[:-1])
        return 10 ** np.concatenate([[2 * lc[0] - mid[0]], mid, [2 * lc[-1] - mid[-1]]])

    def velocity_wedge_mesh(self,pitch_deg, energy_eV, values, mass=stl.m_e,
                            pitch_edges=None, energy_edges=None,
                            sub=1, v_unit=1.0):
        """Mesh for plotting (pitch, energy) data as wedges in (v_par, v_perp).

        Parameters
        ----------
        pitch_deg, energy_eV : 1-D bin centres (deg, eV)
        values       : 2-D array shaped (n_pitch, n_energy); transpose first if
                       yours is (n_energy, n_pitch)
        pitch_edges  : optional n_pitch+1 bin edges (deg). Default: midpoints
                       between centres, clipped to [0, 180].
        energy_edges : optional n_energy+1 bin edges (eV). Default: geometric
                       midpoints (for log-spaced energy bins).
        sub          : sub-steps per pitch-angle bin, so the arcs draw smoothly
        v_unit       : divide velocities by this (e.g. 1e7 for units of 10^4 km/s)

        Returns
        -------
        X, Y, C  ready for  ax.pcolormesh(X, Y, C, shading="flat")
                 X = v_par, Y = v_perp, both shaped (n_energy+1, n_pitch*sub+1);
                 C shaped (n_energy, n_pitch*sub).
        """
        pitch_deg = np.asarray(pitch_deg, dtype=float)
        energy_eV = np.asarray(energy_eV, dtype=float)
        values = np.asarray(values)
        if values.shape != (pitch_deg.size, energy_eV.size):
            raise ValueError(f"values shape {values.shape} != (n_pitch, n_energy) = "
                             f"({pitch_deg.size}, {energy_eV.size})")

        a_edges = (np.asarray(pitch_edges, dtype=float) if pitch_edges is not None
                   else np.clip(self._lin_edges(pitch_deg), 0.0, 180.0))
        e_edges = (np.asarray(energy_edges, dtype=float) if energy_edges is not None
                   else self._log_edges(energy_eV))

        # Split each pitch-angle bin into `sub` steps so its arc is smooth
        a_fine = np.concatenate( [np.linspace(a_edges[i], a_edges[i + 1], sub, endpoint=False) for i in range(pitch_deg.size)] + [[a_edges[-1]]])

        v_edges = self.to_Vel(e_edges, mass) / v_unit
        A, V = np.meshgrid(np.radians(a_fine), v_edges)  # (n_E+1, n_a_fine)
        v_par_mesh = V * np.cos(A)  # v_par
        v_perp_mesh = V * np.sin(A)  # v_perp
        C = np.repeat(values.T, sub, axis=1)  # (n_E, n_pitch*sub)
        return v_par_mesh, v_perp_mesh, C

    def _vperp_integral(self,vp, a_edges, v_edges, values, measure):
        """Exact integral over v_perp at each v_par in `vp` (1-D, m/s).

        At fixed v_par, the wedge of bin (i, j) covers v_perp in [u1, u2], bounded
        by the two speed edges (circles) and the two pitch-angle edges (lines)."""
        vp = np.asarray(vp, dtype=float)[:, None, None]  # (N, 1, 1)
        s = np.abs(vp)
        pos = vp >= 0
        a_lo, a_hi = a_edges[:-1][None, :, None], a_edges[1:][None, :, None]
        # pitch-angle edges as angles from the nearer v_par axis (0..90 deg)
        b_lo = np.where(pos, a_lo, 180.0 - a_hi)
        b_hi = np.where(pos, a_hi, 180.0 - a_lo)
        with np.errstate(invalid="ignore", over="ignore"):
            ang_lo = np.where(b_lo < 90.0, s * np.tan(np.radians(np.minimum(b_lo, 89.9999999))), np.inf)
            ang_hi = np.where(b_hi < 90.0, s * np.tan(np.radians(np.minimum(b_hi, 89.9999999))), np.inf)
        rad_lo = np.sqrt(np.maximum((v_edges[:-1] ** 2)[None, None, :] - vp ** 2, 0.0))
        rad_hi = np.sqrt(np.maximum((v_edges[1:] ** 2)[None, None, :] - vp ** 2, 0.0))
        u1 = np.maximum(ang_lo, rad_lo)
        u2 = np.minimum(ang_hi, rad_hi)
        good = u2 > u1
        u1 = np.where(good, u1, 0.0)
        u2 = np.where(good, u2, 0.0)
        wt = np.pi * (u2 ** 2 - u1 ** 2) if measure == "cylindrical" else (u2 - u1)
        return np.einsum("nae,ae->n", wt, values)

    def reduce_to_vpar(self, pitch_deg, energy_eV, values, mass=stl.m_e,
                       pitch_edges=None, energy_edges=None,
                       vpar_edges=None, n_vpar=400, measure="cylindrical"):
        """Integrate (pitch, energy) data over v_perp to get a reduced C(v_par).

        Each (alpha, E) bin holds a constant value over its wedge in the
        (v_par, v_perp) plane, and the integral is done exactly from that:

            measure="cylindrical":  C(v_par) = integral C(v_par, v_perp) 2 pi v_perp dv_perp
                Integral over the whole perpendicular plane of a gyrotropic 3-D
                velocity-space quantity. Result has units of values * (m/s)^2, and
                sum(C_par * dv_par) equals the full 3-D integral (see total_integral).
            measure="flat":         C(v_par) = integral C(v_par, v_perp) dv_perp
                Result has units of values * m/s.

        The output is the average of C(v_par) over each v_par bin. Between the
        points where a wedge boundary switches (v_par = v_edge * cos(alpha_edge)),
        the cylindrical integrand is a quadratic in v_par, so Simpson's rule on
        each piece makes the bin average exact.

        Parameters
        ----------
        values      : 2-D array shaped (n_pitch, n_energy)
        vpar_edges  : output v_par bin edges in m/s. Default: n_vpar bins over
                      +/- the top of the highest energy bin.

        Returns
        -------
        vpar_centres (m/s), C_par   -- both 1-D, length = len(vpar_edges) - 1
        """
        if measure not in ("cylindrical", "flat"):
            raise ValueError("measure must be 'cylindrical' or 'flat'")
        pitch_deg = np.asarray(pitch_deg, dtype=float)
        energy_eV = np.asarray(energy_eV, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape != (pitch_deg.size, energy_eV.size):
            raise ValueError(f"values shape {values.shape} != (n_pitch, n_energy)")

        a_edges = (np.asarray(pitch_edges, dtype=float) if pitch_edges is not None
                   else np.clip(self._lin_edges(pitch_deg), 0.0, 180.0))
        v_edges = self.to_Vel(np.asarray(energy_edges, dtype=float) if energy_edges is not None
                        else self._log_edges(energy_eV), mass)
        vmax = v_edges[-1]
        if vpar_edges is None:
            vpar_edges = np.linspace(-vmax, vmax, n_vpar + 1)
        vpar_edges = np.asarray(vpar_edges, dtype=float)

        # every v_par where some wedge boundary changes, plus the output bin edges
        brk = (v_edges[:, None] * np.cos(np.radians(a_edges))[None, :]).ravel()
        pts = np.unique(np.concatenate([brk, v_edges, -v_edges, [0.0], vpar_edges]))
        pts = pts[(pts >= vpar_edges[0]) & (pts <= vpar_edges[-1])]

        lo, hi = pts[:-1], pts[1:]
        mid = 0.5 * (lo + hi)
        f = self._vperp_integral(np.concatenate([lo, mid, hi]), a_edges, v_edges, values, measure)
        n = lo.size
        piece = (hi - lo) / 6.0 * (f[:n] + 4.0 * f[n:2 * n] + f[2 * n:])  # Simpson per piece

        which = np.searchsorted(vpar_edges, mid, side="right") - 1  # output bin of each piece
        C_par = np.bincount(which, weights=piece, minlength=vpar_edges.size - 1)
        C_par = C_par[:vpar_edges.size - 1] / np.diff(vpar_edges)
        return 0.5 * (vpar_edges[1:] + vpar_edges[:-1]), C_par

    def vperp_integral(self, pitch_deg, energy_eV, FPC,mass=stl.m_e):
        '''
            pitch_deg [1D]
            energy_eV [1D]
            FPC [2D], [pitch, energy]
        '''

        alpha_mesh, energy_mesh_eV = np.meshgrid(pitch_deg, energy_eV)

        energy_mesh = energy_mesh_eV*stl.q0

        # get the chain rule expansion terms
        term1 = np.sin(np.radians(alpha_mesh))*simpson(y=FPC*np.sqrt(1/(2*mass*energy_mesh)),x=energy_mesh,axis=1)
        term2 = np.sqrt(2*energy_mesh/mass)*simpson(y=FPC* np.cos(np.radians(alpha_mesh)),x=np.radians(pitch_deg),axis=0)

        return 2*np.pi*(term1 + term2)







    def total_integral(self,pitch_deg, energy_eV, values, mass=stl.m_e,
                       pitch_edges=None, energy_edges=None):
        """Exact integral of the binned data over all 3-D velocity space,
        sum_ij C_ij * 2 pi (v_hi^3 - v_lo^3)/3 * (cos a_lo - cos a_hi).
        Use it to check reduce_to_vpar: sum(C_par * dv_par) should match."""
        a_edges = np.radians(np.asarray(pitch_edges, dtype=float) if pitch_edges is not None
                             else np.clip(self._lin_edges(pitch_deg), 0.0, 180.0))
        v_edges = self.to_Vel(np.asarray(energy_edges, dtype=float) if energy_edges is not None
                        else self._log_edges(energy_eV), mass)
        d_ang = np.cos(a_edges[:-1]) - np.cos(a_edges[1:])  # (n_pitch,)
        d_rad = (v_edges[1:] ** 3 - v_edges[:-1] ** 3) / 3.0  # (n_energy,)
        return 2 * np.pi * np.sum(np.asarray(values) * d_ang[:, None] * d_rad[None, :])