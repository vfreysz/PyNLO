# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:29:08 2014
This file is part of pyNLO.

    pyNLO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pyNLO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pyNLO.  If not, see <http://www.gnu.org/licenses/>. 
    
@author: dim1
"""

import numpy as np
import matplotlib.pyplot as plt
from pynlo.interactions import FourWaveMixing
from pynlo.media.fibers import fiber
from pynlo.light.DerivedPulses import SechPulse


plt.close('all')

steps = 200

centerwl = 850.0
gamma = 1e-3
fiber_length = 1600.0
T0 = 1
beta = [-1, 0]
P0 = (abs(beta[0] * 1e-3) / gamma / T0**2) * 16

init =SechPulse(power                   = P0, 
                T0_ps                   = T0,
                center_wavelength_nm    = centerwl,
                time_window_ps         = 10.0,
                NPTS = 2**13)

fiber1 = fiber.FiberInstance()
fiber1.generate_fiber(fiber_length, centerwl, beta, gamma, 0, "ps^n/km")

evol = FourWaveMixing.SSFM.SSFM(disable_Raman = True, disable_self_steepening = True,
            local_error = 0.001, suppress_iteration = True)

y = np.zeros(steps)
AW = np.complex64(np.zeros((init.NPTS, steps)))
AT = np.complex64(np.copy(AW))

y, AW, AT, pulse_out = evol.propagate(pulse_in = init, fiber = fiber1, 
                                         n_steps = steps)
                                         
wl = init.wl_nm 

loWL = 820
hiWL = 870
                         
iis = np.logical_and(wl>loWL,wl<hiWL)

iisT = np.logical_and(init.T_ps>-5,init.T_ps<5)

xW = wl[iis]
xT = init.T_ps[iisT]
zW_in = np.transpose(AW)[:,iis]
zT_in = np.transpose(AT)[:,iisT]
zW = 10*np.log10(np.abs(zW_in)**2)
zT = 10*np.log10(np.abs(zT_in)**2)
mlIW = np.max(zW)
mlIT = np.max(zT)

D = fiber1.Beta2_to_D(init)

x = (init.V_THz) / (2* np.pi) * T0
x2 = init.T_ps / T0
b2 = beta[0] / 1e3 # in ps^2 / m
LD = T0**2 / abs(b2)
ynew = y / LD

plt.figure()
plt.subplot(121)
plt.pcolormesh(x2, ynew, 10*np.log10(np.abs(np.transpose(AT)[:, :-1])**2),
               vmin = mlIT - 20.0, vmax = mlIT, cmap = plt.cm.gray)
plt.autoscale(tight=True)
plt.xlim([-4, 4])
plt.xlabel(r'($T / T_0)$')
plt.ylabel(r'Distance ($z/L_{NL})$')
plt.subplot(122)
plt.pcolormesh(x, ynew, 10*np.log10(np.abs(np.transpose(AW)[:, :-1])**2),
               vmin = mlIW - 20.0, vmax = mlIW, cmap = plt.cm.gray)
plt.autoscale(tight=True)
plt.xlim([-4, 4])
plt.xlabel(r'($\nu - \nu_0) \times T_0$')
plt.ylabel(r'Distance ($z/L_{NL})$')

#plt.figure()
#plt.subplot(121)
#plt.pcolormesh(xW, y, zW, vmin = mlIW - 40.0, vmax = mlIW)
#plt.autoscale(tight=True)
#plt.xlim([loWL, hiWL])
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Distance (m)')
#
#plt.subplot(122)
#plt.pcolormesh(xT, y, zT, vmin = mlIT - 40.0, vmax = mlIT)
#plt.autoscale(tight=True)
#plt.xlabel('Delay (ps)')
#plt.ylabel('Distance (m)')

plt.show()