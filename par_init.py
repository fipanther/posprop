"""
    ____                __  _________
    / __ \____  _____   /  \/  / ____/
    / /_/ / __ \/ ___/  / /\_/ / /
    / ____/ /_/ (__  )  / /  / / /___
    /_/    \____/____/  /_/  /_/\____/
    
    Initialize and update parameters
    Fiona H. Panther, Australian National University
    v. 1.2
"""
from __future__ import division, print_function

import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': 22})

c_dir = os.getcwd()

sys.path.append(c_dir)

import user_functions as usr
import load as ld

#   Import the external simulation information

s_yr = 3.15576*10**7
timing_ext = np.linspace(0, 1E6*s_yr, 1E3)
#arr_nesim = [1]*100000
#arr_nHsim = [1000]*100000
arr_nHesim = [0]*100000
arr_xHsim = [0.01]*100000
arr_xHesim = [0]*100000
arr_xHe2sim = [0]*100000
arr_temperaturesim = [8000]*100000
arr_Bsim = [50]*100000
arr_rhosim = [1E-25]*100000

model_parameters = [0.100, 500]



#   Initialize the simulation
def initialize_MCmp(paramset, arrays0, arrays1, arrays2, arrays3, arrays4, arrays5, arrays6, arrays7, arraystime, arrays8):
    """ Initialize the data arrays for each MC particle
        Parameters:
            paramset: tuple length 9 containing the first value in the arrays from the external simulation
                      [E, H, He, xH, xHe, xHe2, T, B, rho]
    """
    if not arrays0:
        arrays0.append(paramset[0])
    if not arrays1:
        arrays1.append(paramset[1])
    if not arrays2:
        arrays2.append(paramset[2])
    if not arrays3:
        arrays3.append(paramset[3])
    if not arrays4:
        arrays4.append(paramset[4])
    if not arrays5:
        arrays5.append(paramset[5])
    if not arrays6:
        arrays6.append(paramset[6])
    if not arrays7:
        arrays7.append(paramset[7])
    if not arraystime:
        arraystime.append(0)
    if not arrays8:
        arrays8.append(paramset[8])
    return

def dumparr(arrays0, arrays1, arrays2, arrays3, arrays4, arrays5, arrays6, arrays7, arrays8, arraystime, timeidx):
    """ 
    Dump the data arrays at the end of the particle's life (deprecated)
    """
    del arrays0[:], arrays1[:], arrays2[:], arrays3[:], arrays4[:], arrays5[:], arrays6[:], arrays7[:], arrays8[:], arraystime[:]
    timeidx = 0
    return

#   Update the energy
def update_epacket(current_energy, energy_lost, arrname):
    return arrname.append(current_energy - energy_lost)

#   Update the time
def update_timing(dt, tnow, arrname):
    return arrname.append(tnow+dt)

#   Update the ISM parameters add in rho
def update_params(dt, idx_now, idx_next, arrays1, arrays2, arrays3, arrays4, arrays5, arrays6, arrays7, arrays8, time_sim, time_ext, sim_nH):
    """
    Updates the ISM parameters after a timestep.
    """
    #   Set the times as parameters inside function
    t_now = time_sim[idx_now]
    t_next = time_ext[idx_next]
    t_up = time_sim[idx_now]+dt
    t_mid = t_now+((t_next-t_now)/2)
    
    if t_up <t_mid:
        """
        If the updated time is closer to the current time than the end of the external timestep, stick with the ISM parameters from the previous timestep
        """
        nH_next = arrays1[idx_now]
        nHe_next = arrays2[idx_now]
        xH_next = arrays3[idx_now]
        xHe_next = arrays4[idx_now]
        xHe2_next = arrays5[idx_now]
        T_next = arrays6[idx_now]
        B_next = arrays7[idx_now]
        rho_next = arrays8[idx_now]
    elif t_up > t_mid:
        """
        If the updated time is closer to the end of the external timestep than the beginning, use the ISM parameters from the beginning of the next external timestep
        """
        nH_next = sim_nH[idx_next]
        nHe_next = arr_nHesim[idx_next]
        xH_next = arr_xHsim[idx_next]
        xHe_next = arr_xHesim[idx_next]
        xHe2_next = arr_xHe2sim[idx_next]
        T_next = arr_temperaturesim[idx_next]
        B_next = arr_Bsim[idx_next]
        rho_next = arr_rhosim[idx_next]
    elif t_up == t_mid:
        """
        If the updated time is right in the middle, use the ISM parameters from the beginning of the next external timestep. This is a separate elif statememt because for some reason having it in the same elif statement as either of the others (>= or <=) seems to break the simulation. Probably some Python thing?
        """
        nH_next = sim_nH[idx_next]
        nHe_next = arr_nHesim[idx_next]
        xH_next = arr_xHsim[idx_next]
        xHe_next = arr_xHesim[idx_next]
        xHe2_next = arr_xHe2sim[idx_next]
        T_next = arr_temperaturesim[idx_next]
        B_next = arr_Bsim[idx_next]
        rho_next = arr_rhosim[idx_now]

    return arrays1.append(float(nH_next)), arrays2.append(float(nHe_next)), arrays3.append(float(xH_next)), arrays4.append(float(xHe_next)), arrays5.append(float(xHe2_next)), arrays6.append(float(T_next)), arrays7.append(float(B_next)), arrays8.append(float(rho_next))



def X_ism(arrays0, arrays1, arrays2, arrays3, arrays4, arrays5, arrays6, arrays7, arrays8):
    """
    Read in the current parameters from the arrays
    """
    energy = arrays0[len(arrays0)-1]
    n_H = arrays1[len(arrays0)-1]
    n_He = arrays2[len(arrays0)-1]
    x_H = arrays3[len(arrays0)-1]
    x_He = arrays4[len(arrays0)-1]
    x_He2 = arrays5[len(arrays0)-1]
    T = arrays6[len(arrays0)-1]
    B = arrays7[len(arrays0)-1]
    rho = arrays8[len(arrays0)-1]
    return [energy, n_H, n_He, x_H, x_He, x_He2, T, B, rho]


def Hxsec(energy, nH_neu):
    """
    Load the hydrogen cross sections into a convenient array
    """
    if nH_neu ==0:
        """
        Avoid those pesky NaN's if there are no neutral hydrogen atoms
        """
        return [0,0,0]
    
    
    else:
        temp = usr.find_nearest(ld.arr_H_ion[:,0], energy)
        if energy<14.6648 or energy> 684.091:
            xsec_Hi = 0
        else:
            xsec_Hi = ld.arr_H_ion[temp[0],1]

        temp = usr.find_nearest(ld.arr_H_ex[:,0], energy)
        if energy<9.977 or energy>597.605:
            xsec_Hex = 0
        else:
            xsec_Hex = ld.arr_H_ex[temp[0],1]
    
        temp = usr.find_nearest(ld.arr_H_cx[:,0], energy)
        if energy<6.87 or energy>960:
            xsec_Hc = 0
        else:
            xsec_Hc = ld.arr_H_cx[temp[0],1]

        return [xsec_Hi, xsec_Hex, xsec_Hc]

#   Load the current helium cross sections into a convenient array
def Hexsec(energy, nHe_neu):
    """
    Load the helium cross sections into a convenient array
    """
    if nHe_neu==0:
        """
        Avoid those pesky NaNs if there are no neutral helium atoms
        """
        return [0,0,0]
    
    else:
        temp = usr.find_nearest(ld.arr_He_ion[:,0], energy)
        if energy<31.97 or energy>985.959:
            xsec_Hei = 0
        else:
            xsec_Hei = ld.arr_He_ion[temp[0],1]
            
        temp = usr.find_nearest(ld.arr_He_ex[:,0], energy)
        if energy<23.09 or energy>1045.4:
            xsec_Heex = 0
        else:
            xsec_Heex = ld.arr_He_ex[temp[0],1]

        temp = usr.find_nearest(ld.arr_He_cx[:,0], energy)
        if energy<18.94 or energy>995.66:
            xsec_Hec = 0
        else:
            xsec_Hec = ld.arr_He_cx[temp[0],1]

        return [xsec_Hei, xsec_Heex, xsec_Hec]


def d_xsec(energy):
    """
    define the direct annihilation x-section
    """
    if energy<70*ld.kev:
        """
        At low energies, use the classical x-sec with quantum correction factor
        """
        return (np.pi*ld.r0**2/(usr.rel_bet(energy, ld.mp)))*((2*np.pi*ld.a/usr.rel_bet(energy, ld.mp))/(1-np.exp(-2*np.pi*ld.a/usr.rel_bet(energy, ld.mp))))
    else:
        """
        At high energies use the Dirac cross-section
        """
        return (np.pi*ld.r0**2/(usr.rel_gam(energy, ld.mp)+1))*(((usr.rel_gam(energy, ld.mp)**2 + 4*usr.rel_gam(energy,ld.mp) + 1)/(usr.rel_gam(energy, ld.mp)**2 - 1))*np.log(usr.rel_gam(energy, ld.mp)+ np.sqrt(usr.rel_gam(energy, ld.mp)**2 - 1)) - ((usr.rel_gam(energy, ld.mp) + 3)/(np.sqrt(usr.rel_gam(energy, ld.mp)**2 - 1))))

def norm_xsec(energy, nH_tot, nHe_tot, x_H, x_He, x_He2):
    """
    Total normalized interaction cross-section
    """
    nH_neu = (1-x_H)*nH_tot
    nHe_neu = (1-x_He-x_He2)*nHe_tot
    ne = (nH_tot)+(2*nHe_tot)
    norm_H = nH_neu*sum(Hxsec(energy, nH_neu))
    norm_He = nHe_neu*sum(Hexsec(energy, nHe_neu))
    norm_e = ne*d_xsec(energy)
    return norm_H+norm_He+norm_e

# DO THIS

##   Load the current thermalized annihilation rates at current temperature into a convenient array
#def arr_rate(temperature, n_e, n_H, n_He, x_H, x_He, x_He2):
#    #   Multiply out the density factor in this to simplify the do_science run
#    #   Direct-free annihilation
#    temp = usr.find_nearest(ld.arr_df[:,1],temperature)
#    if temperature>1E8 or n_e ==0:
#        r_df = 0
#    else:
#        r_df = n_e*ld.arr_df[temp[0], 0]
#    
#    temp = usr.find_nearest(ld.arr_rr[:,0],temperature)
#    if temperature>1E8 or n_e==0:
#        r_rr = 0
#    else:
#        r_rr = n_e*ld.arr_rr[temp[0],1]
#    #  grid isn't fine enough on radiative recomb.
#
#    temp = usr.find_nearest(ld.arr_hcx[:,0],temperature)
#    if temperature<1994 or temperature>1E8:
#        r_hcx = 0
#    else:
#        r_hcx = n_H*(1-x_H)*ld.arr_hcx[temp[0],1]
#    
#    temp = usr.find_nearest(ld.arr_hecx[:,0], temperature)
#    if temperature<5843 or temperature>1E8:
#        r_hecx = 0
#    else:
#        r_hecx = n_He*(1-x_He-x_He2)*ld.arr_hecx[temp[0],1]
#
#    temp = usr.find_nearest(ld.arr_hdb[:,0], temperature)
#    if temperature >1E4:
#        r_hdb = 0
#    else:
#        r_hdb = n_H*(1-x_H)*ld.arr_hdb[temp[0],1]
#    
#    temp = usr.find_nearest(ld.arr_hedb[:,0], temperature)
#    if temperature>1E8:
#        r_hedb = 0
#    else:
#        r_hedb = n_He*(1-x_He*x_He2)*ld.arr_hedb[temp[0],1]
#    
#    return [r_df, r_rr, r_hcx, r_hecx, r_hdb, r_hedb]



def eloss_rate(energy, nH_tot, nHe_tot, x_H, x_He, x_He2, B, temperature):
    """
    energy loss rates
    """
    nH_neu = (1-x_H)*nH_tot
    nHe_neu = (1-x_He-x_He2)*nHe_tot
    ne = (x_H*nH_tot)+((x_He+2*x_He2)*nHe_tot)

    if energy>13.8:
        ion = ld.dE_io(energy, nH_neu, 1, 13.8)+ld.dE_io(energy, nHe_neu, 2, 20)
    else:
        ion = 0
    pla = ld.dE_pl(energy, ne, temperature)
    syn = ld.dE_sy(energy, B)
    bre = ld.dE_bri(energy, x_H*nH_tot, 1)+ld.dE_bri(energy, (x_He+2*x_He2)*nHe_tot, 2) + ld.dE_brn(energy, nH_neu, nHe_neu, x_H, x_He, x_He2)
    ic = ld.dE_ic(energy, 0.26)
    return [ion, abs(pla), syn, bre, ic, (ion+pla+syn+bre+ic)]

# ene = np.logspace(0, 12, 1E3)
# ion = []
# pl = []
# syn = []
# bre = []
# ic = []
# for i in ene:
#     ion.append(eloss_rate(i, 1, 0, 0.1, 0, 0, 10, 8000)[0])
#     pl.append(abs(eloss_rate(i, 1, 0, .1, 0, 0, 10, 8000)[1]))
#     syn.append(eloss_rate(i, 1, 0, 0.1, 0, 0, 10, 8000)[2])
#     bre.append(eloss_rate(i, 1, 0, 0.1, 0, 0, 10, 8000)[3])
#     ic.append(eloss_rate(i, 1, 0, 0.1, 0, 0, 10, 8000)[4])

# plt.figure(figsize = (20,10))
# plt.plot(ene, pl, 'k-',lw = 2 ,label = '$\mathrm{Plasma}$')
# plt.plot(ene, ion, 'k--', lw = 2 ,label = '$\mathrm{Ionization}$')
# plt.plot(ene, syn, 'k:',lw = 2 ,label = '$\mathrm{Synchrotron\,(10\mu G)}$')
# plt.plot(ene, bre, 'k-', lw = 2 ,linestyle='dashdot',label = '$\mathrm{Bremsstrahlung\,(total)}$')
# line, = plt.plot(ene, ic, 'k-', lw = 2 ,label = '$\mathrm{Inverse\,Compton}$')
# line.set_dashes([8, 4, 2, 4, 2, 4])
# plt.ylim([1E-17, 1E-2])
# plt.xscale('log')
# plt.yscale('log')
# plt.tick_params(axis='both', which='major', labelsize=28)
# plt.xlabel('$\log[\mathrm{energy/eV}]$', fontsize=35)
# plt.ylabel('$\log[dE/dt\,/\mathrm{s/eV}]$', fontsize=35)
# plt.legend(loc = 'best', fontsize = 28)
# plt.tight_layout()
# plt.savefig('energyloss.pdf',bbox_inches='tight',format='pdf')
