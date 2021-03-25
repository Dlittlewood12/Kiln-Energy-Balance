# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:08:07 2021

@author: Dallin
"""

import numpy as np
import matplotlib.pyplot as plt
from thermo import Stream
from thermo.mixture import Mixture
from scipy.optimize import fsolve
import warnings 
warnings.filterwarnings('ignore') #to ignore RuntimeWarnings created when composition includes carbon dioxide


    
def Analysis(L,OD,N2,H2,H2O,CO2,CO,m):
    """Conditions"""
    comp = ['nitrogen', 'hydrogen', 'water', 'carbon dioxide', 'carbon monoxide']
    zs = [N2, H2, H2O, CO2, CO]
    for i in range(0,len(zs)):
        if zs[i]==0:
            if i==0:
                comp.remove('nitrogen')
            elif i==1:
                comp.remove('hydrogen')
            elif i ==2:
                comp.remove('water')
            elif i ==3:
                comp.remove('carbon dioxide')
            else:
                comp.remove('carbon monoxide')
            
    zs = np.array(zs)
    zeros = len(zs) - np.count_nonzero(zs > 0)
    delete = np.arange(0,zeros)
    for i in delete:
        zs = np.delete(zs,np.where(zs==0))
    
    Pavg = 0.859826 #atm
    Pavg_pa = 87121.86945 #Pa
    R = 0.08206 #L*atm/mol/K
    sigma = 5.67e-8
    Tavg = 681.15 #K
    Ti = 293.15 #K
    
    def Moody_fff(Re,eps):
        if Re<2100:
            return 16/Re
        else:
            y = -np.log10(eps/3.7 - 4.52/Re*np.log10(7/Re + eps/7))
            y = -np.log10(eps/3.7 + 5.02*y/Re)
            y = -np.log10(eps/3.7 + 5.02*y/Re)
            y = -np.log10(eps/3.7 + 5.02*y/Re)
            return 1/(16*y**2)
    
    def Nusselt(Re,Pr):
        if Re<2100:
            return 3.66
        else:
            return 0.023*Re**0.8 * Pr**0.4
        
    Feed = Stream(comp, zs=zs, T=Tavg, P=Pavg_pa, m=m)
    mixture = Mixture(comp, zs=zs, T=Tavg, P=Pavg_pa)


    """Gas Properties"""
    Cp = Feed.Cp #J/kg/K
    MW = Feed.MW #g/mol
    mu = Feed.mu #Pa*s or kg/m/s
    Pr = Feed.Pr
    k = Feed.k #W/m/k
    rho = Feed.rho #kg/m3
    
    """Tubing Properties"""
    Cp1 = 511 #J/kg/C # at 427 C
    rho1 = 8440 #kg/m3
    k1 = 15.7 #W/m/C at 427 C
    epsilon = 0.8 #emissivity
    
    """System"""
    OD *= .0254 #m
    thick = 0.035 * .0254 #m
    ID = (OD - 2*thick) #m
    CSA_i = np.pi * ID**2  /4 #m2
    SA_i = np.pi * ID * L

    """Flud Analysis"""
    E = 0.000196721
    v = m / rho / CSA_i
    Re = rho * v * ID / mu
    fff = Moody_fff(Re,E)
    dP = 4*fff*rho*(L/ID)*(v**2/2) #Pa
    Nu = Nusselt(Re,Pr)
    h = Nu * k / ID
    h_fc = 25 #W/m2/K
    Rthermal = (1/h/np.pi/ID/L) + np.log(OD/ID)/2/np.pi/L/k1
    U = 1/Rthermal/SA_i #W/m2/K
    
    """Solution"""
    n = 1000 #number of tubing node
    x = np.linspace(0,L,n)
    dx = L/n
    Tw = 1073.15 #K
    Tcsa = Tw - (Tw - Ti)*np.exp(-np.pi*ID*x*h/m/Cp)
    Tg = np.linspace(Ti,Tw,n)


    for i in range(0,n):
        def res(T):
            Tgi = T[0]
            Tti = T[1]
            eq1 = np.pi*OD*dx*sigma*epsilon*(Tw**4-Tti**4) + h_fc*np.pi*OD*dx*(Tw-Tti) - np.pi*ID*dx*U*(Tti-Tgi)
            eq2 = np.pi*ID*dx*U*(Tti-Tgi) - m*Cp*(Tgi-Tg[i-1])
            return[eq1,eq2]

        if i == 0:
            inletT= lambda Tt0: np.pi*OD*dx*sigma*epsilon*(1073.15**4-Tt0**4) + h_fc*np.pi*OD*dx*(Tw-Tt0) - np.pi*ID*dx*U*(Tt0-Tg[i])
            inletTt = fsolve(inletT,600)
            Tt = np.linspace(inletTt[0],Tw,n)

        else:
            sol = fsolve(res,[Tg[i-1],Tt[i-1]])
            Tg[i] = sol[0]
            Tt[i] = sol[1]


    plt.clf()
    plt.plot(x,Tg-273.15,color='blue',label='Gas T')
    plt.plot(x,Tt-273.15,color='red',label='Tube T')
    plt.plot(x,Tcsa-273.15,color='orange',label='Constant Surface T Assumption')
    plt.axis([0,L,0,850])
    plt.xlabel('Distance (m)')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='lower right')
    plt.show()
    
        
    
    print('----- Gas Properites -----')
    print('Cp = {:.3f}   J/kg/K'.format(Cp))
    print('MW = {:.3f}     g/mol'.format(MW))
    print(' μ = {:.3e}  kg/m/s'.format(mu))
    print('Pr = {:.3f}'.format(Pr))
    print(' k = {:.3f}      W/m/K'.format(k))
    print(' ρ = {:.3f}      kg/m3'.format(rho))
    print(' h = {:.3f}     W/m2/K'.format(h))
    print('ΔP = {:.3f}    Pa'.format(dP))
    print('--------------------------')
    if Tg[-1] > (795+273.15):
        L795 = np.interp(795+273.15,Tg,x)
        print('T = 795 C after {:.1f} m'.format(L795))
    else:
        print('T = {:.2f} C after {:.1f} m'.format(Tg[-1]-273.15,L))
