# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:13:54 2018

@author: simon X Tanguy
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.collections as collections

choice_colors = ['tab:blue',
            'tab:orange',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']

choice_colors1 = ['#045a8d',
            '#238b45',
            '#bd0026',
            '#6a51a3',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']


choice_colors2 = ['#74a9cf',
            '#74c476',
            '#fc4e2a',
            '#9e9ac8',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']


class StockCellRecord :
    # Register of all past states and flows of a cell

    def __init__(self, deltat, resourceName,
                 Xt, Xh_init, Xl_init, Xs_init,
                 muH_init, muL_init, K0, Rp0 ) :

        self.deltat = deltat
        self.resourceName = resourceName
        # Time Vector
        self.t = [0]
        # State of the reservoirs
        self.Xt = Xt
        self.Xh = [Xh_init]
        self.Xl = [Xl_init]
        self.Xs = [Xs_init]
        self.Xb = [0]
        # Potentials
        self.muH = [muH_init]
        self.muL = [muL_init]
        # Flows
        self.Fhp = [0]
        self.Flp = [0]
        self.G = [0]
        self.Gused = [0]
        self.Fr = [0]
        self.Fnr = [0]
        # Intensities
        self.Jp = [0]
        self.Jpd = [0]
        self.Jpmax = [0]
        # Other
        self.efficiency = [0]
        self.hasStock = [-1] # -1 if no stock, 0 if stock included "near" the target stock, 1 if stock > otherwise
        self.isLimitated = [False]
        self.K = [K0]
        self.Rp = [Rp0] # Friction

    def toString(self) :
        s = "ressource : " + str(self.resourceName)
        s+= "t = " +str(self.t[-1])
        s += "Xt = " + str(self.Xt)
        s += "Xh = " + str(self.Xh[-1])
        s += "Xl = " + str(self.Xl[-1])
        s += "Xs = " + str(self.Xs[-1])
        s += "Xs = " + str(self.Xs[-1])
        s += "Xb = " + str(self.Xb[-1])
        s += "muH = " + str(self.muH[-1])
        s += "muL = " + str(self.muL[-1])
        s += "Fhp = " + str(self.Fhp[-1])
        s += "Flp = " + str(self.Flp[-1])
        s += "G = " + str(self.G[-1])
        s += "Gused = " + str(self.Gused[-1])
        s += "Fr = " + str(self.Fr[-1])
        s += "Fnr = " + str(self.Fnr[-1])
        s += "Jp = " + str(self.Jp[-1])
        s += "Jpd = " + str(self.Jpd[-1])
        s += "Jpmax = " + str(self.Jpmax[-1])
        s += "efficiency = " + str(self.efficiency[-1])
        s += "hasStock = "  + str(self.hasStock[-1])
        s += "isLimitated = " + str(self.isLimitated[-1])
        s += "Rp = " + str(self.Rp[-1])
        s += "\n"
        return s
    
    def actualize(self,
                  Xh, Xl, Xs, Xb, muH, muL, Fhp, Flp, G, Gused, Fr, Fnr, Jp, Jpd, Jpmax,
                  efficiency, hasStock, isLimitated, K, Rp):
        self.t.append(self.t[-1] + self.deltat)
        self.Xh.append(Xh)
        self.Xl.append(Xl)
        self.Xs.append(Xs)
        self.Xb.append(Xb)
        self.muH.append(muH)
        self.muL.append(muL)
        self.Fhp.append(Fhp)
        self.Flp.append(Flp)
        self.G.append(G)
        self.Gused.append(Gused)
        self.Fr.append(Fr)
        self.Fnr.append(Fnr)
        self.Jp.append(Jp)
        self.Jpd.append(Jpd)
        self.Jpmax.append(Jpmax)
        self.efficiency.append(efficiency)
        self.hasStock.append(hasStock)
        self.isLimitated.append(isLimitated)
        self.K.append(K)
        self.Rp.append(Rp)

#----------------------------------------------------------------------------------------    

class StockCell :
    # Sheet describing a stock resource

    def __init__(self,
                 deltat,
                 name,
                 Xt,
                 Xh_init,
                 Xl_init,
                 Rp0,
                 K0,
                 recyclingEnergyFlux,
                 isEnergy,
                 r,
                 to,
                 stock_cible,
                 alpha,
                 delta,
                 xc,
                 x0):
        if Xh_init + Xl_init != Xt :
            print("ERROR : {} : Xh_init + Xl_init != Xt".format(name))
        self.deltat = deltat
        self.name = name
        self.Xt = Xt            # total recoverable quantity 
        self.Xh = Xh_init       # High reservoir
        self.Xl = Xl_init       # Low reservoir
        self.Xs = Xt - Xh_init - Xl_init
        self.Xb = 0                                     # Xbuffer: quantity that is "pending," ready to be sent to production (of a recipe) or to the stock at the next time step.
        self.Jp = 0.1                                     # Nominal operating intensity
        self.Jpd = self.Jp                               #Huh?
        self.K = K0
        self.Rp0 = Rp0
        self.recyclingEnergyFlux = recyclingEnergyFlux  # Energy flux required to obtain a recycling flux of 1 unit of resource per unit of time
        self.isEnergy = isEnergy                        # -1 if it is not an energy cell. Otherwise, isEnergy (>0) is the resource flux required to produce one unit of energy
        self.r = r
        self.to = to
        self.stock_cible = stock_cible
        self.alpha = alpha
        self.delta = delta
        self.xc = xc
        self.x0 = x0
        self.RT = 1
        self.muH0 = -1 * self.RT * math.log( xc ) # RT * math.log( x0/xc )  # muH0 is set such as muH(xH=0)=0
        self.muL0 = -1 * self.RT * math.log( xc ) # RT * math.log( x0/xc )  # muLO is set such as muL(xL=1)=0 (<=> to muH0 = muL0)
        self.record = StockCellRecord(deltat, name, Xt, Xh_init, Xl_init, Xt - Xh_init - Xl_init, self.muH(), self.muL(), K0, Rp0)


    def toString(self) :
        print(self.record.toString())


    def muH(self) :
        if self.Xh > self.xc:
            return self.muH0 + self.RT * math.log( self.Xh / self.Xt * self.x0  )
        if self.Xh <= self.xc:  # xc is the minimal concentration
            return self.muH0 + self.RT * math.log( self.xc  )


    def muL(self) :
        return self.muL0 + self.RT * math.log( self.xc  )



    def deltaXl(self, i) :
        return self.record.Flp[-i] + self.record.Gused[-i]


    def deltaXs(self) :
        return self.record.Xb[-1] - self.record.Gused[-1]*self.deltat



    def deltaPi(self) :
        return self.muH() - self.muL()


    def Jpmax(self) :
        return (self.deltaPi())/(2*self.Rp())


    def Jpopt(self, Gd) :
        delta = self.deltaPi()*self.deltaPi() - 4*self.Rp()*(Gd - self.Xs/self.deltat)
        if delta < 0 :
            return self.JpMax()
        else :
            Jp =(self.deltaPi() - math.sqrt(delta))/(2*self.Rp())
            if Jp < 0 :            # No negative flux allowed
                Jp = 0
                return Jp
            else :
                return Jp
    
    
    def actualize(self, inputs):
        self.K = inputs[0]
        self.recyclingEnergyFlux = inputs[1]
        self.to = inputs[2]
        self.stock_cible = inputs[3]
        self.delta = inputs[4]
    
    
    def Rp(self) :
        Rp = self.Rp0*self.record.K[0]/self.K + 4e-6
        if Rp > 1e100 :
            Rp = 1e100
        return Rp
    
    def iterate(self, Gused, Fr) :
        # This function performs one iteration of Xh, Xl, and Xs 
        
        self.K = self.K*(1-self.delta*self.deltat)
        if self.K < 1e-100 :
            self.K = 1e-100
        Rp = self.Rp()
        
        x = self.Xs
        dx = (self.record.G[-1] - self.record.Gused[-1])*self.deltat
        self.Jpd += 10/self.stock_cible/self.Xt * \
            (self.stock_cible*self.Xt - x - 25*dx/self.deltat) * self.deltat

        if self.Jpd < 1e-14 :
            self.Jpd = 1e-14
        
        isLimitated = False  
        if self.Jpd > self.Jpmax() :
            self.Jpd = self.Jpmax()
            isLimitated = True
            
        self.Jp = (self.Jp + (self.deltat/self.to)*self.Jpd)/(1 + self.deltat/self.to)
        
        if self.Jp > self.Jpmax():
            self.Jp = self.Jpmax()
        
        # Avoid negative values for Jp
        if self.Jp < 0 :
            self.Jp = 0
        
        # Compute the values of the flows during a time step.
        if self.muH()*self.Jp < self.muL()*self.Jp + Rp*self.Jp*self.Jp : 
            # One should never enter this loop because the intensity of a cell should always be less than or equal to the maximum production intensity.
            print("\n\t If you read this message, something is wrong!!!")
            print(self.Jpd)
            print(self.Jp)
            print(self.Jpmax())
            print(self.Rp())
            print(self.K)
         
        else :
            Fhp = self.muH()*self.Jp
            Flp = self.muL()*self.Jp + Rp*self.Jp*self.Jp
            if Fhp > self.Xh/self.deltat :
                Fhp = self.Xh/self.deltat
                self.Jp = Fhp/self.muH()
                self.Jpd = Fhp/self.muH()
                Flp = self.muL()*self.Jp + Rp*self.Jp*self.Jp
            G = Fhp - Flp
            
        efficiency = 0
        if Fhp != 0 :
            efficiency = G/Fhp
        
        # Calculate what will be naturally recycled during this time step
        Fnr = 0
        if self.r > 0 :
            # Beware, the function in the paper is self.r * self.Xh * (1 - Th) * (Th / self.s - 1)
            Fnr = self.r*self.Xl*(1-math.exp(-(self.Xl/self.Xt)/0.5))
        
            
        if Fr + Fnr > self.Xl/self.deltat :
            print("Attention, energy is being lost - see calculation of Fr for cell {}".format(self.name))
            if Fnr > self.Xl/self.deltat  :
                Fnr = self.Xl/self.deltat
                Fr = 0
            elif Fr > self.Xl/self.deltat :
                Fr = self.Xl/self.deltat
                Fnr = 0
            else :
                Fnr = self.Xl/self.deltat- Fr
                
        efficiency = 0
        if Fhp != 0 :
            efficiency = G/Fhp
        
        if self.Xs < 1e-14 :
            hasStock = 0
        elif self.Xs < self.stock_cible*self.Xt*0.9 :
            hasStock = 0.5
        elif self.Xs > self.stock_cible*self.Xt*1.1 :
            hasStock = 1.5
        else : 
            hasStock = 1.0
        
        # Update Xh, Xl, Xs
        self.Xs = self.Xs + self.Xb - Gused*self.deltat
        self.Xh = self.Xh + (- Fhp + Fr + Fnr)*self.deltat
        self.Xl = self.Xl + (Flp + Gused - Fr - Fnr)*self.deltat
        self.Xb = G * self.deltat
        
        self.record.actualize(self.Xh, self.Xl, self.Xs, self.Xb, self.muH(), self.muL(), Fhp, Flp, G, Gused, Fr, Fnr, self.Jp, self.Jpd, self.Jpmax(), efficiency, hasStock, isLimitated, self.K, Rp)

        
    def plot(self, indFigure):
        
        record = self.record
        
        plt.figure(indFigure, figsize = (20, 14), tight_layout = True)
        mpl.rcParams['axes.linewidth'] = 5
        fs = 25
        plt.subplot(3,3,1)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.plot(record.t, record.Xh,label='$X_h}$',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.Xl,label='$X_l}$',linewidth=4, color=choice_colors1[2])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Reservoirs',fontsize=fs)
        plt.legend(fontsize = fs)
        plt.grid(False)

    
        plt.subplot(3,3,2)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.Fhp,label='$F_hp}$',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.Flp,label='$F_lp}$',linewidth=4, color=choice_colors1[2])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Flows',fontsize=fs)
        plt.legend(fontsize = fs)
        plt.grid(False)

    
    
        plt.subplot(3,3,3)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.G,label='$G}$',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.Gused,label='$G_{used}$',linewidth=4, color=choice_colors1[2])
#        plt.plot(record.t, (np.array(record.Xb) + np.array(record.Xs))/self.deltat,label='$Available$',linewidth=1)
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Production',fontsize=fs)
        plt.legend(fontsize = fs)
        plt.grid(False)

        
        
        plt.subplot(3,3,4)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.Fnr,label='$F_{nr}$',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.Fr,label='$F_r}$',linewidth=4, color=choice_colors1[2])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Recycling',fontsize=fs)
        plt.legend(fontsize = fs)
        plt.grid(False)


        
        plt.subplot(3,3,5)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
        plt.plot(record.t,record.Jp,label='$I_p}$',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t,record.Jpmax, label='$I_p^{max}}$', linewidth=4, color=choice_colors1[2])
        plt.plot(record.t, record.Jpd, label='$I_p^{d}}$', linewidth=4, color=choice_colors1[1])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Prod Intensity',fontsize=fs-5)
        plt.legend(fontsize = fs)
        plt.grid(False)

        
    
        plt.subplot(3,3,6)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.Xs,linewidth=4, color=choice_colors1[0])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Stock',fontsize=fs)
        #plt.legend(fontsize = fs)
        plt.grid(False)
    
        
        plt.subplot(3,3,7)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=0, labelsize=24)
        
        plt.yscale('log')
        plt.plot(record.t, record.Rp,linewidth=4, color=choice_colors1[0])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('R$_p$',fontsize=fs)
        #plt.legend(fontsize = fs)
        plt.grid(False)

        
    
        plt.subplot(3,3,8)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, np.array(record.muH)-np.array(record.muL),linewidth=4, color=choice_colors1[0])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('$\Delta \pi$',fontsize=fs)
        # plt.ylim((0, 1))
        #plt.legend(fontsize = fs)
        plt.grid(False)

    
        
        plt.subplot(3,3,9)
        plt.xticks([])
        plt.yticks([])
        
        plt.text(0.05, 0.85 , self.name, fontsize=fs-5)
        plt.text(0.05, 0.70 , r'Total='    +str(self.Xt), fontsize=fs-5)
        plt.text(0.05, 0.55 , r'Nat$_R$='    +str(self.r), fontsize=fs-5)
        plt.text(0.05, 0.40 , r'Duration=' +str(int(len(record.t)-1)*self.deltat), fontsize=fs-5)
        plt.text(0.05, 0.25 , r'Time step='+str(self.deltat), fontsize=fs-5)
        
        plt.grid(False)

    
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.6)
        plt.tight_layout()
        
#############################################################################################################################################

class FlowCellRecord :
    
    def __init__(self, deltat, name, eff_init, installed_surface_init, stockMax_init, K1_init, K2_init):
        self.deltat = deltat
        self.name = name
        self.t = [0]
        self.G = [0]
        self.Gused = [0]
        self.Xs = [0]
        self.Xb = [0]
        self.eff = [eff_init]
        self.installed_surface = [installed_surface_init]
        self.stockMax = [stockMax_init]
        self.K1 = [K1_init]
        self.K2 = [K2_init]

    def actualize(self, G, Gused, Xs, Xb, eff, installed_surface, stockMax, K1, K2):
        self.t.append(self.t[-1] + self.deltat)
        self.G.append(G)
        self.Gused.append(Gused)
        self.Xs.append(Xs)
        self.Xb.append(Xb)
        self.eff.append(eff)
        self.installed_surface.append(installed_surface)
        self.stockMax.append(stockMax)
        self.K1.append(K1)
        self.K2.append(K2)

#-----------------------------------------------------------------------------------------------------------------------

class FlowCell :
    # Sheet describing a flow resource

    def __init__(self, deltat, name, incidentFlow, eff_init, installed_surface, isEnergy, stockMax_init, delta):
        self.deltat = deltat
        self.name = name        
        self.incidentFlow = incidentFlow
        self.Xs = 0
        self.Xb = 0
        self.efficiency = eff_init      
        self.installed_surface_init = installed_surface
        self.isEnergy = isEnergy
        self.stockMax_init = stockMax_init
        self.K1 = 1
        self.K2 = 1
        self.delta = delta
        self.record = FlowCellRecord(deltat, name, eff_init, self.installed_surface_init, stockMax_init, self.K1, self.K2)
        
    
    def actualize(self, inputs):
        self.efficiency = inputs[0]
        self.K1 = inputs[1]
        self.K2 = inputs[2]
        self.delta = inputs[3]


    def installed_surface(self):
        installed_surface = self.installed_surface_init*math.log(1+self.K1/self.record.K1[0])
        if installed_surface > 1 :
            installed_surface = 1
        return installed_surface
        
    def stockMax(self):
        return self.stockMax_init*math.log(1+self.K2/self.record.K2[0])
    
        
    def iterate(self, Gused):
        self.K1 = self.K1*(1 - self.deltat*self.delta)
        self.K2 = self.K2*(1 - self.deltat*self.delta)
        
        installed_surface = self.installed_surface()
        stockMax = self.stockMax()
        
        G = self.efficiency * installed_surface * self.incidentFlow
        self.Xb = G * self.deltat
        self.Xs = self.Xs + self.Xb - Gused * self.deltat
        
        if self.Xs > stockMax :
            self.Xs = stockMax
        
        self.record.actualize(G, Gused, self.Xs, self.Xb, self.efficiency, installed_surface, stockMax, self.K1, self.K2)
        
        
    def plot(self, indFigure):
        record = self.record
        plt.figure(figsize = (20, 14), tight_layout = True)
        
        fs=25
        
        plt.subplot(2, 3, 1)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.G, label='$G$', linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.Gused, label='$G_{used}$', ls='None', marker='x', markersize=5, markevery=30, color=choice_colors1[2])
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Production',fontsize=fs)
        plt.legend(fontsize = fs)
        plt.grid(False)
        
        plt.subplot(2, 3, 2)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.Xs, label='Stock',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.stockMax, label='Stock$_{Max}$', linestyle='-.',linewidth=4, color=choice_colors1[2])
        plt.ylabel('Stocks',fontsize=fs)
        plt.xlabel('Time',fontsize=fs)
        plt.legend(fontsize = fs)
        plt.grid(False)  
        
        plt.subplot(2, 3, 3)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.eff, label='efficiency', linewidth=4, color=choice_colors1[2])
        plt.ylabel('$\eta$',fontsize=fs)
        plt.xlabel('Time',fontsize=fs)
        #plt.legend(fontsize = fs)
        plt.grid(False)
        
        
        plt.subplot(2, 3, 4)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.K1, label='$K_1$',linewidth=4, color=choice_colors1[0])
        plt.plot(record.t, record.K2, label='$K_2$',linewidth=4, color=choice_colors1[2])
        plt.legend(fontsize = fs)
        plt.xlabel('Time',fontsize=fs)
        plt.grid(False)
        
        
        plt.subplot(2, 3, 5)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(width=5, direction="in", length=10, labelsize=20, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        plt.plot(record.t, record.installed_surface, linewidth=4, color=choice_colors[0])
        plt.ylabel('Installed Surface',fontsize=fs)
        plt.xlabel('Time',fontsize=fs)
        plt.grid(False)
                
#############################################################################################################################################
            
class Cell :
    
    def __init__(self, cell, type):
        self.type = type
        self.cell = cell
        
    def iterate(self, Gd, *args):
        if self.type == "flow" :
            return self.cell.iterate(Gd)
        if self.type == "stock" :
            return self.cell.iterate(Gd, args[0])
    
        
    def plot(self, indFigure):
        return self.cell.plot(indFigure)
    
    
    
    
#############################################################################################################################################
        
class PhysicalWorldRecord :
    
    def __init__(self, deltat, n_recipe, n_cells) :
        self.deltat = deltat
        self.t = [0]
        self.production = [[0]*n_recipe]
        self.request = [[0]*n_recipe]
        self.realEnergyMix = [[0]*n_cells]
        self.requestedEnergyMix = [[0]*n_cells]
        
    def actualize(self, production, request, realEnergyMix, requestedEnergyMix) :
        self.t.append(self.t[-1] + self.deltat)
        self.production.append(production)
        self.request.append(request)
        self.realEnergyMix.append(realEnergyMix)
        self.requestedEnergyMix.append(requestedEnergyMix)

#----------------------------------------------------------------------------------        
        
class PhysicalWorld :
    # Set of stock and flow sheets
    
    def __init__(self, deltat, cells, recipeMatrix, recipeRequestArray_init, requestedEnergyMix, recipeNames) :
        self.deltat = deltat
        self.cells = cells
        self.recipeMatrix = recipeMatrix      # Each column ndicates the recipe (materials and energy quantities) to produce one unit of a specific good per unit of time. This is the non-aggregated demand.
        self.n_cells = len(cells)    # Number of cells in the physical world
        self.recipeToRecycle = [-1]*self.n_cells  # Correspondence array between a resource and the position of the recipe to recycle it in the recipe matrix. Position = -1 if there is no recipe to recycle the resource.
        self.recipeNames = recipeNames
        print(recipeNames[0])
        self.recipeRequestArray = []
        for i in range(len(recipeRequestArray_init)):
            self.recipeRequestArray.append(recipeRequestArray_init[i])
        
        
        for i in range(self.n_cells) :            # Adding recycling recipes to recipeMatrix and filling recipeToRecycle
            if self.cells[i].type == "stock" :    
                ind = np.shape(recipeMatrix)[0]
                if self.cells[i].cell.recyclingEnergyFlux >= 0 :      # -1 for a resource that can't be recycled
                    array = [0]*(self.n_cells+1)
                    array[-1] = self.cells[i].cell.recyclingEnergyFlux  # Energy cost for recycling
                    self.recipeMatrix.append(array)   # Adding the new recipe
                    self.recipeToRecycle[i] = ind     # ind is the position of the recycling recipe for the cell resource i
                    self.recipeNames.append('Recycled ' +self.cells[i].cell.name)  # Adding the recipe name
                    self.recipeRequestArray.append(0)
                    
        self.requestedEnergyMix = requestedEnergyMix
        self.record = PhysicalWorldRecord(deltat, len(self.recipeRequestArray), self.n_cells)

    def isARecyclingRecipe(self, i):
        # Returns the location of the resource in self.cells if the recipe on line i in RecipeMatrix is a recycling recipe. Returns -1 otherwise
        for k in range(self.n_cells):
            if self.recipeToRecycle[k] == i :
                return k
        return -1
    
    
    
    def availableResource(self):
        availableResource =  [0]*(self.n_cells + 1)
        for i in range(self.n_cells) :
            availableResource[i] = (self.cells[i].cell.record.Xb[-1] + self.cells[i].cell.record.Xs[-1])/self.deltat   # Available flux of resource (buffer + stock)
            if self.cells[i].cell.isEnergy:
                availableResource[-1] += availableResource[i]
        return availableResource
       
        
    
    def actualizeRecipeRequest(self, newProdRequest) :  
        for i in range(len(self.recipeRequestArray)):
            self.recipeRequestArray[i] = newProdRequest[i]
    
    
    def test(self, array):
        for i in range(len(array)):
            if array[i] :
                return True
        return False
    
    def somme(self, req, j):  # Total quantity of resource j required 
        s = 0
        for i in range(len(req)):
            if req[i]:  # The good i is required
                s += self.recipeRequestArray[i]*self.recipeMatrix[i][j]   # Quantity of goods * quantity of resources per good
        return s

    def indMiniPos(self, l):    # Index of the smallest positive number (or -.1 if no positive numbers)
        ind = -0.1
        for i in range(len(l)):
            if l[i] > 0 :
                ind = i
                miniPos = l[i]
        for i in range(len(l)):
            if l[i] > 0 and l[i] < miniPos :
                miniPos = l[i]
                ind = i
        return ind
    
    
    def produce(self) :
        # Returns the 'used' array containing the quantities taken from each resource for production (Gused).
        # Also returns the quantities produced for each recipe in this iteration.
        n_ing = self.n_cells + 1
        n_req = len(self.recipeRequestArray)  # Number of types of goods required
        coeff = self.recipeRequestArray
        avail = self.availableResource()        # Available resources at the current state
        used = [0]*n_ing  # used resources                       
        prod = [0]*n_req  # produced goods

        disp = []
        for i in range(len(avail)):
            disp.append(avail[i])    # Simple copy of avail
    
        req = [True]*n_req    # True for a required good
        prodNeeded = [True]*n_req  
        s = []  # Will represent the priority of the resource
        k = 0
        
        while self.test(req) and self.test(prodNeeded):     # Something still needs to be produced
            array = []
            for j in range(n_ing):
                if self.somme(req, j) != 0:   # Resource j is required
                    array.append(disp[j]/self.somme(req, j))  # Priority of the resource
                else :
                    array.append(-1)
            s.append(array)
            
            j0 = self.indMiniPos(s[k]) # Resource with the highest priority
            if j0 != - 0.1 : 
                for i in range(n_req) :
                    if self.recipeMatrix[i][j0] > 0 and req[i] :
                        prod[i] = coeff[i]*min(s[k][j0], 1)  # Produces what is required or less if no resource availability is limited
                        for j in range(n_ing):
                            disp[j] -= self.recipeMatrix[i][j]*prod[i]  # less available resources
                            used[j] += self.recipeMatrix[i][j]*prod[i]  # more used resources
                        req[i] = False
                for i in range(n_req) :
                    if prod[i] >= coeff[i]:
                        prodNeeded[i] = False
                k += 1
            else :
                for i in range(n_req) :
                    req[i] = False
                

        # Distribution of energy usage among different energy cells
        tempNrjMix = []
        for j in range(len(self.requestedEnergyMix)):
            tempNrjMix.append(self.requestedEnergyMix[j])
        flags = []
        for j in range(self.n_cells):
            if self.requestedEnergyMix[j] > 0 :
                flags.append(True)
            else :
                flags.append(False)
        Utemp = 0  
        
        while self.test(flags):
            array = []
            for j in range(self.n_cells):
                if tempNrjMix[j] != 0 and flags[j]:
                    array.append(avail[j]/tempNrjMix[j])
                else :
                    array.append(-1)
            j0 = self.indMiniPos(array)
            if j0 != -0.1 :
                Utemptemp = Utemp
                for j in range(self.n_cells) :
                    if flags[j]:
                        add = min(tempNrjMix[j]*(used[-1] - Utemp), tempNrjMix[j]*array[j0])
                        used[j] += add
                        avail[j] -= add
                        Utemptemp += add
                tempNrjMix[j0] = 0
                flags[j0] = False
                Utemp = Utemptemp
                somme = sum(tempNrjMix)
                if somme > 0:
                    for j in range(len(tempNrjMix)):
                        tempNrjMix[j] = tempNrjMix[j]/somme
            else : # Exit the while loop if no suitable energy cell found
                for j in range(self.n_cells):
                    flags[j] = False

        return (used, prod)
    
    
    
    def computeEnergyMix(self, Gused) :
        energyMix = [0]*self.n_cells
        for i in range(self.n_cells):
            if self.cells[i].cell.isEnergy and Gused[-1] > 0 :
                energyMix[i] = Gused[i]/Gused[-1]
        return energyMix
    
    
    def quantityProduced(self):
        qty = [0]*len(self.recipeRequestArray)
        for i in range(len(qty)) :
            for k in range(len(self.record.t)) :
                qty[i] += self.record.production[k][i]*self.deltat
        return qty
    
    
    
    def iterate(self):
        # produce
        (Gused, produced) = self.produce()
        
        # actualize record
        request = []
        for i in range(len(self.recipeRequestArray)):
            request.append(self.recipeRequestArray[i])
        self.record.actualize(produced, request, self.computeEnergyMix(Gused), self.requestedEnergyMix)
        
        # iterate cells
        Fr = [0]*self.n_cells
        for i in range(self.n_cells):
            if self.cells[i].type =="stock" and self.cells[i].cell.recyclingEnergyFlux >= 0 :
                Fr[i] = produced[self.recipeToRecycle[i]]
        for i in range(len(self.cells)) :
            if self.cells[i].type =="stock":
                self.cells[i].iterate(Gused[i], Fr[i])
            else : 
                self.cells[i].iterate(Gused[i])
        
    
    
    def actualize(self, inputs):
        cellsInputs = inputs[0]
        globalInputs = inputs[1]
        for i in range(self.n_cells):
            self.cells[i].cell.actualize(cellsInputs[i])
        self.recipeMatrix = globalInputs[0]
        self.recipeRequestArray = globalInputs[1]
        self.requestedEnergyMix = globalInputs[2]
        
    
    
    
    def plot(self) :
        
        for k in range(self.n_cells):
            self.cells[k].plot(k+1) 
            
        plt.figure(figsize = (20, 14), tight_layout = True)
        mpl.rcParams['axes.linewidth'] = 10
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)

        fs = 40
        productionRecord = np.array(self.record.production)
        requestRecord = np.array(self.record.request)
        energyMix = np.array(self.record.realEnergyMix)
        qty = self.quantityProduced()
        for i in range(len(self.recipeRequestArray)) :
            plt.plot(self.record.t, productionRecord[:, i], label=self.recipeNames[i], color=choice_colors1[i], linewidth = 2)
            plt.xlabel('$Time$',fontsize=fs)
            plt.ylabel('$Production$',fontsize=fs)
            plt.grid(False)
            print("recipe {}, aggregated production : {}".format(i, qty[i]))
        plt.legend(fontsize = 30)
        plt.tight_layout()
        
        
        
        
        
        
        
        
        plt.figure(figsize = (20, 14), tight_layout = True)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)

        for i in range(len(self.recipeRequestArray)) :
            
            # Index of the production max
            Pos_max = np.argmax(productionRecord[:, i])
            # 0.5 * Number of production values greater than max/100
            Width_peak = int(0.5 * np.size(productionRecord[:, i][productionRecord[:, i]>(max(productionRecord[:, i])/10)]))
            
            plt.plot(self.record.t[Pos_max - Width_peak : Pos_max + Width_peak], productionRecord[:, i][Pos_max - Width_peak : Pos_max + Width_peak], label=self.recipeNames[i], color=choice_colors1[i], linewidth = 7)
            plt.plot(self.record.t[Pos_max - Width_peak : Pos_max + Width_peak], requestRecord[:, i][Pos_max - Width_peak : Pos_max + Width_peak], ls = '--', color=choice_colors2[i], label = "Requested", linewidth = 5)
            plt.xlabel('$Time$',fontsize=fs)
            plt.ylabel('$Production$',fontsize=fs)
            plt.grid(False)
        plt.legend(fontsize = 30)
        plt.tight_layout()
        
        
        
        
        
        
        
        
        
        plt.figure(figsize = (20, 14), tight_layout = True)
        plt.subplot(1, 2, 1)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        
        for i in range(self.n_cells):
            if self.cells[i].cell.isEnergy :
                plt.plot(self.cells[0].cell.record.t, energyMix[:, i], label='instant ' + str(self.cells[i].cell.name), color=choice_colors1[i], linewidth = 3)
                plt.xlabel('Time',fontsize=fs)
                plt.ylabel('Energy Mix',fontsize=fs)    
                plt.grid(False)
        plt.legend(fontsize=30)
        
        
        plt.subplot(1, 2, 2)
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(width=10, direction="in", length=18, labelsize=30, pad=20)
        ax.xaxis.set_tick_params(which = 'minor', width=2, length=0, labelsize=15)
        ax.yaxis.set_tick_params(which = 'minor', width=5, direction="in", length=13, labelsize=24)
        energyConsumption = np.zeros(len(self.cells[0].cell.record.t))
        for i in range(self.n_cells):
            if self.cells[i].cell.isEnergy :
                energyConsumption += np.array(self.cells[i].cell.record.Gused)
        plt.plot(self.cells[0].cell.record.t, energyConsumption, color=choice_colors1[3], linewidth = 3)
        plt.grid(False)
        plt.xlabel('Time',fontsize=fs)
        plt.ylabel('Energy Consumption',fontsize=fs) 
        plt.tight_layout()
        
        
        
        
        
        
        
        
        
        
        
        
        fig, axs = plt.subplots(self.n_cells, 1, sharex = True, figsize = (30, 21), tight_layout = True)
        fig.subplots_adjust(hspace=0)
        for i in range(self.n_cells):
            record = self.cells[i].cell.record
            t = np.array(record.t)
            if self.cells[i].type == "stock" :
                axs[i].plot(t, np.array(record.Xs)/(self.cells[i].cell.stock_cible*self.cells[i].cell.Xt), label='% target stock', linewidth=10, color = choice_colors1[0])
                axs[i].grid(False)
                hasStock = np.array(record.hasStock)
                isLimitated = np.array(record.isLimitated)
                ymaxi = max(record.Xs)/(self.cells[i].cell.stock_cible*self.cells[i].cell.Xt)
                
                collection = collections.BrokenBarHCollection.span_where(
                        t, ymin=0, ymax=ymaxi, where=hasStock == 1.5, facecolor='black', alpha=0.8)
                #axs[i].add_collection(collection)
                collection = collections.BrokenBarHCollection.span_where(
                    t, ymin=0, ymax=ymaxi, where=hasStock == 1.0, facecolor='grey', alpha=0.8)
                #axs[i].add_collection(collection)
                collection = collections.BrokenBarHCollection.span_where(
                    t, ymin=0, ymax=ymaxi, where=hasStock == 0.1, facecolor='grey', alpha=0.2)
                #axs[i].add_collection(collection)
                collection = collections.BrokenBarHCollection.span_where(
                    t, ymin=0, ymax=ymaxi/10.0, where=isLimitated == True, facecolor='red', alpha=0.2)
                #axs[i].add_collection(collection)
                
                
                ax = axs[i].twinx()
                ax.plot(t, np.array(record.muH) - np.array(record.muL),label='$\Delta \mu$', linewidth=10, color = choice_colors1[2])
                ax.legend(fontsize=35, loc='lower right')
                ax.yaxis.set_tick_params(width=10, direction="in", length=18, labelsize=40, pad=20, labelcolor=choice_colors1[2])
                
                
            if self.cells[i].type == "flow" :
                axs[i].plot(t, np.array(record.Xs)/(record.stockMax), label='% stock max', color = choice_colors[0], linewidth=10)
                axs[i].grid(False)
                stockMax=self.cells[i].cell.stockMax
                ymaxi = max(record.Xs)
                
                collection = collections.BrokenBarHCollection.span_where(
                        t, ymin=0, ymax=ymaxi, where=np.array(record.Xs)>1e-14, facecolor='grey', alpha=0.2)
                #axs[i].add_collection(collection)
                collection = collections.BrokenBarHCollection.span_where(
                        t, ymin=0, ymax=ymaxi, where=np.array(record.Xs) ==stockMax, facecolor='grey', alpha=0.8)
                #axs[i].add_collection(collection)
                
            axs[i].legend(fontsize=35)
            axs[i].set_title(self.cells[i].cell.name, loc='center', x=-0.1, y=0.5, rotation='horizontal', fontsize=50)
            axs[i].yaxis.set_tick_params(width=10, direction="in", length=18, labelsize=40, pad=20, labelcolor=choice_colors1[0])
            axs[i].xaxis.set_tick_params(width=10, direction="in", length=18, labelsize=40, pad=20)
            plt.tight_layout()
    

#######################################################################################################################################################    
#------------------------- GET PARAMETERS------------------------------------------------------------

def extractCellParameters(fichier, rep_p) :  # Read data from *.txt
    fichier = open(rep_p + fichier, "r")
    array = []
    line = fichier.readline()
    ind_line = 0
    while line != "" :
        splitline = line.split(": ")
        value = splitline[1].split("\n")[0]
        name = splitline[0]
        if ind_line > 1:
            if name == 'isEnergy ':
                value = value == 'True' or value == 'True '
            else :
                value = float(value)
                #print value
        array.append(value)
        line = fichier.readline()
        ind_line += 1
    # print(array)
    return array

def createCell(fichier, rep_p, stock_cible, alpha, delta, deltat):  # 
    p = extractCellParameters(fichier, rep_p)
    if p[0]== "stock" :
        return StockCell(deltat, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], stock_cible, alpha, delta, xc=p[11], x0=p[12])
    if p[0] == "flow" :
        return FlowCell(deltat, p[1], p[2], p[3], p[4], p[5], p[6], delta)
    
def extractWorldParameters(fichier, rep_p):  # 
    fichier = open(rep_p + fichier, "r")
    array = []
    line = fichier.readline()
    while line != "" :
        splitline = line.split(": ")
        name = splitline[0]
        value = splitline[1].split("\n")[0]
        if name == "alpha " or name == "delta " or name == "stock_cible ":
            value = float(value)
        elif name == "recipeMatrix " or name=="recipeRequestArray_init " or name == "energyMix " or name == "recipeNames ":
            value = value.split("], ")
            value[0] = value[0][1:]
            value[-1] = value[-1][:-1]
            for j in range(len(value)):
                if j < len(value)-1 :
                    value[j] += "]"
                value[j] = value[j].split(", ")
                if name == "recipeMatrix " :
                    value[j][0] = value[j][0][1:]
                    value[j][-1] = value[j][-1][:-1]
                if name != "recipeNames ":
                    for i in range(len(value[j])):            
                        value[j][i] = float(value[j][i])
                if name =="recipeRequestArray_init " or name =="energyMix " or name =="recipeNames ":
                    value = value[0]
        elif name == "cells ":
            value = value.split(", ")
            value[0] = value[0][1:]
            value[-1] = value[-1][:-1]
        array.append(value)    
        line = fichier.readline()
    return array

def createPhysicalWorld(fichier, rep_p, deltat):  # 
    p = extractWorldParameters(fichier, rep_p)
    #print p
    cells = []
    for i in range(len(p[6])):      # get resource cell content (Wood, oil...)
        cellName = p[5][i] + ".txt"  
        q = extractCellParameters(cellName, rep_p)
        cells.append(Cell(createCell(cellName, rep_p, p[0], p[1], p[2], deltat), q[0]))
    return PhysicalWorld(deltat, cells, p[3], p[4], p[6], p[-1])
