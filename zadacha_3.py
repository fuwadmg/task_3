import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, maxSize_m, dx, y_min, y_max):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, maxSize_m, dx), [0]*int(maxSize_m/dx))[0]
        self.ax.plot(probePos, 0, 'xr')
        self.ax.plot(sourcePos, 0, 'ok')
        self.ax.set_xlim(0, maxSize_m)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('E_z, В/м')
        self.ax.grid()
        
    def updateData(self, data, t):
        self.line.set_ydata(data)
        self.ax.set_title('t = {:.4f} нc'.format(t*1e9))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def off(self):
        plt.ioff()

    def drawProbe(self, probePos):
        self.ax.plot(probePos, 0, 'xr')

    def drawSource(self, sourcePos):
        self.ax.plot(sourcePos, 0, 'ok')
    
class Probe:
    def __init__(self, probePos, Nt, dt):
        self.Nt = Nt
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.Nt)
        
    def addData(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

def showProbeSignal(probe):
        fig, ax = plt.subplots(1,2)
        ax[0].plot(np.arange(0, probe.Nt*probe.dt, probe.dt), probe.E)
        ax[0].set_xlabel('t, c')
        ax[0].set_ylabel('E_z, B/м')
        ax[0].set_xlim(0, probe.Nt*probe.dt)
        ax[0].grid()
        sp = np.abs(fft(probe.E))
        sp = fftshift(sp)
        df = 1/(probe.Nt*probe.dt)
        freq = np.arange(-probe.Nt*df /2, probe.Nt*df/2, df)
        ax[1].plot(freq, sp/max(sp))
        ax[1].set_xlabel('f, Гц')
        ax[1].set_ylabel('|S|/|S_max|')
        ax[1].set_xlim(0, 100e6)
        ax[1].grid()
        plt.subplots_adjust(wspace = 0.4)
        plt.show()

eps = 7.5
W0 = 120*np.pi
Nt = 5100
Nx = 1760
c = 299792458
maxSize_m = 8.5
dx = maxSize_m/Nx
maxSize = int(maxSize_m/dx)
probePos = maxSize_m*0.8
sourcePos = maxSize_m/2
probePos_N = int(probePos/dx)
sourcePos_N = int(sourcePos/dx)
Sc = 1
dt = dx*np.sqrt(eps)*Sc/c
probe = Probe(probePos_N, Nt, dt)
display = FieldDisplay(maxSize_m, dx, -1.5, 1.5)
display.drawProbe(probePos)
display.drawSource(sourcePos)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)
Sc1 = Sc/np.sqrt(eps)
k = (Sc1-1)/(Sc1+1)
k1 = -1 / (1 / Sc1 + 2 + Sc1)
k2 = 1 / Sc1 - 2 + Sc1
k3 = 2 * (Sc1 - 1 / Sc1)
k4 = 4 * (1 / Sc1 + Sc1)
Ez_oldLeft1 = np.zeros(3)
Ez_oldLeft2 = np.zeros(3)
f_0 = 50e6
Nl = c/f_0/dx/np.sqrt(eps)
layer_x = Nx-10
loss = np.zeros(maxSize)
loss[layer_x:] = 0.03 
ceze = (1.0 - loss) / (1.0 + loss)
cezh = W0 / (eps * (1.0 + loss))
chye = 1/W0/(1+loss)
chyh = (1-loss)/(1+loss)
for q in range(1, Nt):
    Hy[1:] = chyh[1:]*Hy[1:] +chye[1:]*(Ez[:-1]-Ez[1:])
    Ez[:-1] = ceze[:-1]*Ez[:-1] + cezh[:-1]*(Hy[:-1] - Hy[1:])
    Ez[sourcePos_N] += np.sin(2 * np.pi/Nl*Sc * q)
    Ez[0] = (k1*(k2*(Ez[2]+Ez_oldLeft2[0])+k3*(Ez_oldLeft1[0]+Ez_oldLeft1[2]-Ez[1]-Ez_oldLeft2[1])-k4*Ez_oldLeft1[1])-Ez_oldLeft2[2])
    Ez_oldLeft2[:] = Ez_oldLeft1[:]
    Ez_oldLeft1[:] = Ez[:3]
    probe.addData(Ez)
    if q % 50 == 0:
        display.updateData(Ez, q*dt)

display.off()
showProbeSignal(probe)



