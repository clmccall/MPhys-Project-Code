import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats as ss
import pandas as pd

data1 = pd.read_csv('steward_bllac.csv')
data = pd.DataFrame(data1)

x1 = data['mjd']

x2 = data['evpa']

y1 = data['evpa_err']

for i in range(len(x2)-1):
    x2[-1]=x2[0]
    if abs(x2[i]-x2[i-1]) < abs((x2[i]+180)-x2[i-1]) and abs(x2[i]-x2[i-1]) < abs((x2[i]-180)-x2[i-1]):
        pass 
    else:
        if abs((x2[i]+180)-x2[i-1]) < abs(x2[i]-x2[i-1]) and abs((x2[i]+180)-x2[i-1]) < abs((x2[i]-180)-x2[i-1]):
            x2[i]=x2[i]+180
        else:
            if abs((x2[i]-180)-x2[i-1]) < abs(x2[i]-x2[i-1]) and abs((x2[i]-180)-x2[i-1]) < abs((x2[i]+180)-x2[i-1]):
                x2[i]=x2[i]-180
x3 = x2.drop([-1])
x3 = x3.reset_index(drop=True)
data['EVPA180'] = x3 

data.to_csv('steward_data.csv')

cap1 = 'Plot showing the EVPA data from the Steward Observatory after adjustmnt for the 180$^\circ$ ambiguity.'

fig1 = plt.figure(1, figsize=(10,5))
plt.plot(x1,x3,'.',markersize=2,color='black')
plt.xlabel('MJD',fontsize=14)
plt.ylabel('EVPA ($^\circ$)',fontsize=14)
plt.errorbar(x1,x3,yerr=y1,linestyle='none',color='black',elinewidth=0.5)
fig1.text(0.51,-0.05,cap1,ha='center')
plt.savefig('steward_fig1', bbox_inches='tight',dpi=500)

def moving_average(array, av_size):
    return np.convolve(array, np.ones(av_size), 'valid') / av_size

av_size = 10 #larger av size will remove more of the fluctuation

cap2 = 'Plot showing the moving average of the adjusted EVPA data from the Steward Observatory. The moving average has a period of '+str(av_size)+'.'

fig2 = plt.figure(2, figsize=(10,5))
plt.plot(x1,x3,'.',markersize=2,color='black')
plt.plot(moving_average(x1,av_size),moving_average(x3,av_size),color='black',linewidth=0.5)
plt.xlabel('MJD',fontsize=14)
plt.ylabel('EVPA ($^\circ$)',fontsize=14)
fig2.text(0.51,-0.05,cap2,ha='center')
plt.savefig('Steward_fig2', bbox_inches='tight', dpi=750) 





