import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats as ss
import pandas as pd

data1 = pd.read_csv('bllac-f-evpa.csv')
data = pd.DataFrame(data1) #loads first data into df

data2 = pickle.load(open('bllac-f.pkl','rb'))
data2 = pd.DataFrame(data2) #loads second data into df

data['pol_err'] = data2['pol_err'] #adds necessary data from both df into one df

data.sort_values(by=['mjd'], inplace=True)
# Must be set to mjd as default for graphs

data = data.reset_index(drop=True) #resets index number for df
    
data = data.drop([0,1,2,3,4,5,6,7,9,10,145,167,180,189,190,344,407,545])
# data = data.drop([6,5,189,180,190,167,2,3,1,4,7,407,0])
#cloud/shake or no/poor image
#WCS_err = 3

data = data.reset_index(drop=True) #resets index number for df

x = data['mjd']

x11 = data['mjd'] #keeps values from x11 with mjd<56638 for use with magnitude data
x1 = x11[x>56638] #drops values from x11 with mjd<56638 for use with polarisation data
x11 = x11.reset_index(drop=True)
x1 = x1.reset_index(drop=True)

x22 = data['mag_src']
x2 = x22[x>56638]
x22 = x22.reset_index(drop=True)
x2 = x2.reset_index(drop=True)

x3 = data['per_p']
x3 = x3[x>56638]
x3 = x3.reset_index(drop=True)

x4 = data['EVPA']
x4 = x4[x>56638]
x4 = x4.reset_index(drop=True)

x5 = data['q_value']
x5 = x5[x>56638]
x5 = x5.reset_index(drop=True)

x6 = data['u_value']
x6 = x6[x>56638]
x6 = x6.reset_index(drop=True)

y11 = data['mag_src_err']
y1 = y11[x>56638]
y11 = y11.reset_index(drop=True)
y1 = y1.reset_index(drop=True)

y2 = data['pol_err']
y2 = y2[x>56638]
y2 = y2.reset_index(drop=True)

y3 = data['EVPA_err']
y3 = y3[x>56638]
y3 = y3.reset_index(drop=True)

y4 = data['q_err']
y4 = y4[x>56638]
y4 = y4.reset_index(drop=True)

y5 = data['u_err']
y5 = y5[x>56638]
y5 = y5.reset_index(drop=True)

fig1 = plt.figure(1)
plt.plot(data['mjd'],data['mag_src'],'.',markersize=2,color='green')
plt.gca().invert_yaxis()
plt.xlabel('MJD')
plt.ylabel('Magnitude')
plt.errorbar(data['mjd'],data['mag_src'],yerr=data['mag_src_err'],linestyle='none',color='green',elinewidth=0.5)
fig1.align_labels()
plt.savefig('Gfig1', dpi=500)

fig2 = plt.figure(2)
plt.plot(x1,x3,'.',markersize=2,color='green')
plt.xlabel('MJD')
plt.ylabel('Polarisation (%)')
plt.errorbar(x1,x3,yerr=y2,linestyle='none',color='green',elinewidth=0.5)
fig2.align_labels()
plt.savefig('Gfig2', dpi=500)

fig3, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(7,5))
ax1.plot(data['mjd'],data['mag_src'],'.',markersize=2,color='green')
ax1.invert_yaxis()
ax1.set(ylabel='Magnitude')
ax1.errorbar(data['mjd'],data['mag_src'],yerr=data['mag_src_err'],linestyle='none',color='green',elinewidth=0.5)

ax2.plot(x1,x3,'.',markersize=2,color='green')
ax2.set(xlabel='MJD')
ax2.set(ylabel='Polarisation (%)')
ax2.errorbar(x1,x3,yerr=y2,linestyle='none',color='green',elinewidth=0.5)
fig3.tight_layout()
fig3.align_labels()
plt.savefig('Gfig3', dpi=500)

n1 = 2
n2 = 136
n3 = 288
n4 = 370
n5 = 418
n6 = 587
n7 = 589
#values indicating the 'breaks' in polarisation data apparent from first plots
#each value is the start of the new 'section'

plt.figure(4)
plt.plot(x3,x2,'.',markersize=2,color='green')
plt.errorbar(x3,x2,xerr=y2,yerr=y1,linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P1, C1 = ss.spearmanr(x3,x2)
plt.title('C='+str(round((C1),2))+', P='+str(round((P1),2))+'',fontsize=14)
plt.xlabel('Polarisation (%)',fontsize=14)
plt.ylabel('Magnitude',fontsize=14)
plt.savefig('Gfig4', dpi=500)

fig5 = plt.figure(figsize=(9,9))
plt.subplot(331)
plt.plot(x3[0:n1],x2[0:n1],'.',markersize=2,color='green')
plt.errorbar(x3[0:n1],x2[0:n1],xerr=y2[0:n1],yerr=y1[0:n1],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P2, C2 = ss.spearmanr(x3[0:n1],x2[0:n1])
plt.title('C='+str(round((C2),2))+', P='+str(round((P2),2))+'',fontsize=12)
plt.ylabel('Magnitude',fontsize=14)

plt.subplot(332)
plt.plot(x3[n1:n2],x2[n1:n2],'.',markersize=2,color='green')
plt.errorbar(x3[n1:n2],x2[n1:n2],xerr=y2[n1:n2],yerr=y1[n1:n2],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P3, C3 = ss.spearmanr(x3[n1:n2],x2[n1:n2])
plt.title('C='+str(round((C3),2))+', P='+str(round((P3),2))+'',fontsize=12)

plt.subplot(333)
plt.plot(x3[n2:n3],x2[n2:n3],'.',markersize=2,color='green')
plt.errorbar(x3[n2:n3],x2[n2:n3],xerr=y2[n2:n3],yerr=y1[n2:n3],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P4, C4 = ss.spearmanr(x3[n2:n3],x2[n2:n3])
plt.title('C='+str(round((C4),2))+', P='+str(round((P4),2))+'',fontsize=12)

plt.subplot(334)
plt.plot(x3[n3:n4],x2[n3:n4],'.',markersize=2,color='green')
plt.errorbar(x3[n3:n4],x2[n3:n4],xerr=y2[n3:n4],yerr=y1[n3:n4],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P5, C5 = ss.spearmanr(x3[n3:n4],x2[n3:n4])
plt.title('C='+str(round((C5),2))+', P='+str(round((P5),2))+'',fontsize=12)
plt.ylabel('Magnitude',fontsize=14)

plt.subplot(335)
plt.plot(x3[n4:n5],x2[n4:n5],'.',markersize=2,color='green')
plt.errorbar(x3[n4:n5],x2[n4:n5],xerr=y2[n4:n5],yerr=y1[n4:n5],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P6, C6 = ss.spearmanr(x3[n4:n5],x2[n4:n5])
plt.title('C='+str(round((C6),2))+', P='+str(round((P6),2))+'',fontsize=12)
plt.xlabel('Polarisation (%)',fontsize=14)

plt.subplot(336)
plt.plot(x3[n5:n6],x2[n5:n6],'.',markersize=2,color='green')
plt.errorbar(x3[n5:n6],x2[n5:n6],xerr=y2[n5:n6],yerr=y1[n5:n6],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P7, C7 = ss.spearmanr(x3[n5:n6],x2[n5:n6])
plt.title('C='+str(round((C7),2))+', P='+str(round((P7),2))+'',fontsize=12)
plt.xlabel('Polarisation (%)',fontsize=14)

plt.subplot(337)
plt.plot(x3[n6:n7],x2[n6:n7],'.',markersize=2,color='green')
plt.errorbar(x3[n6:n7],x2[n6:n7],xerr=y2[n6:n7],yerr=y1[n6:n7],linestyle='none',color='green',elinewidth=0.5)
plt.gca().invert_yaxis()
P8, C8 = ss.spearmanr(x3[n6:n7],x2[n6:n7])
plt.title('C='+str(round((C8),2))+', P='+str(round((P8),2))+'',fontsize=12)
plt.xlabel('Polarisation (%)',fontsize=14)
plt.ylabel('Magnitude',fontsize=14)
fig5.tight_layout()
fig5.align_labels()
plt.savefig('Gfig5', dpi=500)

Std1 = np.std(y3[0:n1])  
Std2 = np.std(y3[n1:n2]) 
Std3 = np.std(y3[n2:n3]) 
Std4 = np.std(y3[n3:n4]) 
Std5 = np.std(y3[n4:n5]) 
Std6 = np.std(y3[n5:n6]) 
Std7 = np.std(y3[n6:n7])
#standard deviation of errors for each section

stdlim = 1

high_evpa_err1 = pd.concat([x1[0:n1], y3[0:n1]], axis=1)
high_evpa_err1 = high_evpa_err1.drop(high_evpa_err1[high_evpa_err1['EVPA_err'] < stdlim*Std1].index)

high_evpa_err2 = pd.concat([x1[n1:n2], y3[n1:n2]], axis=1)
high_evpa_err2 = high_evpa_err2.drop(high_evpa_err2[high_evpa_err2['EVPA_err'] < stdlim*Std2].index)

high_evpa_err3 = pd.concat([x1[n2:n3], y3[n2:n3]], axis=1)
high_evpa_err3 = high_evpa_err3.drop(high_evpa_err3[high_evpa_err3['EVPA_err'] < stdlim*Std3].index)
    
high_evpa_err4 = pd.concat([x1[n3:n4], y3[n3:n4]], axis=1)
high_evpa_err4 = high_evpa_err4.drop(high_evpa_err4[high_evpa_err4['EVPA_err'] < stdlim*Std4].index)

high_evpa_err5 = pd.concat([x1[n4:n5], y3[n4:n5]], axis=1)
high_evpa_err5 = high_evpa_err5.drop(high_evpa_err5[high_evpa_err5['EVPA_err'] < stdlim*Std5].index)

high_evpa_err6 = pd.concat([x1[n5:n6], y3[n5:n6]], axis=1)
high_evpa_err6 = high_evpa_err6.drop(high_evpa_err6[high_evpa_err6['EVPA_err'] < stdlim*Std6].index)

high_evpa_err7 = pd.concat([x1[n6:n7], y3[n6:n7]], axis=1)
high_evpa_err7 = high_evpa_err7.drop(high_evpa_err7[high_evpa_err7['EVPA_err'] < stdlim*Std7].index)

high_evpa_errs = [high_evpa_err1, high_evpa_err2, high_evpa_err3, high_evpa_err4, high_evpa_err5, high_evpa_err6, high_evpa_err7]
high_evpa_errs = pd.concat(high_evpa_errs)
#creates df containing any data with errors greater than 'stdlim' standard deviations for each section

high_evpa_errs.to_csv('high_errs_G.csv')

high_errs = pd.read_csv('high_errs')

EVPA_data = pd.concat([x1,x4,y3,x3,y2,x5,y4,x6,y5,x2,y1], axis=1)
EVPA_data = EVPA_data[~EVPA_data.mjd.isin(high_errs.mjd)]
EVPA_data = EVPA_data.reset_index(drop=True)
EVPA_data = EVPA_data.drop([212]) #added as this value does not appear in red but does in blue and green
EVPA_data = EVPA_data.reset_index(drop=True)
#drops the data with errrors larger than 'stdlim' standard deviations

x7 = EVPA_data['EVPA']

for i in range(len(x7)-1):
    x7[-1]=x7[0]
    if abs(x7[i]-x7[i-1]) < abs((x7[i]+180)-x7[i-1]) and abs(x7[i]-x7[i-1]) < abs((x7[i]-180)-x7[i-1]):
        pass 
    else:
        if abs((x7[i]+180)-x7[i-1]) < abs(x7[i]-x7[i-1]) and abs((x7[i]+180)-x7[i-1]) < abs((x7[i]-180)-x7[i-1]):
            x7[i]=x7[i]+180
        else:
            if abs((x7[i]-180)-x7[i-1]) < abs(x7[i]-x7[i-1]) and abs((x7[i]-180)-x7[i-1]) < abs((x7[i]+180)-x7[i-1]):
                x7[i]=x7[i]-180
x8 = x7.drop([-1])
x8 = x8.reset_index(drop=True)
EVPA_data['EVPA180'] = x8
EVPA_data.to_csv('EVPA_dataG.csv')
#loop to check if adding or subtracting 180 from evpa gives better looking plot
#then adds new evpa data into origional dataframe in correct mjd position to account for data removal

plt.figure(6)
plt.plot(EVPA_data['mjd'],EVPA_data['EVPA180'],'.',markersize=2,color='green')
plt.xlabel('MJD',fontsize=14)
plt.ylabel('EVPA ($^\circ$)',fontsize=14)
plt.errorbar(EVPA_data['mjd'],EVPA_data['EVPA180'],yerr=EVPA_data['EVPA_err'],linestyle='none',color='green',elinewidth=0.5)
plt.savefig('Gfig6', dpi=500)

a1 = EVPA_data['EVPA_err']
a2 = EVPA_data['pol_err']
a3 = EVPA_data['q_err']
a4 = EVPA_data['u_err']

fig7, ax = plt.subplots(4,4,figsize=(12,12))
ax[0,0].hist(abs(a2),49,color='green',edgecolor='black')
ax[0,0].set(xlabel='Polarisation Error (%)',ylabel='Counts')
ax2 = ax[0,0].twinx()
ax2.hist(abs(a2),49,density=True,cumulative=True,histtype='step',color='purple')

ax[0,1].hist(abs(a2),49,color='green',edgecolor='black')
ax[0,1].set(xlabel='Polarisation Error (%)',xscale='log',yscale='log')
ax[0,1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[0,1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[0,1].get_yaxis().set_major_formatter(ScalarFormatter())
ax[0,1].minorticks_off()
ax2 = ax[0,1].twinx()
ax2.hist(abs(a2),49,density=True,cumulative=True,histtype='step',color='purple')

ax[0,2].hist(abs(a2),49,color='green',edgecolor='black')
ax[0,2].set(xlabel='Polarisation Error (%)',xscale='log')
ax[0,2].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[0,2].get_xaxis().set_major_formatter(ScalarFormatter())
ax[0,2].minorticks_off()
ax2 = ax[0,2].twinx()
ax2.hist(abs(a2),49,density=True,cumulative=True,histtype='step',color='purple')

ax[0,3].hist(abs(a2),49,color='green',edgecolor='black')
ax[0,3].set(xlabel='Polarisation Error (%)',yscale='log')
ax[0,3].get_yaxis().set_major_formatter(ScalarFormatter())
ax2 = ax[0,3].twinx()
ax2.hist(abs(a2),49,density=True,cumulative=True,histtype='step',color='purple')
ax2.set(ylabel='Cumulative Distribution')

ax[1,0].hist(abs(a1),49,color='green',edgecolor='black')
ax[1,0].set(xlabel='EVPA Error ($^\circ$)')
ax[1,0].set(ylabel='Counts')
ax2 = ax[1,0].twinx()
ax2.hist(abs(a1),49,density=True,cumulative=True,histtype='step',color='purple')

ax[1,1].hist(abs(a1),49,color='green',edgecolor='black')
ax[1,1].set(xlabel='EVPA Error ($^\circ$)',xscale='log',yscale='log')
# ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(2))
ax[1,1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[1,1].get_yaxis().set_major_formatter(ScalarFormatter())
ax[1,1].minorticks_off()
ax2 = ax[1,1].twinx()
ax2.hist(abs(a1),49,density=True,cumulative=True,histtype='step',color='purple')

ax[1,2].hist(abs(a1),49,color='green',edgecolor='black')
ax[1,2].set(xlabel='EVPA Error ($^\circ$)',xscale='log')
# ax[1,2].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[1,2].get_xaxis().set_major_formatter(ScalarFormatter())
ax[1,2].minorticks_off()
ax2 = ax[1,2].twinx()
ax2.hist(abs(a1),49,density=True,cumulative=True,histtype='step',color='purple')

ax[1,3].hist(abs(a1),49,color='green',edgecolor='black')
ax[1,3].set(xlabel='EVPA Error ($^\circ$)',yscale='log')
ax[1,3].get_yaxis().set_major_formatter(ScalarFormatter())
ax2 = ax[1,3].twinx()
ax2.hist(abs(a1),49,density=True,cumulative=True,histtype='step',color='purple')
ax2.set(ylabel='Cumulative Distribution')

ax[2,0].hist(abs(a3),49,color='green',edgecolor='black')
ax[2,0].set(xlabel='q Error',ylabel='Counts')
ax2 = ax[2,0].twinx()
ax2.hist(abs(a3),49,density=True,cumulative=True,histtype='step',color='purple')

ax[2,1].hist(abs(a3),49,color='green',edgecolor='black')
ax[2,1].set(xlabel='q Error',xscale='log',yscale='log')
ax[2,1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[2,1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[2,1].get_yaxis().set_major_formatter(ScalarFormatter())
ax[2,1].minorticks_off()
ax2 = ax[2,1].twinx()
ax2.hist(abs(a3),49,density=True,cumulative=True,histtype='step',color='purple')

ax[2,2].hist(abs(a3),49,color='green',edgecolor='black')
ax[2,2].set(xlabel='q Error',xscale='log')
ax[2,2].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[2,2].get_xaxis().set_major_formatter(ScalarFormatter())
ax[2,2].minorticks_off()
ax2 = ax[2,2].twinx()
ax2.hist(abs(a3),49,density=True,cumulative=True,histtype='step',color='purple')

ax[2,3].hist(abs(a3),49,color='green',edgecolor='black')
ax[2,3].set(xlabel='q Error',yscale='log')
ax[2,3].get_yaxis().set_major_formatter(ScalarFormatter())
ax2 = ax[2,3].twinx()
ax2.hist(abs(a3),49,density=True,cumulative=True,histtype='step',color='purple')
ax2.set(ylabel='Cumulative Distribution')

ax[3,0].hist(abs(a4),49,color='green',edgecolor='black')
ax[3,0].set(xlabel='u Error',ylabel='Counts')
ax2 = ax[3,0].twinx()
ax2.hist(abs(a4),49,density=True,cumulative=True,histtype='step',color='purple')

ax[3,1].hist(abs(a4),49,color='green',edgecolor='black')
ax[3,1].set(xlabel='u Error',xscale='log',yscale='log')
ax[3,1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[3,1].get_xaxis().set_major_formatter(ScalarFormatter())
ax[3,1].get_yaxis().set_major_formatter(ScalarFormatter())
ax[3,1].minorticks_off()
ax2 = ax[3,1].twinx()
ax2.hist(abs(a4),49,density=True,cumulative=True,histtype='step',color='purple')

ax[3,2].hist(abs(a4),49,color='green',edgecolor='black')
ax[3,2].set(xlabel='u Error',xscale='log')
ax[3,2].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[3,2].get_xaxis().set_major_formatter(ScalarFormatter())
ax[3,2].minorticks_off()
ax2 = ax[3,2].twinx()
ax2.hist(abs(a4),49,density=True,cumulative=True,histtype='step',color='purple')

ax[3,3].hist(abs(a4),49,color='green',edgecolor='black')
ax[3,3].set(xlabel='u Error',yscale='log')
ax[3,3].get_yaxis().set_major_formatter(ScalarFormatter())
ax2 = ax[3,3].twinx()
ax2.hist(abs(a4),49,density=True,cumulative=True,histtype='step',color='purple')
ax2.set(ylabel='Cumulative Distribution')
fig7.tight_layout()
fig7.align_labels()
plt.savefig('Gfig7', dpi=750) 

X1 = data['mjd']
Y1 = data['mag_src']
Z1 = data['mag_src_err']

X2 = EVPA_data['mjd']
Y2 = EVPA_data['EVPA180']
Z2 = EVPA_data['EVPA_err']

def moving_average(array, av_size):
    return np.convolve(array, np.ones(av_size), 'valid') / av_size

av_size = 10 #larger av size will remove more of the fluctuation

fig8, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(7,5))
ax1.plot(X1,Y1,'.',markersize=2,color='green')
ax1.errorbar(X1,Y1,yerr=Z1,linestyle='none',color='green',elinewidth=0.5)
ax1.plot(moving_average(X1,av_size),moving_average(Y1,av_size),color='purple',linewidth=0.5)
ax1.invert_yaxis()
ax1.set(ylabel='Magnitude')

ax2.plot(X2,Y2,'.',markersize=2,color='green')
ax2.errorbar(X2,Y2,Z2,linestyle='none',color='green',elinewidth=0.5)
ax2.plot(moving_average(X2,av_size),moving_average(Y2,av_size),color='purple',linewidth=0.5)
ax2.set(xlabel='MJD')
ax2.set(ylabel='EVPA ($^\circ$)')
fig8.tight_layout()
fig8.align_labels()
plt.savefig('Gfig8', dpi=750) 

X3 = EVPA_data['q_value']
Y3 = EVPA_data['q_err']

X4 = EVPA_data['u_value']
Y4 = EVPA_data['u_err']

plt.figure(9)
plt.title('Camera f',fontsize=14)
plt.scatter(X3[247:308],X4[247:308],c=X2[247:308],cmap='viridis')
plt.errorbar(X3[247:308],X4[247:308],xerr=Y3[247:308],yerr=Y4[247:308],linestyle='none',elinewidth=0.5)
plt.plot(0,0,'x',color='black')
plt.xlabel('q value',fontsize=14)
plt.xlim(-0.2,0.2)
plt.ylabel('u value',fontsize=14)
plt.ylim(-0.2,0.2)
plt.colorbar()
plt.savefig('Gfig9', dpi=500)










