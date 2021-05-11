import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats as ss
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import math as m

high_errs_R = pd.read_csv('high_errs_R.csv')
high_errs_B = pd.read_csv('high_errs_B.csv')
high_errs_G = pd.read_csv('high_errs_G.csv')
#Imports large EVPA error data from each camera

high_errs = pd.concat([high_errs_R, high_errs_B, high_errs_G], axis=0)
high_errs.to_csv('high_errs')
#Complies all large EVPA error data into a sinle dataframe and save it to be used for each camera

dataR = pd.read_csv('EVPA_dataR.csv')
dataB = pd.read_csv('EVPA_dataB.csv')
dataG = pd.read_csv('EVPA_dataG.csv')

NOTdata = pd.read_csv('NOT_data.csv')
NOTdata = pd.DataFrame(NOTdata)

misc_data = pd.read_csv('misc_data.csv')
misc_data = pd.DataFrame(misc_data)
misc_data = misc_data.drop([384])
# Dropped due to extremely large error

steward_data = pd.read_csv('steward_data.csv')

def moving_average(array, av_size):
    return np.convolve(array, np.ones(av_size), 'valid') / av_size

av_size = 10 #larger av size will remove more of the fluctuation

x1 = dataR['mjd']
x2 = dataB['mjd']
x3 = dataG['mjd']
x4 = steward_data['mjd']
x5 = dataR['mag_src']
x6 = dataB['mag_src']
x7 = dataG['mag_src']

y1 = dataR['EVPA180']
y2 = dataB['EVPA180']
y3 = dataG['EVPA180']
y4 = steward_data['EVPA180']
y5 = dataR['per_p']
y6 = dataB['per_p']
y7 = dataG['per_p']

z1 = dataR['EVPA_err']
z2 = dataB['EVPA_err']
z3 = dataG['EVPA_err']
z4 = steward_data['evpa_err']
z5 = dataR['pol_err']
z6 = dataB['pol_err']
z7 = dataG['pol_err']
z8 = dataR['mag_src_err']
z9 = dataB['mag_src_err']
z10 = dataG['mag_src_err']

def mjd(met):
    mjdrefi=51910
    mjdreff=7.4287e-4
    mjd = mjdrefi+ met/86400.0
    return mjd

file = "BLLac_86400.lc.txt"

hdulist = fits.open(file)
cols=hdulist[1].columns
print(cols)
t = hdulist[1].data

mjd_list = []
for row in t:
    mid = (row['START'] + row['STOP']) / 2
    mjd_list.append(mjd(mid))
#Code supplied to extract Fermi flux data

fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(7,5))
ax1.plot(x1,y1,'.',markersize=2,color='red',label='Camera d')
ax1.errorbar(x1,y1,yerr=z1,linestyle='none',color='red',elinewidth=0.5)
ax1.set(ylabel='Camera d \n EVPA ($^\circ$)')
ax1.grid()

ax2.plot(x2,y2,'.',markersize=2,color='blue',label='Camera e')
ax2.errorbar(x2,y2,yerr=z2,linestyle='none',color='blue',elinewidth=0.5)
ax2.set(ylabel='Camera e \n EVPA ($^\circ$)')
ax2.grid()

ax3.plot(x3,y3,'.',markersize=2,color='green',label='Camera f')
ax3.errorbar(x3,y3,yerr=z3,linestyle='none',color='green',elinewidth=0.5)
ax3.set(xlabel='MJD')
ax3.set(ylabel='Camera f \n EVPA ($^\circ$)')
ax3.grid()
fig1.tight_layout()
fig1.align_labels()
plt.savefig('All_fig1', bbox_inches='tight', dpi=750) 

fig2 = plt.figure(2)
plt.plot(moving_average(x1,av_size),moving_average(y1,av_size),color='red',linewidth=0.5,label='Camera d')
plt.plot(moving_average(x2,av_size),moving_average(y2,av_size),color='blue',linewidth=0.5,label='Camera e')
plt.plot(moving_average(x3,av_size),moving_average(y3,av_size),color='green',linewidth=0.5,label='Camera f')
plt.xlabel('MJD',fontsize=14)
plt.ylabel('EVPA ($^\circ$)',fontsize=14)
plt.legend(bbox_to_anchor=(1,1),loc='best',fontsize=10)
plt.savefig('All_fig2' ,bbox_inches='tight', dpi=750) 

n1 = 247
n2 = 309 #zoomed in area of interest

# n1 = 0
# n2 = 63 #second zoomed in area of interest. Only uncomment if other is commented and all figs have been renamed to avoid overwriting

fig3, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize=(12,20))
ax1.plot(x1[n1:n2],y1[n1:n2],'.',markersize=8,color='red',label='Camera d')
ax1.errorbar(x1[n1:n2],y1[n1:n2],yerr=z1[n1:n2],linestyle='none',color='red',elinewidth=0.5)
ax1.set_ylabel('Camera d EVPA ($^\circ$)',fontsize=20)
ax1.tick_params(axis='both',labelsize='large')
ax1.grid()

ax2.plot(x1[n1:n2],y5[n1:n2],'.',markersize=8,color='red',label='Camera d')
ax2.errorbar(x1[n1:n2],y5[n1:n2],yerr=z5[n1:n2],linestyle='none',color='red',elinewidth=0.5)
ax2.set_ylabel('Camera d Pol (%)',fontsize=20)
ax2.tick_params(axis='both',labelsize='large')
ax2.grid()

ax3.plot(x2[n1:n2],y2[n1:n2],'.',markersize=8,color='blue',label='Camera e')
ax3.errorbar(x2[n1:n2],y2[n1:n2],yerr=z2[n1:n2],linestyle='none',color='blue',elinewidth=0.5)
ax3.set_ylabel('Camera e EVPA ($^\circ$)',fontsize=20)
ax3.tick_params(axis='both',labelsize='large')
ax3.grid()

ax4.plot(x1[n1:n2],y6[n1:n2],'.',markersize=8,color='blue',label='Camera d')
ax4.errorbar(x1[n1:n2],y6[n1:n2],yerr=z6[n1:n2],linestyle='none',color='blue',elinewidth=0.5)
ax4.set_ylabel('Camera e Pol (%)',fontsize=20)
ax4.tick_params(axis='both',labelsize='large')
ax4.grid()

ax5.plot(x3[n1:n2],y3[n1:n2],'.',markersize=8,color='green',label='Camera f')
ax5.errorbar(x3[n1:n2],y3[n1:n2],yerr=z3[n1:n2],linestyle='none',color='green',elinewidth=0.5)
ax5.set_ylabel('Camera f EVPA ($^\circ$)',fontsize=20)
ax5.tick_params(axis='both',labelsize='large')
ax5.grid()

ax6.plot(x1[n1:n2],y7[n1:n2],'.',markersize=8,color='green',label='Camera d')
ax6.errorbar(x1[n1:n2],y7[n1:n2],yerr=z7[n1:n2],linestyle='none',color='green',elinewidth=0.5)
ax6.set_ylabel('Camera f Pol (%)',fontsize=20)
ax6.set_xlabel('MJD',fontsize=20)
ax6.tick_params(axis='both',labelsize='large')
ax6.grid()
fig3.tight_layout()
fig3.align_labels()
plt.subplots_adjust(hspace=.2)
plt.savefig('All_fig3', bbox_inches='tight', dpi=750) 

fig4 = plt.figure(4, figsize=(10,5))
plt.plot(moving_average(x1[n1:n2],av_size),moving_average(y1[n1:n2],av_size),color='red',linewidth=0.5,label='Camera d')
plt.plot(moving_average(x2[n1:n2],av_size),moving_average(y2[n1:n2],av_size),color='blue',linewidth=0.5,label='Camera e')
plt.plot(moving_average(x3[n1:n2],av_size),moving_average(y3[n1:n2],av_size),color='green',linewidth=0.5,label='Camera f')
plt.xlabel('MJD',fontsize=14)
plt.ylabel('EVPA ($^\circ$)',fontsize=14)
plt.legend(bbox_to_anchor=(1,1),loc='best',fontsize=14)
fig4.tight_layout()
plt.savefig('All_fig4', bbox_inches='tight', dpi=750) 

fig5, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(7,5))
ax1.plot(x3[n1:n2],y3[n1:n2],'.',color='green',markersize=2,label='RINGO3 Camera f')
ax1.errorbar(x3[n1:n2],y3[n1:n2],yerr=z3[n1:n2],linestyle='none',color='green',elinewidth=0.5)
ax1.plot(x4,y4,'.',color='black',markersize=2,label='Steward Observatory')
ax1.errorbar(x4,y4,yerr=z4,linestyle='none',color='black',elinewidth=0.5)
ax1.plot(NOTdata['MJD'],NOTdata['EVPA'],'.',color='orange',markersize=2,label='NOT')
ax1.errorbar(NOTdata['MJD'],NOTdata['EVPA'],yerr=NOTdata['EVPA_err'],linestyle='none',color='orange',elinewidth=0.5)
ax1.plot(misc_data['MJD'],misc_data['EVPA180'],'.',color='purple',markersize=2,label='Misc')
ax1.errorbar(misc_data['MJD'],misc_data['EVPA180'],yerr=misc_data['EVPA_err'],linestyle='none',color='purple',elinewidth=0.5)
ax1.set(xlabel='MJD')
ax1.set(ylabel='EVPA ($^\circ$)')
ax1.legend(bbox_to_anchor=(1,1),loc='best',fontsize=8)

ax2.plot(mjd_list,t["FLUX_100_300000"],'.',color='black',markersize=2,label='Fermi Gamma-ray')
ax2.errorbar(x=mjd_list,y=t["FLUX_100_300000"],yerr=t["ERROR_100_300000"],linestyle='none',color='black',elinewidth=0.5)
ax2.set_xlim(58000,58500)    
ax2.set(xlabel='MJD')
ax2.set(ylabel='Flux \n $(photons/cm^2/s^1)$')
ax2.set_yscale('log')
ax2.legend(bbox_to_anchor=(1,1),loc='best',fontsize=8)
fig5.tight_layout()
fig5.align_labels()
plt.savefig('All_fig5' ,bbox_inches='tight', dpi=750) 

fig6, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(moving_average(x3[n1:n2],av_size),moving_average(y3[n1:n2],av_size),color='green',linewidth=0.5,label='Camera f')
ax1.plot(moving_average(x4,av_size),moving_average(y4,av_size),color='black',linewidth=0.5,label='Steward \nObservatory')
ax1.plot(moving_average(misc_data['MJD'],av_size),moving_average(misc_data['EVPA180'],av_size),color='purple',linewidth=0.5,label='Misc')
ax1.set(xlabel='MJD')
ax1.set(ylabel='EVPA ($^\circ$)')
ax1.legend(bbox_to_anchor=(1,1),loc='best',fontsize=10)

ax2.plot(moving_average(mjd_list,av_size),moving_average(t["FLUX_100_300000"],av_size),color='black',linewidth=0.5,label='Fermi \nGamma-ray')
ax2.set_xlim(58000,58500)    
ax2.set(xlabel='MJD')
ax2.set(ylabel='Flux \n $(photons/cm^2/s^1)$')
ax2.set_yscale('log')
ax2.legend(bbox_to_anchor=(1,1),loc='best',fontsize=10)
fig6.tight_layout()
fig6.align_labels()
plt.savefig('All_fig6', bbox_inches='tight', dpi=750) 

RBerr = (z8**2+z9**2)**(1/2)
RGerr = (z8**2+z10**2)**(1/2)
BGerr = (z9**2+z10**2)**(1/2)
#Error propagation for colour analysis

fig7 = plt.figure(figsize=(9,9))
plt.subplot(221)
plt.plot(x5,(x6-x7),'.',markersize=2,color='black')
plt.errorbar(x5,(x6-x7),xerr=z8,yerr=BGerr,linestyle='none',color='black',elinewidth=0.5)
slopeR, interceptR = np.polyfit(x5, (x6-x7), 1)
plt.plot(np.unique(x5), np.poly1d(np.polyfit(x5, (x6-x7), 1))(np.unique(x5)), color = 'purple')
P1, C1 = ss.spearmanr(x5,(x6-x7))
plt.title(''r'$\nabla$='+str(round((slopeR),2))+'\n C='+str('{:.1e}'.format(C1))+', P='+str(round((P1),2))+'',fontsize=14)
plt.xlabel('R',fontsize=14)
plt.ylabel('B-G',fontsize=14)
plt.xlim([13,16])
plt.gca().invert_xaxis()
plt.grid()

plt.subplot(222)
plt.plot(x6,(x7-x5),'.',markersize=2,color='black')
plt.errorbar(x6,(x7-x5),xerr=z9,yerr=RGerr,linestyle='none',color='black',elinewidth=0.5)
slopeB, interceptB = np.polyfit(x6, (x7-x5), 1)
plt.plot(np.unique(x6), np.poly1d(np.polyfit(x6, (x7-x5), 1))(np.unique(x6)), color = 'purple')
P2, C2 = ss.spearmanr(x6,(x7-x5))
plt.title(''r'$\nabla$='+str(round((slopeB),2))+'\n C='+str('{:.1e}'.format(C2))+', P='+str(round((P2),2))+'',fontsize=14)
plt.xlabel('B',fontsize=14)
plt.ylabel('G-R',fontsize=14)
plt.xlim([13,16])
plt.gca().invert_xaxis()
plt.grid()

plt.subplot(223)
plt.plot(x7,(x6-x5),'.',markersize=2,color='black')
plt.errorbar(x7,(x6-x5),xerr=z10,yerr=RBerr,linestyle='none',color='black',elinewidth=0.5)
slopeG, interceptG = np.polyfit(x7, (x6-x5), 1)
plt.plot(np.unique(x7), np.poly1d(np.polyfit(x7, (x6-x5), 1))(np.unique(x7)), color = 'purple')
P3, C3 = ss.spearmanr(x7,(x6-x5))
plt.title(''r'$\nabla$='+str(round((slopeG),2))+'\n C='+str('{:.1e}'.format(C3))+', P='+str(round((P3),2))+'',fontsize=14)
plt.xlabel('G',fontsize=14)
plt.ylabel('B-R',fontsize=14)
plt.xlim([13,16])
plt.gca().invert_xaxis()
plt.grid()
fig7.tight_layout()
fig7.align_labels()
plt.savefig('All_fig7', bbox_inches='tight', dpi=750) 

count=0
for i in mjd_list:
    if i>x1[n1]:
        count=count+1 #Finds number of flux values with mjd in period of interest

lc = len(t)-count #Finds index of first point within period of interest
lcdata = t[lc:len(t)] #Gets data within period of interest
lavnum = m.ceil(len(lcdata)/len(x1[n1:n2])) #Finds number of points needed to make up the average and round up
lc = len(t)-lavnum*len(x1[n1:n2]) #Finds new index of first point needed to mnake average work
lcdata = t[lc:len(t)] #Gets new data within period of interest
lcdataerr = t[lc:len(t)] #Same as above but with new name for errors
lcdata = np.asarray(lcdata["FLUX_100_300000"]).astype(np.float32) #Extracts required data into array
lcdataerr = np.asarray(lcdataerr["ERROR_100_300000"]).astype(np.float32) #Same as above but for errors

lcdataav = np.mean(lcdata[:(len(lcdata)//lavnum)*lavnum].reshape(-1,lavnum), axis=1) #Takes the average of every 17 (or lavnum) data points
lcdataaverr = np.mean(lcdataerr[:(len(lcdataerr)//lavnum)*lavnum].reshape(-1,lavnum), axis=1) #Same as above but for errors

fig8 = plt.figure()
plt.plot(lcdataav,y7[n1:n2],'.',markersize=2,color='black')
plt.errorbar(lcdataav,y7[n1:n2],xerr=lcdataaverr,yerr=z7[n1:n2],linestyle='none',color='black',elinewidth=0.5)
P7, C7 = ss.spearmanr(lcdataav,y7[n1:n2])
plt.title('C='+str(round((C7),2))+', P='+str(round((P7),2))+'',fontsize=12)
plt.xlabel('Fermi gamma-ray Flux \n $(photons/cm^2/s^1)$',fontsize=14)
plt.ylabel('RINGO3 Camera f Polarisation \n (%)',fontsize=14)
plt.grid()
fig8.tight_layout()
fig8.align_labels()
plt.savefig('All_fig8', bbox_inches='tight', dpi=750) 

fig9 = plt.figure(figsize=(9,9))
plt.subplot(221)
plt.plot(y5[n1:n2],y1[n1:n2],'.',markersize=2,color='red')
plt.errorbar(y5[n1:n2],y1[n1:n2],xerr=z5[n1:n2],yerr=z1[n1:n2],linestyle='none',color='red',elinewidth=0.5)
P4, C4 = ss.spearmanr(y5[n1:n2],y1[n1:n2])
plt.title('C='+str(round((C4),2))+', P='+str(round((P4),2))+'',fontsize=14)
plt.xlabel('Camera d Pol',fontsize=14)
plt.ylabel('Camera d EVPA',fontsize=14)
plt.grid()

plt.subplot(222)
plt.plot(y6[n1:n2],y2[n1:n2],'.',markersize=2,color='blue')
plt.errorbar(y6[n1:n2],y2[n1:n2],xerr=z6[n1:n2],yerr=z2[n1:n2],linestyle='none',color='blue',elinewidth=0.5)
P5, C5 = ss.spearmanr(y6[n1:n2],y2[n1:n2])
plt.title('C='+str(round((C5),2))+', P='+str(round((P5),2))+'',fontsize=14)
plt.xlabel('Camera e Pol',fontsize=14)
plt.ylabel('Camera e EVPA',fontsize=14)
plt.grid()

plt.subplot(223)
plt.plot(y7[n1:n2],y3[n1:n2],'.',markersize=2,color='green')
plt.errorbar(y7[n1:n2],y3[n1:n2],xerr=z7[n1:n2],yerr=z3[n1:n2],linestyle='none',color='green',elinewidth=0.5)
P6, C6 = ss.spearmanr(y7[n1:n2],y3[n1:n2])
plt.title('C='+str(round((C6),2))+', P='+str(round((P6),2))+'',fontsize=14)
# plt.title('C='+str('{:.1e}'.format(C6))+', P='+str(round((P6),2))+'',fontsize=14) #example of how to do rounded value with scientific notation
plt.xlabel('Camera f Pol',fontsize=14)
plt.ylabel('Camera f EVPA',fontsize=14)
plt.grid()
fig9.tight_layout()
fig9.align_labels()
plt.savefig('All_fig9', bbox_inches='tight', dpi=750) 

fig10 = plt.figure()
plt.plot(lcdataav,y3[n1:n2],'.',markersize=2,color='black')
plt.errorbar(lcdataav,y3[n1:n2],xerr=lcdataaverr,yerr=z3[n1:n2],linestyle='none',color='black',elinewidth=0.5)
P7, C7 = ss.spearmanr(lcdataav,y3[n1:n2])
plt.title('C='+str(round((C7),2))+', P='+str(round((P7),2))+'',fontsize=12)
plt.xlabel('Fermi gamma-ray Flux \n $(photons/cm^2/s^1)$',fontsize=14)
plt.ylabel('RINGO3 Camera f Polarisation \n (%)',fontsize=14)
plt.grid()
fig10.tight_layout()
fig10.align_labels()
plt.savefig('All_fig10', bbox_inches='tight', dpi=750) 

