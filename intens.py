#!/usr/bin/python

# Pilot implementation of XAS and RIXS intensities calculation for Li2CuO2
# 
# The LSOP and WFs extraction parts initially done by Vamshi M. Katukuri (2013)
# Improved and expanded by Nikolay A. Bogdanov (2014)
#
# The RIXS and XAS parts by Nikolay A. Bogdanov (2014, 2015)
# Licensed under the GNU General Public License v3.0

# The script needs two input parameters: 
# 1. the Molpro output file 
# 2. extension to the output files 
# the output files are lsop.'n_states'_extension and wfc.'n_states'_extension
# here is an example:
# $ ./intens.py nnd6_r13.out _t2g_cas

#from sys import argv
import sys 
import math
import cmath
import numpy as np
import scipy as sp
from scipy.stats import cauchy
from scipy.stats import norm
from matplotlib.mlab import normpdf
#import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import collections as collections
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


filename = sys.argv[1]
typ = sys.argv[2]

f = open(filename).readlines()                     # read Molpro output file
tst = open('tst','w')

wfc_lookup = 'Eigenvectors of spin-orbit matrix' 
wfc_lookup_end = ' Composition of spin-orbit eigenvectors'


lsop_lookup_start = ' Spin-Orbit Matrix (CM-1)'
lsop_lookup_end = 'No symmetry adaption'

NOCI_lookup_start = ' Bi-orthogonal integral transformation finished'
NOCI_lookup_end = ' **********************************************************************************************************************************'

#lzImg_lookup = 'LZ (TRANSFORMED, IMAG)'
#lzRel_lookup = ' LZ (TRANSFORMED, REAL)'

LSOPstart_line=[]
LSOPend_line=[]
WFstart_line=[]
#WFstates_line=[]
NOCIstart_line=[]
NOCIend_line=[]
LSOPlines=False
#WFlines=False
NOCIlines=False


for num1, line in enumerate(f):                                     # Find all the blocks we want to extract
#
    if lsop_lookup_start in line:
        LSOPstart_line.append(num1+5)                               # Start of the Spin-Orbit Matrix data
        LSOPlines=True
#        print('Found LSOP at line', num1)
    if lsop_lookup_end in line and LSOPlines==True:
        LSOPend_line.append(num1-2)                                 # End of the Spin-Orbit Matrix data
        LSOPlines=False
#
    if wfc_lookup in line:
        WFstart_line.append(num1+7)                                 # Start of the wave function coefficients
#        WFlines=True
#        print('Found WF at line', num1)
#    if wfc_lookup in line and WFlines==True:
#        WFstates_line.append(num1-2)                                # Line of which the number of states is read
#        WFlines=False
#
    if NOCI_lookup_start in line :
        NOCIstart_line.append(num1+1)
        NOCIlines=True
    if NOCI_lookup_end in line and NOCIlines==True:
        NOCIend_line.append(num1-2)
        NOCIlines=False
#
#if LSOPlines==True or WFlines==True or  NOCIlines==True :
if LSOPlines==True or  NOCIlines==True :
    print('One of the blocks has start, but has no end')

if len(LSOPstart_line)!=len(WFstart_line):
    print('numbers of found WFs is',len(WFstart_line), 'L.S operators is',len(LSOPstart_line), 'may imply wrong output')

nstates=[]
WF=[]
LSOP=[]
SOC_E=[]
for i, num in enumerate(WFstart_line):
    nstates.append(int(f[num-10][0:4]))                             # last line of energies output
    WF.append(np.zeros((nstates[i],nstates[i]), dtype=np.complex128)) # fill WFs array with zeros
    WF[i]=WF[i]+100+100j                                            # fill WFs array with 100's to prevent errors
    LSOP.append(np.zeros((nstates[i],nstates[i]), dtype=np.complex128)) # 
    LSOP[i]=LSOP[i]+100+100j                                        #
    SOC_E.append(np.zeros(nstates[i]))                              # fill energies arrays with zeros
#
#print (nstates)
for ns, n_states in enumerate(nstates):
    wfc_fname = typ+".wfc."+str(ns)+"."+str(n_states)+"s"           # create files for WFs
    wfc_out = open(wfc_fname,'w')
    lsop_fname = typ+".lsop."+str(ns)+"."+str(n_states)+"s"         # create files for L.S operators
    lsop_out = open(lsop_fname,'w')
    socE_fname = typ+".esoc."+str(ns)+"."+str(n_states)+"s"         # create files for energies
    socE_out = open(socE_fname,'w')
#
#############
###### LSOP block details
#############
#
    lsop_bcs_d = 10                                                 # way it is printed in molpro
    lsop_block_col_size = lsop_bcs_d                                # equal to default if n_states < default
    if n_states <= lsop_block_col_size:
        lsop_n_blocks = 1                                           
        lsop_block_col_size = n_states
        lsop_last_block_col_size = 0                  
    else: 
        lsop_n_blocks= n_states/lsop_block_col_size                 # No of blocks the matrix is written
        if lsop_n_blocks*lsop_block_col_size != n_states:
            lsop_last_block_col_size = n_states-(lsop_n_blocks*lsop_block_col_size)
            lsop_n_blocks=lsop_n_blocks+1
        else:
            lsop_last_block_col_size = 0
#
#############
###### WFC block details
#############
#
    wfc_bcs_d = 8                                                   # way it is printed in molpro
    wfc_block_col_size = 8                                          # equal to default if n_states < default
    if n_states <= wfc_block_col_size:                              # if we have less than 8 SO-states
        wfc_n_blocks = 1
        wfc_block_col_size = n_states
        wfc_last_block_col_size = 0
    else: 
        wfc_n_blocks= n_states/wfc_block_col_size                   # find out how many 8-column blocks are printed
        if wfc_n_blocks*wfc_block_col_size != n_states:             # if number of SO-states isn't 8*N
            wfc_last_block_col_size = n_states-(wfc_n_blocks*wfc_block_col_size) # add last block with width <8
            wfc_n_blocks=wfc_n_blocks+1
        else:
            wfc_block_col_size = 8                                  # if number of SO-states is 8*N
            wfc_last_block_col_size = 0                             # no additional blocks needed
    wfc_block_rows = n_states*3                            # row size for each block
#    print(wfc_block_rows)
    wfc_block_gap = 3
    lop_block_rows = n_states*3                            # row size for each block 
    lop_block_gap = 3
#
#############
###### Extraction of WFC coefficients
#############
#
    for nbl in range(0,wfc_n_blocks):                               # loop over the No of blocks
        if wfc_last_block_col_size != 0 and nbl == wfc_n_blocks-1:  # make less columns for last block 
            wfc_block_col_size = wfc_last_block_col_size            # 
#                                                                   # specify block of 2*Nroots rows and 8 columns
        for n in range(WFstart_line[ns] + nbl*wfc_block_rows + nbl*wfc_block_gap, WFstart_line[ns] + (nbl+1)*wfc_block_rows + nbl*wfc_block_gap) : 
                if f[n]!='\n' and f[n]!=' \n' :                     # not to consider empty lines
                    if f[n][2]!=' ' :                               # rows with real parts also contain No, and (j, mj) information (only No is now used)
                        for col in range(0,wfc_block_col_size) :
#                            print WF[ns]
#                                                                   # select only relevant part of the line and read
                            WF[ns][int(f[n].split()[0])-1,col+nbl*wfc_bcs_d]=complex(float(f[n][22:135].split()[col]),0)
                    else :                                          # rows with imaginary parts have spaces in the beginning
                        for col in range(0,wfc_block_col_size) :
#                                                                   #  col+nbl*8 is 'absolute' No of column
                            WF[ns][int(f[n-1].split()[0])-1][col+nbl*wfc_bcs_d]=WF[ns][int(f[n-1].split()[0])-1][col+nbl*wfc_bcs_d]+complex(0,float(f[n][22:135].split()[col])) 
    for nr in range(n_states) :                                     # print to file
        prnt=''                                                     # specify empty line, then 'collect' the whole row
        for nc in range(n_states):
            prnt += str("%11.9f"%WF[ns].real[nr,nc])+str("+I(%11.9f) "%WF[ns].imag[nr,nc]) # normal output, WFs in columns
#            prnt += str("%11.9f"%WF[ns].real[nc,nr])+str("+I(%11.9f) "%WF[ns].imag[nc,nr]) # transposed output, WFs in rows
        print >> wfc_out, prnt
    wfc_out.close()
#
#############
###### LSOP extraction
#############
#
    for nbl in range(0,lsop_n_blocks):                              # loop over the No of blocks 
        if lsop_last_block_col_size != 0 and nbl == lsop_n_blocks-1:
            lsop_block_col_size = lsop_last_block_col_size 
#                                                                   # loop over block lines
        for n in range(LSOPstart_line[ns]+nbl*lop_block_rows+nbl*lop_block_gap, LSOPstart_line[ns]+(nbl+1)*lop_block_rows+nbl*lop_block_gap):
                if f[n]!='\n' and f[n]!=' \n' :                     # not to consider empty lines
                    if f[n][2]!=' ' :                               # rows with real parts also contain No, and (j, mj) information (only No is now used)
                        for col in range(0,lsop_block_col_size) :
                            LSOP[ns][int(f[n].split()[0])-1][col+nbl*lsop_bcs_d]=complex(float(f[n][19+11*col:30++11*col]),0) # select only relevant part of the line and read
                    else :                                          # rows with imaginary parts have spaces in the beginning
                        for col in range(0,lsop_block_col_size) :
                            LSOP[ns][int(f[n-1].split()[0])-1][col+nbl*lsop_bcs_d]=LSOP[ns][int(f[n-1].split()[0])-1][col+nbl*lsop_bcs_d]+complex(0,float(f[n][19+11*col:30+11*col])) # 19 symbol in line is the first for LSOP, every number is 11 symbols => 30 - end of the first number
#
    for nr in range(n_states) :                                     # print to file
        prnt=''                                                     # specify empty line, then 'collect' the whole row
        for nc in range(n_states):
            prnt += str(LSOP[ns].real[nr,nc])+"+I("+str(LSOP[ns][nr][nc].imag)+") " # normal output, same way as in molpro 
#            prnt += str("%11f"%LSOP[ns][nr][nc].real)+str("+I(%11f) "%LSOP[ns].imag[nr,nc]) # normal output, same way as in molpro (other format(?))
#            prnt += str("%11f"%LSOP[ns].real[nc,nr])+str("+I(%11f) "%LSOP[ns].imag[nc,nr]) # transposed output, WFs in rows
        print >> lsop_out, prnt
    lsop_out.close()
#
#############
###### Energies extraction
#############
#
    for n in range(LSOPend_line[ns]+10,WFstart_line[ns]-9):
        SOC_E[ns][int(f[n].split()[0])-1]=float(f[n].split()[1])  # *27.211396
#    for nr in range(n_states) :
        print >> socE_out,SOC_E[ns][int(f[n].split()[0])-1]
    socE_out.close()
#    
#
#############
###### NOCI block details
###### Find out dimensions (dNCI[i]) of each NOCI block (nN), if any
#############
#
dNCI=[]
D=[[] for ni in range(3)]                                           # DX=D[0], DY=D[1], DZ=D[2]
dN=0
#
#print NOCIstart_line
for nN in range(len(NOCIstart_line)) :
    for lin in f[NOCIstart_line[nN]:NOCIend_line[nN]] :
        for char in ['<','>','|']:
            lin=lin.replace(char, ' ')
        if  lin[0]!='\n' and lin.split()[0]=='!MRCI' and lin.split()[3]=='H' :
#            print lin
            dN=max(dN,lin.split()[2],lin.split()[4])
#    print dN
    dN=int(float(dN)-0.1)
    dNCI.append(dN)
    for x in range(3):
        D[x].append(np.zeros((dN,dN)))
#print dNCI,DX
#
#############
###### Reading and writing dipole moment matrices from NOCI blocks
#############
#
for nN, dN in enumerate(dNCI):
    dm_bname = ".dm."+str(nN)+"."+str(dN)+"s."                       # base name for dipole moments files
    fileDX = open(typ+dm_bname+"X",'w')                              # dm.0.8s.X_ci
    fileDY = open(typ+dm_bname+"Y",'w')
    fileDZ = open(typ+dm_bname+"Z",'w')                         
#
    for lin in f[NOCIstart_line[nN]:NOCIend_line[nN]] :
        for char in ['<','>','|']:                                   # replace brackets with space
            lin=lin.replace(char, ' ')
        if  lin[0]!='\n' and lin.split()[0]=='!MRCI' and lin.split()[3][0:2]=='DM' : # find DM lines
            if lin.split()[3] == 'DMX' :                                                                
                D[0][nN][int(float(lin.split()[2]))-1][int(float(lin.split()[4]))-1]=lin.split()[5]      # save matrix element
                if D[0][nN][int(float(lin.split()[4]))-1][int(float(lin.split()[2]))-1]==0 :             # if trere is no transpose-congugate ME
                    D[0][nN][int(float(lin.split()[4]))-1][int(float(lin.split()[2]))-1]=lin.split()[5]  # write this matrix element also there
            elif lin.split()[3] == 'DMY' :
                D[1][nN][int(float(lin.split()[2]))-1][int(float(lin.split()[4]))-1]=lin.split()[5]
                if D[1][nN][int(float(lin.split()[4]))-1][int(float(lin.split()[2]))-1]==0 :
                    D[1][nN][int(float(lin.split()[4]))-1][int(float(lin.split()[2]))-1]=lin.split()[5]
            elif lin.split()[3] == 'DMZ' :
                D[2][nN][int(float(lin.split()[2]))-1][int(float(lin.split()[4]))-1]=lin.split()[5]
                if D[2][nN][int(float(lin.split()[4]))-1][int(float(lin.split()[2]))-1]==0 :
                    D[2][nN][int(float(lin.split()[4]))-1][int(float(lin.split()[2]))-1]=lin.split()[5]
            else :
                print('Something is wrong !MRCI <i.1|DM_|j.1> lines, at line:',lin)
    for item in D[0][nN] :
        print >> fileDX, item[0], " ".join(map(str, item[1:]))
    for item in D[1][nN] :
        print >> fileDY, item[0], " ".join(map(str, item[1:]))
    for item in D[2][nN] :
        print >> fileDZ, item[0], " ".join(map(str, item[1:]))
#
#############
###### XAS and RIXS intensities calculation for Cu2+ d9, special case
###### Here WF[0] - core hole WFs, WF[1] - valence WFs, WF[2] - alternative core hole WFs
###### First construct 'total' WF matrix 16x16
#############
#
n_GS=2                                                              # GS is doubly degenerate S=+-1/2
E_ch=SOC_E[0]*27.211396
E_v=SOC_E[1]*27.211396
WF_ch = WF[0]                                                       # WF6  = WF[0]
WF_v = WF[1]                                                        # WF10 = WF[1]
CHns = nstates[0]
Vns  = nstates[1]
WF_tot = np.asmatrix(np.zeros((Vns+CHns,Vns+CHns), dtype=np.complex128))
WF_tot[0 : Vns/2, 0 : Vns] = WF_v[0 : Vns/2, :]                     # WF16[0:5,0:10]=WF10[0:5,:]
WF_tot[Vns/2 : Vns/2+CHns/2, Vns : Vns+CHns] = WF_ch[0 : CHns/2, :] # WF16[5:8,10:16]=WF6[0:3,:]
WF_tot[Vns/2+CHns/2 : Vns+CHns/2, 0 : Vns]=WF_v[Vns/2 : Vns, :]     # WF16[8:13,0:10]=WF10[5:10,:]
WF_tot[Vns+CHns/2 : Vns+CHns, Vns : Vns+CHns] = WF_ch[CHns/2 : CHns, :] # WF16[13:16,10:16]=WF6[3:6,:]
#coord=['X','Y','Z']                        # molpro notation
#ncoord=[0,1,2]
coord=['Z','X','Y']                        # Valentina's notation
ncoord=[1,2,0]                             # to get X-Y-Z order for plotting
Dsmall=[[] for ni in range(3)]                                      # DX=D[0], DY=D[1], DZ=D[2]
D_tot=[[] for ni in range(3)]                                       # DX=D[0], DY=D[1], DZ=D[2]
#
## Rotation to the scattering geometry
#
gamma=math.radians(68.7026)         # angle between plaquette plane (100) and scattering plane (101)
Drot=[[] for ni in range(3)]        # DM operator is described by three matrices
##################################################
# axes: 0=a, 1=b, 2=c MOLPRO, crystallographic   #
#       x=1, y=2, z=0 Valentina b->x, c->y, a->z #
#       x'=x, y'=y*sinG-z*cosG, z'=y*cosG+z*sinG #
##################################################
Drot[1]=D[1][0]                                                       # Drot[x']=D[x][first NOCI in the file]
Drot[2]=D[2][0]*math.sin(gamma)-D[0][0]*math.cos(gamma)               # Drot[y']=D[y]*sin(gam)-D[z]*cos(gam)
Drot[0]=D[2][0]*math.cos(gamma)+D[0][0]*math.sin(gamma)               # Drot[z']=D[y]*cos(gam)+D[z]*sin(gam)
#
## Calculate D in SOC basis
#
for x in range(3):
    Dsmall[x]=Drot[x]/0.52917721092                                              # using rotated DM TME, transform to Aengstrems
    D_tot[x]=np.asmatrix(np.zeros((Vns+CHns,Vns+CHns), dtype=np.complex128))
    D_tot[x][0:Vns/2+CHns/2,0:Vns/2+CHns/2]=Dsmall[x]
    D_tot[x][Vns/2+CHns/2:Vns+CHns,Vns/2+CHns/2:Vns+CHns]=Dsmall[x]
    D_tot[x]=WF_tot.H*D_tot[x]*WF_tot                                                               # D_SOC=Transpose(Congugate(WF)).D_tot.WF
    Dsmall[x]=D_tot[x][Vns:Vns+CHns,0:Vns]                                                          # Dsmall[x]=D_tot[x][10:16,1:10]
#
##
### XAS spectra  for experimental geometry
### sigma, pi polarization; theta=10, 90
##
#
thetaXAS=[10,90]                                                      # Specify angles of incomming beam
#
xX=np.linspace(min(E_ch-E_v[0])-10,max(E_ch-E_v[0])+10,5000)          # range for the plot (Emin-10 eV to Emax+10 eV), 1000 points
xX=np.sort(np.append(xX,E_ch-E_v[0]))                                 # add resonance energies to incident range array (and sort)
XAsig=[np.zeros_like(xX, dtype=np.float64) for theta in thetaXAS]                   # XAS sigma plots in xX space (new)
XApi=[np.zeros_like(xX, dtype=np.float64) for theta in thetaXAS]                    # XAS pi plots in xX space (new)
#
#
XAS_sigPl=[np.zeros_like(xX) for theta in thetaXAS]                   # XAS sigma plots in xX space
XAS_piPl=[np.zeros_like(xX) for theta in thetaXAS]                    # XAS pi plots in xX space
#
XAS_sigI=[np.zeros(CHns) for theta in thetaXAS]                       # XAS sigma intensities
XAS_piI=[np.zeros(CHns) for theta in thetaXAS]                        # XAS pi intensities
#
Gch=1.0                          # 1 eV
# prefact=4*np.pi**2/137*(0.52917721092**2) # \AA**2
# prefact=4*np.pi**2/137*(0.52917721092**2)*1e-16 # cm**2
# prefact=4*np.pi**2/137*(0.52917721092**2)*1e8 # barn
prefact=4*np.pi**2/137*(0.52917721092**2)*1e2 # Megabarn
#
## scattering plane is x'z'
## sigma = y', pi = x'*sinTh + z'*cosTh
#
for Nth, theta in enumerate([math.radians(degr) for degr in thetaXAS]):               # for theta in thetaXAS [radians]
    Dsig=Dsmall[2]                                                    # sigma = y'
    Dpi =Dsmall[1]*math.sin(theta)+Dsmall[0]*math.cos(theta)          # pi = x'*sinTh + z'*cosTh
#
### Need to think, maybe one needs to calculate and mix afterwards (!)
#
####\begin{new}
    for i,Ei in enumerate(xX):
        # print Nth, theta, i, Ei
        sig_gs, pi_gs =0.,0.
        for gst in range(n_GS):
            sig_ch, pi_ch =0.,0.
            for chst in range(CHns):
                broad_ch = Gch/(2*np.pi)/((E_v[gst]+Ei-E_ch[chst])**2 +(Gch**2)/4)
                sig_ch += Dsig[chst,gst]*Dsig[chst,gst].conj()*broad_ch
                pi_ch  += Dpi[chst,gst]*Dpi[chst,gst].conj()*broad_ch
                # print i,Ei,broad_ch,sig_ch, pi_ch
            if sig_ch.imag!=0 : print sig_ch
            if pi_ch.imag!=0 : print pi_ch
            sig_gs += sig_ch/n_GS
            pi_gs  += pi_ch/n_GS
        XAsig[Nth][i] = sig_gs*Ei*prefact
        XApi[Nth][i]  = pi_gs*Ei*prefact
        # print Ei*prefact
        # print sig_gs, pi_gs
    # pl_sig=zip(xX,XAsig[Nth])
    # pl_pi =zip(xX,XApi[Nth])
    # fig=plt.figure(300,figsize=(7,5),dpi=200)
####\end{new}
    for state in range(n_GS):
        if state==0 :
#            print theta
            XAS_sigI[Nth]=np.multiply(Dsig[:,state],Dsig.conj()[:,state]).real           # Intens[X]=sum_GS[ sum_i[ |<i|DX|GS>|^2 ]]
            XAS_piI[Nth]=np.multiply(Dpi[:,state],Dpi.conj()[:,state]).real              # Intens[X]=sum_GS[ sum_i[ |<i|DX|GS>|^2 ]]
        else:
            XAS_sigI[Nth]=XAS_sigI[Nth]+np.multiply(Dsig[:,state],Dsig.conj()[:,state]).real
            XAS_piI[Nth]=XAS_piI[Nth]+np.multiply(Dpi[:,state],Dpi.conj()[:,state]).real
#        print 'XAS_sigI.imag[',state,'] = ', np.multiply(Dsig[:,state],Dsig.conj()[:,state]).imag
#        print 'XAS_piI.imag[',state,'] = ', np.multiply(Dpi[:,state],Dpi.conj()[:,state]).imag
    for st in range(CHns):
            XAS_sigPl[Nth]=XAS_sigPl[Nth]+cauchy(E_ch[st]-E_v[0],0.5).pdf(xX)*float(XAS_sigI[Nth][st])
            XAS_piPl[Nth]=XAS_piPl[Nth]+cauchy(E_ch[st]-E_v[0],0.5).pdf(xX)*float(XAS_piI[Nth][st])
#
##new
print sp.integrate.simps(XAsig[0],dx=xX[1]-xX[0])
print sp.integrate.simps(XAsig[1],dx=xX[1]-xX[0])
print sp.integrate.simps(XApi[0],dx=xX[1]-xX[0])
print sp.integrate.simps(XApi[1],dx=xX[1]-xX[0])
XAmax=max(np.nanmax(XAsig),np.nanmax(XApi))
fig=plt.figure(110,dpi=300)
for Nth, theta in enumerate(thetaXAS):
    sigNth=plt.subplot(len(thetaXAS),2,2*len(thetaXAS)-1-2*Nth)
    plt.plot(xX,XAsig[Nth])
    plt.ylim([0,XAmax])
    plt.gca().yaxis.set_major_locator(MultipleLocator(XAmax))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(XAmax/2))
#
    # if len(thetaXAS)-Nth==1:
    if Nth==0:
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    else:
        sigNth.axes.get_xaxis().set_visible(False)
        sigNth.axes.get_yaxis().set_visible(False)
#
    piNth=plt.subplot(len(thetaXAS),2,2*len(thetaXAS)-2*Nth)
    plt.plot(xX,XApi[Nth])
    plt.ylim([0,XAmax])
    plt.gca().yaxis.set_major_locator(MultipleLocator(XAmax))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(XAmax/2))
    # if len(thetaXAS)-Nth==1:
    if Nth==0:
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    else:
        piNth.axes.get_xaxis().set_visible(False)
        piNth.axes.get_yaxis().set_visible(False)
#
plt.subplots_adjust(hspace=.2,wspace=.2)
plt.savefig(typ+'.XA_boxes.pdf',dpi=300)
##\new
sys.exit("Fine for now")
xmax=max(np.amax(XAS_sigPl),np.amax(XAS_piPl))
sig_list=[zip(xX,XAS_sigPl[Nth]) for Nth, theta in enumerate(thetaXAS)]
pi_list=[zip(xX,XAS_piPl[Nth]) for Nth,theta in enumerate(thetaXAS)]
fig=plt.figure(300,figsize=(7,5),dpi=200)
plt.suptitle('Li2CuO2 XAS spectra in experimental geometry, theta = 10, 90')
a=plt.subplot(1,2,1)
lines=collections.LineCollection(sig_list, offsets=(0,xmax/2))
a.add_collection(lines, autolim=True)
a.autoscale_view()
plt.title('sigma polarization')
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
b=plt.subplot(1,2,2)
lines=collections.LineCollection(pi_list, offsets=(0,xmax/2))
b.add_collection(lines, autolim=True)
b.autoscale_view()
plt.title('pi polarization') 
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
fig.subplots_adjust(hspace=.5,wspace=.5)
plt.savefig(typ+'.XAS_geom.svg',dpi=200)
plt.savefig(typ+'.XAS_geom.pdf',dpi=200)
plt.savefig(typ+'.XAS_BS.all.svg',dpi=200)
num=0
dmSOC_bname = ".dmSOCgeom."+str(2*dN)+"s."                       # base name for dipole moments files
fileDXsocRe = open(typ+dmSOC_bname+"Xre",'w')                              # dm.0.8s.X_ci
fileDXsocIm = open(typ+dmSOC_bname+"Xim",'w')                              # dm.0.8s.X_ci
fileDYsocRe = open(typ+dmSOC_bname+"Yre",'w')
fileDYsocIm = open(typ+dmSOC_bname+"Yim",'w')
fileDZsocRe = open(typ+dmSOC_bname+"Zre",'w')
fileDZsocIm = open(typ+dmSOC_bname+"Zim",'w')
#for item in D_tot[0] :	#[Vns+CHns,Vns+CHns] :
#    print >> fileDXsoc, item[0], " ".join(map(str, item[1:]))
#for item in D_tot[1] :	#[Vns+CHns,Vns+CHns] :
#    print >> fileDYsoc, item[0], " ".join(map(str, item[1:]))
#for item in D_tot[2] :	#[Vns+CHns,Vns+CHns] :
#    print >> fileDZsoc, item[0], " ".join(map(str, item[1:]))
np.savetxt(fileDXsocRe,D_tot[0].real,fmt='%10.8f',delimiter=' ')
np.savetxt(fileDXsocIm,D_tot[0].imag,fmt='%10.8f',delimiter=' ')
np.savetxt(fileDYsocRe,D_tot[1].real,fmt='%10.8f',delimiter=' ')
np.savetxt(fileDYsocIm,D_tot[1].imag,fmt='%10.8f',delimiter=' ')
np.savetxt(fileDZsocRe,D_tot[2].real,fmt='%10.8f',delimiter=' ')
np.savetxt(fileDZsocIm,D_tot[2].imag,fmt='%10.8f',delimiter=' ')
#
#
####
####### RIXS spectra in experimental geometry
####
#
#
#### read energies from file (CI energies)
#E_v = np.zeros(Vns)
#Efile = 'smbas.esoc.1.10s'      # name for the file with (CI) energies line by line
#for n, line in enumerate(open(Efile).readlines()):
#    E_v[n] = float(line)*27.211396
####
#
thetaRIXS=[10,20,30,40,50,65,80,90,100,110,120,125]
alphaDEG=50
alpha=math.radians(alphaDEG)
#
RIXS_sigsigCoeff=[np.zeros((Vns,n_GS), dtype=np.complex128) for theta in thetaRIXS]
RIXS_sigpiCoeff=[np.zeros((Vns,n_GS), dtype=np.complex128) for theta in thetaRIXS]
RIXS_pisigCoeff=[np.zeros((Vns,n_GS), dtype=np.complex128) for theta in thetaRIXS]
RIXS_pipiCoeff=[np.zeros((Vns,n_GS), dtype=np.complex128) for theta in thetaRIXS]
RIXS_sigsigI=[np.zeros(Vns) for theta in thetaRIXS]
RIXS_sigpiI=[np.zeros(Vns) for theta in thetaRIXS]
RIXS_pisigI=[np.zeros(Vns) for theta in thetaRIXS]
RIXS_pipiI=[np.zeros(Vns) for theta in thetaRIXS]
xR=np.linspace(min(E_v[0]-E_v)-1,0,1000)
RIXS_sigPl=[np.zeros_like(xR) for theta in thetaRIXS]
RIXS_piPl=[np.zeros_like(xR) for theta in thetaRIXS]
# Vns/2 is a trick to sum over doublets
sigPoints=[np.zeros(Vns/2) for theta in thetaRIXS]                  # intensities summed over GS doublets and FS doublets
piPoints=[np.zeros(Vns/2) for theta in thetaRIXS]
#
L3s=[0,1,2,3]
L2s=[4,5]
Gbroad=[0.02,0.02,0.06,0.06,0.06,0.06,0.06,0.06,0.09,0.09]

# for RIXS map 
# print min(E_ch-E_v[0]),max(E_ch-E_v[0]),min(E_v[0]-E_v)
E_L3 = (np.mean(E_ch[0:4]-E_v[0])) #  resonance energy 
E_L2 = (np.mean(E_ch[5:6]-E_v[0]))
incRange=np.linspace(min(E_ch-E_v[0])-5,max(E_ch-E_v[0])+5,0)
incRange=np.sort(np.append(incRange,[E_L3,E_L2])) # add resonance energies to incident range array (and sort)
#
i_L3=(np.abs(incRange-E_L3)).argmin() # index of the resonance energy in the incident range array
i_L2=(np.abs(incRange-E_L2)).argmin()
#
l_peaks=[]
for fst in range(Vns):
    l_peaks.append(E_v[0]-E_v[fst])
lossRange=np.linspace(min(E_v[0]-E_v)-0.25,0.,100)
lossRange=np.sort(np.append(lossRange,l_peaks))
#
inc,loss=np.meshgrid(incRange,lossRange)
RIXS_sig_plane=[np.zeros_like(inc) for theta in thetaRIXS]
# for i,Ei in enumerate(incRange):
#     for l,El in enumerate(lossRange):
#         print RIXS_sig_plane[0][l,i]
# print I
RIXS_pi_plane =[np.zeros_like(inc) for theta in thetaRIXS]
Gis=1.0                         # 1 eV
Gfs=0.1                         # 0.1 eV
#
## 
# print(np.find_nearest( incRange,  E_L3))
# print E_L3
# print((np.abs(incRange-E_L3)).argmin())
# print(incRange[(np.abs(incRange-E_L3)).argmin()])
# print(incRange)
# incRange.append(E_L3)
# incRange.append(E_L2)
# print(incRange)
# sys.exit()
for Nth, theta in enumerate([math.radians(degr) for degr in thetaRIXS]):              # for theta in thetaXAS [radians]
# for Nth, theta in enumerate([math.radians(degr) for degr in thetaRIXS[0:1]]):              # for theta in thetaXAS [radians]
    Dsig=Dsmall[2]                                                                    # sigma in = sigma out = = y'
    Dinpi =Dsmall[1]*math.sin(theta)+Dsmall[0]*math.cos(theta)                        # pi in = x'*sinTh + z'*cosTh
    Doutpi=Dsmall[1]*math.sin(theta+alpha)+Dsmall[0]*math.cos(theta+alpha)            # pi out = pi(Theta+Alpha)
    for i,Ei in enumerate(incRange):
        for l,El in enumerate(lossRange):
            print Nth,i,Ei,l,El
            sig_gs, pi_gs =0.,0.
            for gst in range(n_GS):
                sigsig_fs,sigpi_fs,pisig_fs,pipi_fs=0.,0.,0.,0.
                for fst in range(Vns):
                    sigsig_int, sigpi_int, pisig_int, pipi_int = 0.,0.,0.,0.
                    for ist in range(CHns):
                        denom = E_v[gst]+Ei-E_ch[ist]+complex(0,1)*Gis/2
                        sigsig_int += Dsig[ist,gst]*Dsig[ist,fst].conj()/denom
                        sigpi_int  += Dsig[ist,gst]*Doutpi[ist,fst].conj()/denom
                        pisig_int += Dinpi[ist,gst]*Dsig[ist,fst].conj()/denom
                        pipi_int  += Dinpi[ist,gst]*Doutpi[ist,fst].conj()/denom
                    broad_fs = Gfs/(2*np.pi)/((E_v[gst]-El-E_v[fst])**2 +(Gfs**2)/4)
                    sigsig_fs += sigsig_int*sigsig_int.conj()*broad_fs
                    sigpi_fs  += sigpi_int*sigpi_int.conj()*broad_fs
                    pisig_fs += pisig_int*pisig_int.conj()*broad_fs
                    pipi_fs  += pipi_int*pipi_int.conj()*broad_fs
                sig_gs += (sigsig_fs + sigpi_fs)/n_GS
                pi_gs  += (pisig_fs + pipi_fs)/n_GS
            RIXS_sig_plane[Nth][l,i] = sig_gs*Ei*((El+Ei)**3)*(1/((137.0**4)))*(1/((27.211**2)))*(0.52917721092**2)
            RIXS_pi_plane[Nth][l,i]  = pi_gs*Ei*((El+Ei)**3)*(1/((137.0**4)))*(1/((27.211**2)))*(0.52917721092**2)
            print RIXS_sig_plane[Nth][l,i]-RIXS_pi_plane[Nth][l,i]
            print Ei*((El+Ei)**3)*(1/((137.0**4)))*(1/((27.211**2)))*(0.52917721092**2)
    fig=plt.figure(Nth,figsize=(5,7),dpi=200)
    a=plt.subplot(2,1,1)
    plt.contourf(incRange,lossRange,RIXS_sig_plane[Nth])
    b=plt.subplot(2,1,2)
    plt.contourf(incRange,lossRange,RIXS_pi_plane[Nth])
    plt.savefig(typ+'.'+str(thetaRIXS[Nth])+'.RIXS_plane.pdf')

RIXS_sig_L3=[RIXS_sig_plane[Nth][:,i_L3] for Nth, theta in enumerate(thetaRIXS)]
RIXS_pi_L3 =[RIXS_pi_plane[Nth][:,i_L3] for Nth, theta in enumerate(thetaRIXS)]
ymax=max(np.amax(RIXS_sig_L3),np.amax(RIXS_pi_L3))
sig_list=[zip(lossRange,RIXS_sig_L3[Nth]) for Nth, theta in enumerate(thetaRIXS)]
pi_list=[zip(lossRange,RIXS_pi_L3[Nth]) for Nth,theta in enumerate(thetaRIXS)]
##
fig=plt.figure(100,figsize=(7,5),dpi=200)
#plt.suptitle('Li2CuO2 RIXS spectra in experimental geometry, theta'+str(thetaRIXS))
a=plt.subplot(1,2,1)
lines=collections.LineCollection(sig_list, offsets=(0,ymax/2))
a.add_collection(lines, autolim=True)
a.autoscale_view()
plt.title('sigma polarization')
#a.axes.get_yaxis().set_visible(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
b=plt.subplot(1,2,2)
lines=collections.LineCollection(pi_list, offsets=(0,ymax/2))
b.add_collection(lines, autolim=True)
b.autoscale_view()
plt.title('pi polarization') 
#b.axes.get_yaxis().set_visible(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
fig.subplots_adjust(hspace=.5,wspace=.5)
plt.savefig(typ+'.RIXS_geom.L3.svg',dpi=200)
plt.savefig(typ+'.RIXS_geom.L3.pdf',dpi=200)
##
###
##
fig=plt.figure(110,dpi=300)
for Nth, theta in enumerate(thetaRIXS):
    # sigNth=plt.subplot(len(thetaRIXS),2,2*Nth+1)
    sigNth=plt.subplot(len(thetaRIXS),2,2*len(thetaRIXS)-1-2*Nth)
    plt.plot(lossRange,RIXS_sig_L3[Nth])
    plt.ylim([0,ymax])
    # plt.title('sigma pol')
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymax))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(ymax/2))
#
    # if len(thetaRIXS)-Nth==1:
    if Nth==0:
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
    else:
        sigNth.axes.get_xaxis().set_visible(False)
        sigNth.axes.get_yaxis().set_visible(False)
#
    # piNth=plt.subplot(len(thetaRIXS),2,2*Nth+2)
    piNth=plt.subplot(len(thetaRIXS),2,2*len(thetaRIXS)-2*Nth)
    plt.plot(lossRange,RIXS_pi_L3[Nth])
    plt.ylim([0,ymax])
    # plt.title('sigma pol')
    plt.gca().yaxis.set_major_locator(MultipleLocator(ymax))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(ymax/2))
    # if len(thetaRIXS)-Nth==1:
    if Nth==0:
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
    else:
        piNth.axes.get_xaxis().set_visible(False)
        piNth.axes.get_yaxis().set_visible(False)
#
plt.subplots_adjust(hspace=.2,wspace=.2)
plt.savefig(typ+'.RIXS_boxes.L3.pdf',dpi=300)


#L2
RIXS_sig_L2=[RIXS_sig_plane[Nth][:,i_L2] for Nth, theta in enumerate(thetaRIXS)]
RIXS_pi_L2 =[RIXS_pi_plane[Nth][:,i_L2] for Nth, theta in enumerate(thetaRIXS)]
ymax=max(np.amax(RIXS_sig_L2),np.amax(RIXS_pi_L2))
sig_list=[zip(lossRange,RIXS_sig_L2[Nth]) for Nth, theta in enumerate(thetaRIXS)]
pi_list=[zip(lossRange,RIXS_pi_L2[Nth]) for Nth,theta in enumerate(thetaRIXS)]
figL2=plt.figure(200,figsize=(7,5),dpi=200)
#plt.suptitle('Li2CuO2 RIXS spectra in experimental geometry, theta'+str(thetaRIXS))
a=plt.subplot(1,2,1)
lines=collections.LineCollection(sig_list, offsets=(0,ymax/2))
a.add_collection(lines, autolim=True)
a.autoscale_view()
plt.title('sigma polarization')
#a.axes.get_yaxis().set_visible(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
b=plt.subplot(1,2,2)
lines=collections.LineCollection(pi_list, offsets=(0,ymax/2))
b.add_collection(lines, autolim=True)
b.autoscale_view()
plt.title('pi polarization') 
#b.axes.get_yaxis().set_visible(False)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
figL2.subplots_adjust(hspace=.5,wspace=.5)
plt.savefig(typ+'.RIXS_geom.L2.svg',dpi=200)
plt.savefig(typ+'.RIXS_geom.L2.pdf',dpi=200)


#
# fig=plt.figure(1,figsize=(7,5),dpi=200)
# fig=plt.contourf(incRange,lossRange,RIXS_sig_plane[0])
# plt.savefig(typ+'.RIXS_plane.pdf')

# fig=plt.figure(1,figsize=(5,7),dpi=200)
# a=plt.subplot(2,1,1)
# plt.contourf(incRange,lossRange,RIXS_sig_plane[0])
# b=plt.subplot(2,1,2)
# plt.contourf(incRange,lossRange,RIXS_pi_plane[0])
# plt.savefig(typ+'.RIXS_plane.pdf')

# for Nth, theta in enumerate([math.radians(degr) for degr in thetaRIXS]):              # for theta in thetaXAS [radians]
#     Dsig=Dsmall[2]                                                                    # sigma in = sigma out = = y'
#     Dinpi =Dsmall[1]*math.sin(theta)+Dsmall[0]*math.cos(theta)                        # pi in = x'*sinTh + z'*cosTh
#     Doutpi=Dsmall[1]*math.sin(theta+alpha)+Dsmall[0]*math.cos(theta+alpha)            # pi out = pi(Theta+Alpha)
#     for gst in range(n_GS):
#         for fst in range(Vns):
#             for ist in L3s:     # /1 eV, Gamma_is=1eV
#                 RIXS_sigsigCoeff[Nth][fst,gst]+= Dsig[ist,gst]*Dsig[ist,fst].conj()             # *(E_ch[ist]-E_v[gst])*pow((E_ch[ist]-E_v[fst]),3)/1   #*Ein*Eout^3/Gis
#                 RIXS_pisigCoeff[Nth][fst,gst] += Dinpi[ist,gst]*Dsig[ist,fst].conj()            # *(E_ch[ist]-E_v[gst])*pow((E_ch[ist]-E_v[fst]),3)/1          #pi in sig out
#                 RIXS_sigpiCoeff[Nth][fst,gst] += Dsig[ist,gst]*Doutpi[ist,fst].conj()           # *(E_ch[ist]-E_v[gst])*pow((E_ch[ist]-E_v[fst]),3)/1       #sig in pi out
#                 RIXS_pipiCoeff[Nth][fst,gst]  += Dinpi[ist,gst]*Doutpi[ist,fst].conj()          # *(E_ch[ist]-E_v[gst])*pow((E_ch[ist]-E_v[fst]),3)/1          #pi in pi out
#             RIXS_sigsigI[Nth][fst] += RIXS_sigsigCoeff[Nth][fst,gst]*RIXS_sigsigCoeff[Nth][fst,gst].conj()            #*1.3676*pow(10,-5)   # *const=[1/(eV^2 A^2)]
#             RIXS_pisigI[Nth][fst]  += RIXS_pisigCoeff[Nth][fst,gst]*RIXS_pisigCoeff[Nth][fst,gst].conj()              #*1.3676*pow(10,-5)
#             RIXS_sigpiI[Nth][fst]  += RIXS_sigpiCoeff[Nth][fst,gst]*RIXS_sigpiCoeff[Nth][fst,gst].conj()              #*1.3676*pow(10,-5)
#             RIXS_pipiI[Nth][fst]   += RIXS_pipiCoeff[Nth][fst,gst]*RIXS_pipiCoeff[Nth][fst,gst].conj()                #*1.3676*pow(10,-5)
#         for fst in range(Vns):
#                 RIXS_sigPl[Nth] += cauchy(E_v[0]-E_v[fst],0.1).pdf(xR)*(RIXS_sigsigI[Nth][fst]+RIXS_sigpiI[Nth][fst])/2  # /2 to account the double counting of GS's
#                 RIXS_piPl[Nth]  += cauchy(E_v[0]-E_v[fst],0.1).pdf(xR)*(RIXS_pisigI[Nth][fst]+RIXS_pipiI[Nth][fst])/2
#                 sigPoints[Nth][fst/2] +=  (RIXS_sigsigI[Nth][fst] + RIXS_sigpiI[Nth][fst])/2
#                 piPoints[Nth][fst/2]  +=  (RIXS_pisigI[Nth][fst]  + RIXS_pipiI[Nth][fst] )/2

# ymax=max(np.amax(RIXS_sigPl),np.amax(RIXS_piPl))
# sig_list=[zip(xR,RIXS_sigPl[Nth]) for Nth, theta in enumerate(thetaRIXS)]
# pi_list=[zip(xR,RIXS_piPl[Nth]) for Nth,theta in enumerate(thetaRIXS)]
# fig=plt.figure(1,figsize=(7,5),dpi=200)
# #plt.suptitle('Li2CuO2 RIXS spectra in experimental geometry, theta'+str(thetaRIXS))
# a=plt.subplot(1,2,1)
# lines=collections.LineCollection(sig_list, offsets=(0,ymax/2))
# a.add_collection(lines, autolim=True)
# a.autoscale_view()
# plt.title('sigma polarization')
# #a.axes.get_yaxis().set_visible(False)
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# b=plt.subplot(1,2,2)
# lines=collections.LineCollection(pi_list, offsets=(0,ymax/2))
# b.add_collection(lines, autolim=True)
# b.autoscale_view()
# plt.title('pi polarization') 
# #b.axes.get_yaxis().set_visible(False)
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# fig.subplots_adjust(hspace=.5,wspace=.5)
# plt.savefig(typ+'.RIXS_geom.L3.svg',dpi=200)
# plt.savefig(typ+'.RIXS_geom.L3.pdf',dpi=200)
# #
# # 
# ## figure with points
# #

# Est=[E_v[0]-E_v[st*2] for st in range(Vns/2)]
# sigPoints_list=[zip(Est,sigPoints[Nth]) for Nth, theta in enumerate(thetaRIXS)]
# piPoints_list= [zip(Est,piPoints[Nth]) for Nth, theta in enumerate(thetaRIXS)]
# #
# absfig=plt.figure(100)
# a=plt.subplot(2,1,1)
# a.stem(Est,piPoints[11], linefmt='b-', markerfmt='bo', basefmt='r-')
# plt.title('Double differential crossection, Pi pol')
# #plt.gca().xlim(min(E_v[0]-E_v)-1,0)
# a.set_xlim([min(E_v[0]-E_v)-0.7,0.05])
# b=plt.subplot(2,1,2)
# b.stem(Est,sigPoints[11], linefmt='b-', markerfmt='bo', basefmt='r-')
# plt.title('Double differential crossection, Sigma pol')
# #plt.gca().xlim(min(E_v[0]-E_v)-1,0)
# b.set_xlim([min(E_v[0]-E_v)-0.7,0.05])
# plt.savefig(typ+'.RIXS_abs_L3.pdf',dpi=200)
