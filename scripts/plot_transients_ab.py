# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:48:06 2023

@author: BENVI
"""

import numpy as np
import matplotlib.pyplot as plt

y = lambda a,x,b: a*x + b

pts = [(-6,1),(0,0),(4,0.5),(3,-1)]

lines = [(pts[0],pts[1]),(pts[1],pts[2]),(pts[1],pts[3])]

h = np.array([1,0.5,0.3,0.2])*0.75

fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=200)

# ax.plot([-8,-7.25,-6.75],[1,1,1.2],color='k')
# ax.plot([-6.6,-6],[1,1],color='k')
ax.plot([pts[0][0],pts[0][0]],[pts[0][1],pts[-1][1]],color='grey',ls='--')
ax.plot([pts[1][0],pts[1][0]],[pts[1][1],pts[-1][1]],color='grey',ls='--')

xl = [(-4,-2),(1,3),(0.75,2.25),(-2,-4)]

for i,l in enumerate(lines):
    x1,x2 = l[0][0],l[1][0]
    y1,y2= l[0][1],l[1][1]
    dx = x2-x1
    a = (y2-y1)/(x2-x1)
    ax.plot([x1,x2],[y1,y2],color='k')
    ax.fill_between([x1,x2],[y1,y2],[y1+h[i],y2+h[i]],alpha=0.5)    

    ax.annotate('', xy=(xl[i][0],y(a,xl[i][0],0)+h[i]/2),
                 xycoords='data',
                 xytext=(xl[i][1], y(a,xl[i][1],0)+h[i]/2),
                 textcoords='data',
                 # arrowprops=dict(arrowstyle= '-[, widthB=1.0, lengthB=0.2, angleB=None')
                 arrowprops=dict(arrowstyle='<-', color='k')
                 )

for i,l in enumerate(lines[:1]):
    x1,x2 = l[0][0],l[1][0]
    y1,y2= l[0][1],l[1][1]
    a = (y2-y1)/(x2-x1)
    ax.fill_between([x1,x2],[y1+h[i],y2+h[i]],[y1+h[i]+h[-1],y2+h[i]+h[-1]],alpha=0.5)    
    
    ax.annotate('', xy=(xl[i][0],y(a,xl[i][0],0)+h[i]+h[-1]/2),
                 xycoords='data',
                 xytext=(xl[i][1], y(a,xl[i][1],0)+h[i]+h[-1]/2),
                 textcoords='data',
                 color='grey',
                 # arrowprops=dict(arrowstyle= '-[, widthB=1.0, lengthB=0.2, angleB=None')
                 arrowprops=dict(arrowstyle='->', color='k')
                 )

for i,p in enumerate(pts):
    ax.scatter(p[0],p[1],color='k',zorder=7)
    if i == 0:
        ps = (p[0]-0.4,p[1]-0.35)
    elif i==1:
        ps = (p[0]-0.4,p[1]-0.25)            
    else:
        ps = (p[0],p[1]-0.35)    
    ax.text(*ps,'$a_' + str(i+1) + '$\n$b_' + str(i+1) + '$' ,ha = 'center',va = 'center',fontsize=12)

ax.text(-3,-1.05,'$\\tau_{1,2}$',ha = 'center',va = 'center',fontsize=12)

ax.annotate('', xy=(-6.0725,-.85),
             xycoords='data',
             xytext=(0.0725, -0.85),
             textcoords='data',
             # arrowprops=dict(arrowstyle= '-[, widthB=1.0, lengthB=0.2, angleB=None')
             arrowprops=dict(arrowstyle='|-|', color='black')
             )

## LIGHTNING
ax.plot([-6,-6.25,-5.75],[1.5,1.25,1.25],color='red')
ax.annotate('', xy=(-5.7,1.275),
             xycoords='data',
             xytext=(-6.05,0.95),
             textcoords='data',
             color='grey',
             zorder=7,
             arrowprops=dict(arrowstyle='<-', color='red')
             )



ax.set_ylim(-2,2)
ax.set_xlim(-7.2,4.5)

# https://stackabuse.com/matplotlib-turn-off-axis-spines-ticklabels-axislabels-grid/
ax.spines[:].set_visible(False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)




