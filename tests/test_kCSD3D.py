'''
This script is used to test Current Source Density Estimates, 
using the kCSD method Jan et.al (2012) for the 3D case

This script is in alpha phase.

This was written by :
Chaitanya Chintaluri, 
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
'''
import os
import time
import sys
sys.path.append('..')

import numpy as np
from scipy.integrate import simps 
from numpy import exp, linspace
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib.mlab import griddata
from scipy.spatial import distance

try:
    from joblib import Parallel, delayed  
    import multiprocessing
    num_cores = multiprocessing.cpu_count()-1
    parallel_available = True
except ImportError:
    parallel_available = False
import csd_profile as CSD
from KCSD3D import KCSD3D

def generate_csd_3D(csd_profile, csd_seed, 
                    start_x=0., end_x=1., 
                    start_y=0., end_y=1., 
                    start_z=0., end_z=1., 
                    res_x=50, res_y=50, 
                    res_z=50):
    """
    Gives CSD profile at the requested spatial location, at 'res' resolution
    """
    csd_x, csd_y, csd_z = np.mgrid[start_x:end_x:np.complex(0,res_x), 
                                   start_y:end_y:np.complex(0,res_y), 
                                   start_z:end_z:np.complex(0,res_z)]
    f = csd_profile(csd_x, csd_y, csd_z, seed=csd_seed) 
    return csd_x, csd_y, csd_z, f

def grid(x, y, z, resX=100, resY=100):
    """
    Convert 3 column data to matplotlib grid
    """
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    zi = griddata(x, y, z, xi, yi, interp='linear')
    return xi, yi, zi

def generate_electrodes(xlims=[0.1,0.9], ylims=[0.1,0.9], zlims=[0.1,0.9], res=5):
    """
    Places electrodes in a square grid
    """
    ele_x, ele_y, ele_z = np.mgrid[xlims[0]:xlims[1]:np.complex(0,res), 
                                   ylims[0]:ylims[1]:np.complex(0,res), 
                                   zlims[0]:zlims[1]:np.complex(0,res)]
    ele_x = ele_x.flatten()
    ele_y = ele_y.flatten()
    ele_z = ele_z.flatten()
    return ele_x, ele_y, ele_z

def make_plots(fig_title, 
               t_csd_x, t_csd_y, t_csd_z, true_csd, 
               ele_x, ele_y, ele_z, pots,
               k_csd_x, k_csd_y, k_csd_z, est_csd):
    """
    Shows 3 plots
    1_ true CSD generated based on the random seed given
    2_ interpolated LFT (NOT kCSD pot though), generated by simpsons rule integration
    3_ results from the kCSD 2D for the default values
    """
    fig = plt.figure(figsize=(10,16))
    #True CSD
    z_steps = 5
    height_ratios = [1 for i in range(z_steps)]
    height_ratios.append(0.1)
    gs = gridspec.GridSpec(z_steps+1, 3, height_ratios=height_ratios)
    t_max = np.max(np.abs(true_csd))
    levels = np.linspace(-1*t_max, t_max, 16)
    ind_interest = np.mgrid[0:t_csd_z.shape[2]:np.complex(0,z_steps+2)]
    ind_interest = np.array(ind_interest, dtype=np.int)[1:-1]
    for ii, idx in enumerate(ind_interest):
        ax = plt.subplot(gs[ii, 0])
        im = plt.contourf(t_csd_x[:,:,idx], t_csd_y[:,:,idx], true_csd[:,:,idx], 
                          levels=levels, cmap=cm.bwr_r)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        title = str(t_csd_z[:,:,idx][0][0])[:4]
        ax.set_title(label=title, fontdict={'x':0.8, 'y':0.8})
        ax.set_aspect('equal')
    cax = plt.subplot(gs[z_steps,0])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks(levels[::2])
    cbar.set_ticklabels(np.around(levels[::2], decimals=2))
    #Potentials
    v_max = np.max(np.abs(pots))
    levels_pot = np.linspace(-1*v_max, v_max, 16)
    ele_res = int(np.ceil(len(pots)**(3**-1)))    
    ele_x = ele_x.reshape(ele_res, ele_res, ele_res)
    ele_y = ele_y.reshape(ele_res, ele_res, ele_res)
    ele_z = ele_z.reshape(ele_res, ele_res, ele_res)
    pots = pots.reshape(ele_res, ele_res, ele_res)
    for idx in range(min(5,ele_res)):
        X,Y,Z = grid(ele_x[:,:,idx], ele_y[:,:,idx], pots[:,:,idx])
        ax = plt.subplot(gs[idx, 1])
        im = plt.contourf(X, Y, Z, levels=levels_pot, cmap=cm.PRGn)
        ax.hold(True)
        plt.scatter(ele_x[:,:,idx], ele_y[:,:,idx], 5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        title = str(ele_z[:,:,idx][0][0])[:4]
        ax.set_title(label=title, fontdict={'x':0.8, 'y':0.8})
        ax.set_aspect('equal')
        ax.set_xlim([0.,1.])
        ax.set_ylim([0.,1.])
    cax = plt.subplot(gs[z_steps,1])
    cbar2 = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar2.set_ticks(levels_pot[::2])
    cbar2.set_ticklabels(np.around(levels_pot[::2], decimals=2))
    # #KCSD
    t_max = np.max(np.abs(est_csd[:,:,:,0]))
    levels_kcsd = np.linspace(-1*t_max, t_max, 16)
    ind_interest = np.mgrid[0:k_csd_z.shape[2]:np.complex(0,z_steps+2)]
    ind_interest = np.array(ind_interest, dtype=np.int)[1:-1]
    for ii, idx in enumerate(ind_interest):
        ax = plt.subplot(gs[ii, 2])
        im = plt.contourf(k_csd_x[:,:,idx], k_csd_y[:,:,idx], est_csd[:,:,idx,0], 
                          levels=levels_kcsd, cmap=cm.bwr_r)
        #im = plt.contourf(k_csd_x[:,:,idx], k_csd_y[:,:,idx], est_csd[:,:,idx,0], 
        #                  levels=levels, cmap=cm.bwr_r)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        title = str(k_csd_z[:,:,idx][0][0])[:4]
        ax.set_title(label=title, fontdict={'x':0.8, 'y':0.8})
        ax.set_aspect('equal')
    cax = plt.subplot(gs[z_steps,2])
    cbar3 = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar3.set_ticks(levels_kcsd[::2])
    #cbar3.set_ticks(levels[::2])
    cbar3.set_ticklabels(np.around(levels_kcsd[::2], decimals=2))
    #cbar3.set_ticklabels(np.around(levels[::2], decimals=2))
    fig.suptitle("Lambda,R,CV_Error,RMS_Error,Time = "+fig_title)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])  
    # #Showing
    #plt.tight_layout()
    plt.show()
    return

def integrate_3D(x, y, z, xlim, ylim, zlim, csd, xlin, ylin, zlin, X, Y, Z):
    """
    X,Y - parts of meshgrid - Mihav's implementation
    """
    Nz = zlin.shape[0]
    Ny = ylin.shape[0]
    m = np.sqrt((x - X)**2 + (y - Y)**2 + (z - Z)**2)
    m[m < 0.0000001] = 0.0000001
    z = csd / m
    Iy = np.zeros(Ny)
    for j in xrange(Ny):
        Iz = np.zeros(Nz)                        
        for i in xrange(Nz):
            Iz[i] = simps(z[:,j,i], zlin)
        Iy[j] = simps(Iz, ylin)
    F = simps(Iy, xlin)
    return F 

def calculate_potential_3D(true_csd, ele_xx, ele_yy, ele_zz, 
                           csd_x, csd_y, csd_z):
    """
    For Mihav's implementation to compute the LFP generated
    """
    xlin = csd_x[:,0,0]
    ylin = csd_y[0,:,0]
    zlin = csd_z[0,0,:]
    xlims = [xlin[0], xlin[-1]]
    ylims = [ylin[0], ylin[-1]]
    zlims = [zlin[0], zlin[-1]]
    sigma = 1.0
    pots = np.zeros(len(ele_xx))
    tic = time.time()
    for ii in range(len(ele_xx)):
        pots[ii] = integrate_3D(ele_xx[ii], ele_yy[ii], ele_zz[ii],
                                xlims, ylims, zlims, true_csd, 
                                xlin, ylin, zlin, 
                                csd_x, csd_y, csd_z)
        print 'Electrode:', ii
    pots /= 4*np.pi*sigma
    toc = time.time() - tic
    print toc, 'Total time taken - series, sims'
    return pots

def calculate_potential_3D_parallel(true_csd, ele_xx, ele_yy, ele_zz, 
                                    csd_x, csd_y, csd_z):
    """
    For Mihav's implementation to compute the LFP generated
    """

    xlin = csd_x[:,0,0]
    ylin = csd_y[0,:,0]
    zlin = csd_z[0,0,:]
    xlims = [xlin[0], xlin[-1]]
    ylims = [ylin[0], ylin[-1]]
    zlims = [zlin[0], zlin[-1]]
    sigma = 1.0
    #tic = time.time()
    pots = Parallel(n_jobs=num_cores)(delayed(integrate_3D)(ele_xx[ii],ele_yy[ii],ele_zz[ii],
                                                            xlims, ylims, zlims, true_csd,
                                                            xlin, ylin, zlin,
                                                            csd_x, csd_y, csd_z) for ii in range(len(ele_xx)))
    pots = np.array(pots)
    pots /= 4*np.pi*sigma
    #toc = time.time() - tic
    #print toc, 'Total time taken - parallel, sims '
    return pots
1
def electrode_config(ele_lims, ele_res, true_csd, csd_x, csd_y, csd_z):
    """
    What is the configuration of electrode positions, between what and what positions
    """
    #Potentials
    ele_x_lims = ele_y_lims = ele_z_lims = ele_lims
    ele_x, ele_y, ele_z = generate_electrodes(ele_x_lims, ele_y_lims, ele_z_lims, ele_res)
    if parallel_available:
        pots = calculate_potential_3D_parallel(true_csd, 
                                               ele_x, ele_y, ele_z, 
                                               csd_x, csd_y, csd_z)
    else:
        pots = calculate_potential_3D(true_csd, 
                                      ele_x, ele_y, ele_z, 
                                      csd_x, csd_y, csd_z)        
    ele_pos = np.vstack((ele_x, ele_y, ele_z)).T     #Electrode configs
    num_ele = ele_pos.shape[0]
    print 'Number of electrodes:', num_ele
    return ele_pos, pots

def do_kcsd(ele_pos, pots, **params):
    """
    Function that calls the KCSD3D module
    """
    num_ele = len(ele_pos)
    pots = pots.reshape(num_ele, 1)
    k = KCSD3D(ele_pos, pots, **params)
    #k.cross_validate(Rs=np.arange(0.2,0.4,0.02))
    #k.cross_validate(Rs=np.arange(0.27,0.36,0.01))
    k.cross_validate(Rs=np.array(0.31).reshape(1))
    est_csd = k.values('CSD')
    return k, est_csd

def main_loop(csd_profile, csd_seed, total_ele, num_init_srcs=1000):
    """
    Loop that decides the random number seed for the CSD profile, 
    electrode configurations and etc.
    """
    csd_name = csd_profile.func_name
    print 'Using sources %s - Seed: %d ' % (csd_name, csd_seed)

    #TrueCSD
    t_csd_x, t_csd_y, t_csd_z, true_csd = generate_csd_3D(csd_profile, csd_seed,
                                                          start_x=0., end_x=1., 
                                                          start_y=0., end_y=1., 
                                                          start_z=0., end_z=1.,
                                                          res_x=100, res_y=100,
                                                          res_z=100)

    #Electrodes
    ele_lims = [0.15, 0.85] #square grid, xy min,max limits
    ele_res = int(np.ceil(total_ele**(3**-1))) #resolution of electrode grid
    ele_pos, pots = electrode_config(ele_lims, ele_res, true_csd, t_csd_x, t_csd_y, t_csd_z)
    ele_x = ele_pos[:, 0]
    ele_y = ele_pos[:, 1]
    ele_z = ele_pos[:, 2]
    
    #kCSD estimation
    gdX = 0.05
    gdY = 0.05
    gdZ = 0.05
    x_lims = [.0,1.] #CSD estimation place
    y_lims = [.0,1.]
    z_lims = [.0,1.]
    params = {'h':50., 
              'gdX': gdX, 'gdY': gdY, 'gdZ': gdZ,
              'xmin': x_lims[0], 'xmax': x_lims[1], 
              'ymin': y_lims[0], 'ymax': y_lims[1],
              'zmin': y_lims[0], 'zmax': y_lims[1],
              'ext': 0.0, 'n_srcs_init': num_init_srcs}
    tic = time.time() #time it
    k, est_csd = do_kcsd(ele_pos, pots, h=50., 
                         gdx=gdX, gdy= gdY, gdz=gdZ,
                         xmin=x_lims[0], xmax=x_lims[1], 
                         ymin=y_lims[0], ymax=y_lims[1],
                         zmin=z_lims[0], zmax=z_lims[1],
                         n_src_init=num_init_srcs)
    toc = time.time() - tic

    #RMS of estimation - gives estimate of how good the reconstruction was
    chr_x, chr_y, chr_z, test_csd = generate_csd_3D(csd_profile, csd_seed,
                                                    start_x=x_lims[0], end_x=x_lims[1],
                                                    start_y=y_lims[0], end_y=y_lims[1],
                                                    start_z=z_lims[0], end_z=z_lims[1],
                                                    res_x=int((x_lims[1]-x_lims[0])/gdX), 
                                                    res_y=int((y_lims[1]-y_lims[0])/gdY),
                                                    res_z=int((z_lims[1]-z_lims[0])/gdZ))
    rms = np.sqrt(abs(np.mean(np.square(test_csd)-np.square(est_csd[:,:,:,0]))))
    rms /= np.sqrt(np.mean(np.square(test_csd))) #Normalizing

    #Plots
    title = str(k.lambd)+','+str(k.R)+', '+str(k.cv_error)+', '+str(rms)+', '+str(toc)
    save_as = csd_name+'_'+str(csd_seed)+'of'+str(total_ele)
    #save_as = csd_name+'_'+str(num_init_srcs)+'_'+str(total_ele)
    make_plots(title, 
               chr_x, chr_y, chr_z, test_csd,
               ele_x, ele_y, ele_z, pots,
               k.estm_x, k.estm_y, k.estm_z, est_csd) 
    #save
    result_kcsd = [k.lambd, k.R, k.cv_error, rms, toc]
    return est_csd, result_kcsd 

if __name__=='__main__':
    total_ele = 125
    #Normal run
    csd_seed = 54 #0-49 are small sources, 50-99 are large sources 
    csd_profile = CSD.gauss_3d_small
    main_loop(csd_profile, csd_seed, total_ele, 1000)
    #fig.savefig(os.path.join(plots_folder, save_as+'.png'))

    #Loop!
    #for ii in range(100):
    #    main_loop(ii, total_ele)

    # import h5py 
    # for n_srcs in [18,20]: #range(12,18,2):
    #     num_init_srcs = n_srcs**3
    #     for ele in range(3,8):
    #         total_ele = ele**3
    #         h = h5py.File('Error_map_3D.h5', 'a')
    #         print 'Starting KCSD for src, ele:', num_init_srcs, total_ele
    #         est_csd, result = main_loop(csd_seed, total_ele, num_init_srcs)
    #         h[str(num_init_srcs)+'/'+str(total_ele)+'/est_csd'] = est_csd
    #         h[str(num_init_srcs)+'/'+str(total_ele)+'/kcsd_para'] = result
    #         print 'Finished KCSD for src, ele:', num_init_srcs, total_ele
    #         h.close()




        