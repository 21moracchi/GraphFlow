# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:34:59 2022

@author: Guillaume PIROT
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from sklearn.metrics import jaccard_score

from LoopFlow import topological_analysis as ta   
from LoopFlow import import_triangulation as it
from LoopFlow import calculate_flow as cf
import os
import pandas as pd
import plotly.graph_objects as go
import math
import flopy
from numba import jit




def calc_edges_Kresistance(df_edges,flowDir,Kxyratio,Kxzratio):
    '''formula without any backward flow, penalizing transversal flow'''
    scalar = (df_edges['vx'].values/ df_edges['length'].values*flowDir[0]+
              df_edges['vy'].values/ df_edges['length'].values*flowDir[1]+
              df_edges['vz'].values/ df_edges['length'].values*flowDir[2])
    factor = np.maximum(scalar,np.zeros(len(scalar)))
    resistance = ((1/factor)* (df_edges['vx'].values**2 +
                                   df_edges['vy'].values**2*Kxyratio**2 +
                                   df_edges['vz'].values**2*Kxzratio**2)**(1/2) 
                                  * 1/(df_edges['K'].values)**(1/2))
    return resistance

def calc_edges_Kresistance_rizzo(df_edges,flowDir,Kxyratio,Kxzratio):
    '''doesnt take direction into account, from Rizzo and de Barros (2017)'''
    scalar = (df_edges['vx'].values/ df_edges['length'].values*flowDir[0]+
              df_edges['vy'].values/ df_edges['length'].values*flowDir[1]+
              df_edges['vz'].values/ df_edges['length'].values*flowDir[2])
    factor = 1
    resistance = ((1/factor)* (df_edges['vx'].values**2 +
                                   df_edges['vy'].values**2*Kxyratio**2 +
                                   df_edges['vz'].values**2*Kxzratio**2)**(1/2) 
                                  * 1/(df_edges['K'].values)**(1/2))
    return resistance

def buildKmodel(nd_lithocodes,mgd0,mgd1,mgd2,log10Kmu,log10Ksi,nd_topo_faults,fltlog10factor,realKfilename,verb, is_fault = True):
    #   ***************************************************************************
    #   BUILD REGULAR GRID MODEL
    #   *************************************************************************** 
    log10Kreal = mgd0 * log10Ksi[0] + log10Kmu[0]

    ix1 = np.where(nd_lithocodes==1)
    ix2 = np.where(nd_lithocodes==2)

    log10Kreal[ix1] = mgd1[ix1] * log10Ksi[1] + log10Kmu[1]
    log10Kreal[ix2] = mgd2[ix2] * log10Ksi[2] + log10Kmu[2]
    if is_fault:
        from scipy.ndimage import binary_dilation
        if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - ASSEMBLE GEOL LAYERS")

        
        if verb: print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - ADD FAULT EFFECTS")
        fault_location_dict = {}
        for i in range(3):
            fault_topology = nd_topo_faults[:,:,:,i]
            a_1 = (fault_topology == 1).astype(fault_topology.dtype)
            a_1 = binary_dilation(a_1).astype(a_1.dtype)

            a_2 = (fault_topology == -1).astype(fault_topology.dtype)
            a_2 = binary_dilation(a_2).astype(a_2.dtype) #fault should be only one pixel
            fault_location_dict[i] = a_1+a_2 == 2
            
    
    
    
            # ixf1 = np.where((ixflt1>0)&(ixflt2==0)&(ixflt3==0))
            # ixf2 = np.where((ixflt2>0)&(ixflt3==0))
            # ixf3 = np.where(ixflt3>0)
        log10Kreal[fault_location_dict[0]] += fltlog10factor[0]
        log10Kreal[fault_location_dict[1] &  ~fault_location_dict[0]] += fltlog10factor[1]
        log10Kreal[fault_location_dict[2] & ~fault_location_dict[0]& ~fault_location_dict[1]] += fltlog10factor[2]
    
    #   ***************************************************************************
    #   CONTROL PLOTS
    #   *************************************************************************** 
    if verb:
        sizex,sizey,sizez = nd_lithocodes.shape
        print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - CONTROL PLOTS")
        sech = log10Kreal[:,:,int(sizez/2)].T
        secv = log10Kreal[:,int(sizex/2),:].T
        plt.figure(dpi=300),plt.imshow(sech,origin='lower'),plt.title('horizontal section log10K'),plt.xlabel('x'),plt.ylabel('y'),plt.show()
        plt.figure(dpi=300),plt.imshow(secv,origin='lower'),plt.title('vertical section log10K'),plt.xlabel('x'),plt.ylabel('z'),plt.show()
    
    realK = 10**log10Kreal
    #   ***************************************************************************
    #   SAVE IN PICKLE FILE
    #   *************************************************************************** 
    with open(realKfilename, 'wb') as f:
        pickle.dump([realK], f)
    return

def get_nodeid_pt(df_nodes,ptx,pty,ptz,dx,dy,dz):
    tmpix = np.where((np.abs(df_nodes['X'].values-ptx)<dx/2) & 
                     (np.abs(df_nodes['Y'].values-pty)<dy/2) &
                     (np.abs(df_nodes['Z'].values-ptz)<dz/2))
    nodeid = df_nodes.loc[tmpix,'id'].values
    return nodeid

def get_nodeid_section(df_nodes,axis,value,resax):
    tmpix = np.where( (np.abs(df_nodes[axis].values-value)<resax/2) )
    nodeid = df_nodes.loc[tmpix,'id'].values
    return nodeid

def calc_edges_Qcapacity(df_edges,qxname,qyname,qzname):
    capacity = (df_edges['vx'].values * df_edges[qxname].values +
                df_edges['vy'].values * df_edges[qyname].values +
                df_edges['vz'].values * df_edges[qzname].values ) 
    return np.abs(np.maximum(capacity,np.zeros(capacity.shape)))

def build_igraph(nd_lithocodes, realK,xxx, yyy, zzz,
                 generalFlowDir, Kxyratio, Kxzratio, 
                 porosity, ctm_ptx, ctm_pty, ctm_ptz, destination=None, 
                 unique_edges=True,simplify=True,res_type = 'standard', verb=False, dimensions = (70,50,40)):
    propfield = []
    propfield.append(ta.GridProp('K','harmonic',realK))
    propfield.append(ta.GridProp('porosity','arithmetic',porosity*np.ones(realK.shape)))
    
    #   ***************************************************************************
    #   GENERATE GRAPH AS DATAFRAME
    #   *************************************************************************** 
    # print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - GRAPH GENERATION - "+destination)
    # if not (destination is None): 
    #     if os.path.isdir(destination)==False:
    #         os.mkdir(destination)
    nd_lithocodes = np.zeros(dimensions, dtype = 'int32')
    nd_lithocodes[0,0,0] = 1
    df_nodes,df_edges = ta.reggrid_topology_graph(xxx,yyy,zzz,nd_lithocodes,[],[],propfields=propfield,unique_edges=unique_edges,simplify=simplify,verb=verb)
    

    #   ***************************************************************************
    #   IDENTIFY SOURCE AND TARGET NODES
    #   *************************************************************************** 
    dx = np.diff(np.unique(xxx))[0]
    dy = np.diff(np.unique(yyy))[0]
    dz = np.diff(np.unique(zzz))[0]
    id_src = get_nodeid_pt(df_nodes,ctm_ptx,ctm_pty,ctm_ptz,dx,dy,dz)
    id_tgt = get_nodeid_section(df_nodes,axis='X',value=xxx.max(),resax=dx)

    #   ***************************************************************************
    #   COMPUTE WEIGHTS
    #   *************************************************************************** 
    if res_type == 'standard':
        calc_edge_function = calc_edges_Kresistance
    if res_type == 'rizzo':
        calc_edge_function = calc_edges_Kresistance_rizzo


    df_edges['weightK'] = calc_edge_function(df_edges,generalFlowDir,Kxyratio,Kxzratio)
    df_edges['capacityK'] = 1/df_edges['weightK'].values
    df_nodes['weightK'] =1/df_nodes['K'].values
    

    from igraph import Graph
    
    df_edges['id_node_src'] = df_edges['id_node_src'].astype('int64')
    df_edges['id_node_tgt'] = df_edges['id_node_tgt'].astype('int64')
    g = Graph.DataFrame(df_edges, directed=True, vertices=df_nodes)

    picklefn = destination 
    with open(picklefn, 'wb') as f:
        pickle.dump([g,id_src,id_tgt], f)


    return g,id_src,id_tgt
    


#   ***************************************************************************
#   MODFLOW UTILS
#   *************************************************************************** 

def write_input_files(gwf,modelname):
    headfile = '{}.hds'.format(modelname)
    head_filerecord = [headfile]
    budgetfile = '{}.cbb'.format(modelname)
    budget_filerecord = [budgetfile]
    saverecord, printrecord = [('HEAD', 'ALL'), ('BUDGET', 'ALL')], [('HEAD', 'ALL')]
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf, pname='oc', saverecord=saverecord, head_filerecord=head_filerecord,
                                                budget_filerecord=budget_filerecord, printrecord=printrecord)
    return oc

def get_data(modelname, workspace):
    fpth = os.path.join(workspace, modelname +'.hds')
    hds = flopy.utils.binaryfile.HeadFile(fpth)  
    # times = hds.get_times()
    head = hds.get_data() #(totim=times[-1])    
    fpth = os.path.join(workspace, modelname +'.cbb')
    breakpoint()
    cbc = flopy.utils.binaryfile.CellBudgetFile(fpth)
    flowja = cbc.get_data(text='FLOW-JA-FACE')[0][0][0]
    spd = cbc.get_data(text='DATA-SPDIS')[0]
    chdflow = cbc.get_data(text='CHD')[-1]
    return(head, spd, chdflow, flowja)

def find_kji(cell,nlay,nrow,ncol): #cellid is zerobased
    import math
    cellid = cell - 1
    k = math.floor(cellid/(ncol*nrow)) # Zero based
    j = math.floor((cellid - k*ncol*nrow)/ncol) # Zero based
    i = cellid - k*ncol*nrow - j*ncol
    return(k,j,i) # ZERO BASED!

def find_cellid(k,j,i,nlay,nrow,ncol): # returns zero based cell id
    return(i + j*ncol + k*ncol*nrow)

def x_to_col(x, delr):
    return(int(x/delr))

def y_to_row(y, delc):
    return(int(y/delc))

def z_to_lay(z, delz, zmax):
    return(int((zmax - z)/delz))

def get_q_dis(spd,nlay,nrow,ncol):
    q, qdir =  np.zeros((nlay,nrow,ncol)), np.zeros((nlay,nrow,ncol))
    qx, qy, qz = np.zeros((nlay,nrow,ncol)), np.zeros((nlay,nrow,ncol)), np.zeros((nlay,nrow,ncol))
    for rec in spd:   
        cell = rec[0]
        k,j,i = find_kji(cell,nlay,nrow,ncol)
        q[k,j,i] = np.sqrt(rec[3]**2 + rec[4]**2 + rec[5]**2)
        qx[k,j,i] = rec[3]
        qy[k,j,i] = rec[4]
        qz[k,j,i] = rec[5]
        qdir[k,j,i] = math.degrees(math.atan(rec[5]/rec[3]))
    return(q,qx,qy,qz,qdir)

def ch_flow(chdflow):
    flow_in, flow_out = 0., 0.
    for j in range(len(chdflow)):        
        if chdflow[j][2]>0: flow_out += chdflow[j][2]
        if chdflow[j][2]<0: flow_in  += chdflow[j][2]      
    return((flow_in, flow_out))

def get_q_disu(spd, flowja, gwf, staggered):
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spd, gwf)
    # if cross-connections, recalculate qx taking into account overlap areas
    if staggered:
        gp = d2d.get_gridprops_disu6()
        iac = gp["iac"]
        ja = gp["ja"]
        ihc = gp["ihc"]
        topbycell = gp["top"]
        botbycell = gp["bot"]
        hwva = gp["hwva"]
        iconn = -1
        icell = -1
        for il in iac:
            icell += 1
            qxnumer = 0.
            qxdenom = 0.
            for ilnbr in range(il):
                iconn += 1
                if ihc[iconn] == 2:
                    inbr = ja[iconn]
                    if (inbr == icell):
                        continue
                    dz = min(topbycell[icell], topbycell[inbr]) - max(botbycell[icell], botbycell[inbr])
                    qxincr = flowja[iconn] / (hwva[iconn] * dz)
                    # equal weight given to each face, but could weight by distance instead
                    if (inbr < icell):
                        qxnumer += qxincr
                    else:
                        qxnumer -= qxincr
                    qxdenom += 1.
            qx[icell] = qxnumer / qxdenom

    print(len(spd))
    qmag, qdir = [], []
    for i in range(len(spd)):
        qmag.append(np.sqrt(qx[i]**2 + qy[i]**2 + qz[i]**2))
        qdir.append(math.degrees(math.atan(qz[i]/qx[i])))      
    return(qmag,qx,qy,qz,qdir)


def mf6_prep_and_run(realKfilename,s,vx,vy,vz,ctm_ptx,ctm_pty,ctm_ptz,
                     xmin,xmax,ymin,ymax,zmin,zmax,xxx,yyy,zzz,dx,dy,dz,plot=False
                     , ponctual = False, path = "./mf_files"):
    nlay = len(vz)
    nrow = len(vy)
    ncol = len(vx) 
    delr, delc = dx, dy
    
    top = 1000.
    bot = np.ones((nlay, nrow, ncol))
    for lay in range(nlay):
        bot_elevation = top - (lay+1) * dz
        bot[lay,:,:] *= bot_elevation
    # LOAD K DATA
    with open(realKfilename, 'rb') as f:
        [realK] = pickle.load(f)
    
    # TDIS
    nper = 2
    perlen = [1, 365 * 10] # Days
    nstp = [1, 12 * 10] # number of time steps 
    tsmult = [1.0, 1.0]

    if ponctual :
        nper = 3
        perlen = [1,1, 365 * 10] # Days
        nstp = [1,1, 12 * 10] # number of time steps 
        tsmult = [1.0,1.0, 1.0]

    tdis_rc = []
    for i in range(nper): 
        tdis_rc.append((perlen[i], nstp[i], tsmult[i])) 
    # steady = [True, False]    
    strt = -500. # IC
    head_left, head_right = 100., 0.
    
    chd_rec = [] 
    for k in range(nlay):
        for j in range(nrow):
            chd_rec.append(((k,j,0), head_left))
            chd_rec.append(((k,j,ncol-1), head_right))
    
    rch_rate = 1 / 365 # m/d  RCH
    rch_rec = [] # RCH
    for j in range(nrow):
        for i in range(ncol):
            rch_rec.append(([0,j,i], rch_rate))
     
    # conversion form m/s to m/d and reodering array axis
    k11 = np.moveaxis(realK*60*60*24, [0,2],[2,0])
    xgrid = np.moveaxis(xxx, [0,2],[2,0])
    ygrid = np.moveaxis(yyy, [0,2],[2,0])
    zgrid = np.moveaxis(zzz, [0,2],[2,0])
    
    # TRANSPORT VARIABLES
    cwell = 100.0
    prsity = 0.25  # Porosity
    al = 1.  # Longitudinal dispersivity 
    trpt = 0.1  # Ratio of transverse to longitudinal dispersitivity
    ath1 = al * trpt # transerve dispersivity
    sconc = 0.0 # initial concentration
    nouter, ninner = 100, 300 # SOLVER VARIABLES
    hclose, rclose, relax = 1e-4, 1e-4, 1.0
    scheme = 'upstream' # upstream, central, or TVD
    xt3d = False

    # WELL
    qwell = 50000 # m3/d
    well_x, well_y, well_z = ctm_ptx,ctm_pty,ctm_ptz
    wel_lay, wel_col, wel_row = z_to_lay(well_z, dz, zmax), x_to_col(well_x, dx), y_to_row(well_y, dy)
    spd_wel = {0: [([wel_lay, wel_row, wel_col], 0., 0.)], 
               1: [([wel_lay, wel_row, wel_col], qwell, cwell)]}
    if ponctual:
        spd_wel = {0: [([wel_lay, wel_row, wel_col], 0., 0.)], 
               1: [([wel_lay, wel_row, wel_col], qwell, cwell)],
               2: [([wel_lay, wel_row, wel_col], 0, 0)]}
    sim_name = 'sim_'+str(s)
    # workspace = '../modelfiles/'
    
    workspace = os.path.join(path, str(s))
    os.makedirs(workspace, exist_ok= True)
    exe_name = 'mf6' # '../exe/mf6.exe'
    
    # SIMULATION
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, version='mf6', exe_name=exe_name, sim_ws=workspace)
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, time_units='DAYS', nper=nper, perioddata=tdis_rc)
    ims = flopy.mf6.ModflowIms(sim, print_option='SUMMARY', complexity='complex',outer_dvclose=1.e-8, inner_dvclose=1.e-8)
    
    # GW FLOW MODEL
    gwfname = 'faulted_gwf_'+str(s) 
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(gwf, pname='dis', xorigin = 0., yorigin = 0., nlay=nlay, nrow=nrow, 
                                                   ncol=ncol, delr=delr,delc=delc,top=top, botm=bot)  
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)       
    chd = flopy.mf6.ModflowGwfchd(gwf,maxbound=len(chd_rec), stress_period_data=chd_rec, save_flows=True)
    #rch = flopy.mf6.ModflowGwfrch(gwf,stress_period_data=rch_rec)
    #npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf, xt3doptions=xt3d,k=k11, k22=k11, k33=k11/10, angle1 = angle1, angle2 = angle2,
    #                                               angle3 = angle3, save_flows=True, save_specific_discharge=True), 
    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf, xt3doptions=xt3d,k=k11, k22=k11, k33=k11/10, 
                                                   save_flows=True, save_specific_discharge=True), 
    
    wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, print_input=True, print_flows=True, maxbound = 1, 
                                                   stress_period_data = spd_wel, 
                                                    save_flows=False, auxiliary="CONCENTRATION", pname="WEL-1", 
                                                   filename="{}.wel".format(gwfname))                                                  
    write_input_files(gwf, gwfname)
    
    
    # GW TRANSPORT MODEL
    gwtname = "faulted_gwt_"+str(s) 
    gwt = flopy.mf6.MFModel(sim, model_type="gwt6", modelname=gwtname, model_nam_file="{}.nam".format(gwtname))
    gwt.name_file.save_flows = True
    
    imsgwt = flopy.mf6.ModflowIms(sim, print_option="ALL", outer_dvclose=hclose, outer_maximum=nouter, under_relaxation="NONE",
             inner_maximum=ninner, inner_dvclose=hclose, rcloserecord=rclose, linear_acceleration="BICGSTAB",
             scaling_method="NONE", reordering_method="NONE",relaxation_factor=relax, filename="{}.ims".format(gwtname))
    
    sim.register_ims_package(imsgwt, [gwt.name])
    
    flopy.mf6.ModflowGwtdis(gwt, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=bot) 
    
    flopy.mf6.ModflowGwtic(gwt, strt=sconc, filename="{}.ic".format(gwtname)) # IC
    
    flopy.mf6.ModflowGwtadv(gwt, scheme=scheme, filename="{}.adv".format(gwtname)) # ADV
    
    flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=al, ath1=ath1, filename="{}.dsp".format(gwtname)) # DSP
    
    flopy.mf6.ModflowGwtmst(gwt, porosity=prsity, first_order_decay=False, decay=None, decay_sorbed=None,
                sorption=None, bulk_density=None, distcoef=None,filename="{}.mst".format(gwtname))
    
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]
    
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray) # SOURCE-SINK MIXING
    
    flopy.mf6.ModflowGwtoc(gwt,concentration_filerecord="{}.ucn".format(gwtname),
                concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
                saverecord=[("CONCENTRATION", "ALL")], printrecord=None)
    
    flopy.mf6.ModflowGwfgwt(sim,exgtype="GWF6-GWT6",exgmnamea=gwfname,exgmnameb=gwtname)
    
    
    # RUN SIMULATION
    sim.write_simulation()
    success, buff = sim.run_simulation()
    print('\nSuccess is: ', success)
    
    
    #GET OUTPUT
    # head, spd, chdflow, flowja= get_data(gwfname, workspace)
    # qmag,qx,qy,qz,qdir = get_q_dis(spd,nlay,nrow,ncol)
    ucn = gwt.output.concentration() # GET CONC OUTPUT
    times = ucn.get_times()
    conc = ucn.get_data(totim=times[-1]) # last time step
    conc_time = ucn.get_alldata() # all time steps
    
    
    #EXPORT
    picklemf6outputfn = workspace+"/mf6results_scenario_"+str(s)+".pickle" #./modelfiles/
    if ponctual:
        picklemf6outputfn = workspace+"/p_mf6results_scenario_"+str(s)+".pickle" #./modelfiles/

    with open(picklemf6outputfn, 'wb') as f:
        pickle.dump([times,conc_time], f)
    # with open(picklemf6outputfn, 'rb') as f:
    #     [head,qmag,qx,qy,qz,times,conc_time] = pickle.load(f)
    return


def get_t_car(mass_time,times):
    ###caracteristic time computed by a geometrical slope method
    slope = np.diff(mass_time)/np.diff(times)
    ix = np.argmax(slope)
    slope_max = slope[ix]
    mass_smax = (mass_time[ix]+mass_time[ix+1])/2
    time_smax = (times[ix]+times[ix+1])/2
    t1 = time_smax - mass_smax/slope_max
    return t1

def get_t_first(cumul_mass_time, times):
    ###time when the mass through outlet reaches 1% of the initial mass
    mass_injected_per_timestep = 1.5 * 10e8
    factor = 0.01 
    mass_threshold = mass_injected_per_timestep*factor 
    
    try:
        t_first = np.where(cumul_mass_time>mass_threshold)[0][0]
        # t_first = np.where(max_voxel>pc_value)[0][0]
    except:
        t_first = len(times)-1
    return times[t_first]



from itertools import repeat
def starmap_with_kwargs(pool, fn,  kwargs_iter):
    args_for_starmap = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn,  kwargs):
    return fn( **kwargs)



def kde2D(x, y, bandwidth = 'silverman', xbins=31j, ybins=1000j, **kwargs): 
    from sklearn.neighbors import KernelDensity
    
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)



def compute_otsu(image, nb_classes = 4):
    from skimage.filters import threshold_multiotsu

    bins = threshold_multiotsu(image,nb_classes)
    image_thresholded = np.digitize(image, bins = bins)
    return image_thresholded, bins




def compute_com_distance(vec_1, vec_2):
    com_1 = np.mean(vec_1, axis=0)
    com_2 = np.mean(vec_2, axis=0)
    com_distance = np.linalg.norm(
        com_1 - com_2
    )
    return com_distance

def compute_variance(vec):

    variance = np.var(vec, axis=0)

    variance = np.linalg.norm(variance)
    return variance

def compute_wass_distance(vec_1, vec_2):
    import ot
    X, Y = np.meshgrid(np.linspace(0, 40, 40),
                        np.linspace(0, 50, 50))
    # Compute pariwise distances between points on 2D grid so we know
    # how to score the Wasserstein distance

    M = ot.dist(vec_2, vec_1)  # choose metric
    n = vec_1.shape[0]  # le calculer en amont...
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    wass_distance = np.sqrt(ot.emd2(a, b, M))
    data_SD = 390  # mean of SD of the mf masks, eventually should be computed properly
    NWD = np.exp(-wass_distance/data_SD)

    return wass_distance, NWD

def compute_jaccard_sim(map_1, map_2):
    jaccard_sim = jaccard_score(
        map_1.flatten(), map_2.flatten(), average= 'binary') #d'autres average sont possibles pas bien co;pris
    return jaccard_sim

def compute_mask_from_nb(map,nb_pixels, dy, dz):
    selected_indices = np.argsort(
                map, axis=None)[:nb_pixels]
            
    Y, Z = np.unravel_index(selected_indices, map.shape)
    vec =  np.concatenate(
        [Y[..., np.newaxis]*dy, Z[..., np.newaxis]*dz], axis=1)
    mask = np.zeros(map.shape)
    for y, z in zip(Y, Z):
        mask[y, z] = 1
    return mask, vec

def compute_max_distance(array:np.ndarray):
    max_distance = 0
    
    for element in array:
        for element_2 in array:
            distance = np.linalg.norm(element-element_2)
            if distance>max_distance:
                max_distance =distance
    return max_distance
            
        
