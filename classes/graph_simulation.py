import sys
sys.path.append("..")
import numpy as np
import fnmatch
import os
import plotly.io as pio
import plotly.graph_objects as go
from utils.utils import *
from scipy import stats
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_multiotsu
import matplotlib.colors as colors
from classes.parameters import default_parameters_dict
from classes.scenario import Scenario
import gstools as gs
class GraphSim:
    def __init__(self, verb=True, data_path = '', parameters_dict = None):
        import sys
        sys.path.append('.')

        self.scenarios = dict()
        self.verb = verb
        self.data_sim = False #are data of simulation generated
        ### Folders
        
        self.graph_fd = os.path.join(data_path,'graph_outputs')
        os.makedirs(self.graph_fd, exist_ok= True)
        self.MF_fd = os.path.join(data_path,'MF_files')
        os.makedirs(self.MF_fd, exist_ok= True)
        self.real_K_folder = os.path.join(data_path, 'real_K')
        os.makedirs(self.real_K_folder, exist_ok= True)
        self.img_folder = os.path.join(data_path, 'img')
        os.makedirs(self.img_folder, exist_ok= True)
        self.reggrid_path = os.path.join(data_path, 'reggrid.pickle')

        self.mgs_folder = os.path.join(data_path,'mgs')
        os.makedirs(self.mgs_folder, exist_ok= True)
        self.results_folder = os.path.join(data_path,'results')
        os.makedirs(self.results_folder, exist_ok= True)
        # ****************************************************************************
        # HYDROGEOLOGICAL PARAMETERS
        # ****************************************************************************
        self.parameters_dict = default_parameters_dict
        if parameters_dict is not None:
            for key,value in parameters_dict.items():
                self.parameters_dict[key] = value

    def generate_sim_data(self, **kwargs):
        pass
        
    def instantiate_scenario(self, scenario_id, res_type='standard', sim_time='first', **kwargs):
        self.generate_sim_data(**kwargs)
        scenario = Scenario(scenario_id)
        self.scenarios[scenario_id] = scenario
        scenario.fault_scenario = self.parameters_dict['fault_array'][scenario.id//10]
        scenario.real_K_path =os.path.join(self.real_K_folder, f'real_K_{scenario.id}.pickle')
        scenario.ctm_ptx, scenario.ctm_pty, scenario.ctm_ptz = self.parameters_dict['location_array'][
            scenario.id%10]
        scenario.res_type = res_type
        scenario.graph_filename = f'scenario_{scenario.id}_{scenario.res_type}.pickle'
        
        scenario.graph_path = os.path.join(self.graph_fd, scenario.graph_filename)
        scenario.MF_path = os.path.join(self.MF_fd,f'{scenario.id}', f'mf6results_scenario_{scenario.id}.pickle')
        scenario.time_mode = sim_time
        scenario.fltlog10factor_vec = scenario.fault_scenario.copy()
        scenario.fltlog10factor_vec = scenario.fltlog10factor_vec * self.parameters_dict['fltlog10factor']

        return scenario    
            
    def generate_MGS(self, scale_factor = 1):
        """Generate multi-gaussian simulations """
        self.generate_sim_data()
        
        myseed = 0
        reference_scale = np.array([8*self.dx, 4*self.dy, 2*self.dz])
        real_scale = scale_factor*reference_scale
        model = gs.Gaussian(dim=3, len_scale= real_scale, angles=(0.0, 0.0, 0.0),nugget=0.12,anis=[3,10])
        
        for i in range(self.nmgs):
            if self.verb:
                print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+" - MGS REAL"+str(i))
            mgs = gs.SRF(model, mean=0, seed=myseed+i)
            mgs = gs.SRF(model, mean=0, seed=myseed+i)
            mgs = gs.SRF(model, mean=0, seed=myseed+i)
            mgs((self.vx, self.vy, self.vz), mesh_type='structured')
            mgd = mgs[0]
            mgrealfilepath = os.path.join(self.mgs_folder, f'mgreal{str(i)}.pickle')
            
            with open(mgrealfilepath, 'wb') as f:
                pickle.dump([mgd], f)

    def generate_K_field(self, scenario:Scenario, mgs_id = None):
        """Generates conductivity field from 3 multi gaussian realisations (3 geological units)."""
        scenario_id = scenario.id
        if mgs_id is None :
            mgs_id = self.mgs_combinations[scenario_id, :]
        dfn0 = os.path.join(self.mgs_folder,f'mgreal{str(mgs_id[0])}.pickle')
        with open(dfn0, 'rb') as f:
            [mgd0] = pickle.load(f)
        dfn1 = os.path.join(self.mgs_folder,f'mgreal{str(mgs_id[1])}.pickle')
        with open(dfn1, 'rb') as f:
            [mgd1] = pickle.load(f)
        
        dfn2 = os.path.join(self.mgs_folder,f'mgreal{str(mgs_id[2])}.pickle')
        with open(dfn2, 'rb') as f:
            [mgd2] = pickle.load(f)
        
        buildKmodel(self.nd_lithocodes, mgd0, mgd1, mgd2, self.parameters_dict['log10Kmu'],self.parameters_dict['log10Ksi'], self.nd_topo_faults,
                    scenario.fltlog10factor_vec, scenario.real_K_path, verb=False, is_fault = self.is_fault)
        
    def run_MODFLOW(self, scenario):
        """Runs modflow simulation."""
        
        print(f'{datetime.now()}'
              " - MF6 RUN SCENARIO "+str(scenario.id))
  
        mf6_prep_and_run(scenario.real_K_path, scenario.id, self.vx, self.vy, self.vz, scenario.ctm_ptx, scenario.ctm_pty, scenario.ctm_ptz,
                         self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax, self.xxx, self.yyy, self.zzz, self.dx, self.dy, self.dz, plot=False,
                         path = self.MF_fd)
        try:
            path = os.path.join(self.MF_fd,str(scenario.id))
            flist = fnmatch.filter(os.listdir(path
                ), 'faulted_gw*')
            for f in range(len(flist)):
                os.remove(os.path.join(path,flist[f]))
        except:
            pass

    def load_hydro_data(self, scenario):
        """Loading hydraulic conductivity data and modflow data.

        Args:
            scenario (_type_): _description_
        """
        s = scenario.id
        # LOAD K DATA
        if self.verb:
            print(f'{datetime.now()} - LOADING K FIELD')
        with open(scenario.real_K_path, 'rb') as f:
            [scenario.realK] = pickle.load(f)
        f.close()
        # LOAD MODFLOW RESULTS
        if self.verb:
            print(f'{datetime.now()} - LOADING MODFLOW RESULTS')
       
        
        with open(scenario.MF_path, 'rb') as f:
            [scenario.times, scenario.conc_time] = pickle.load(f)

    def build_graph(self, scenario):
        """Generates graph using iGraph"""
        if self.verb:
            print(f'{datetime.now()} BUILDING GRAPH')
        G, id_src, id_tgt = build_igraph(
            self.nd_lithocodes, scenario.realK, self.xxx, self.yyy, self.zzz,
            self.parameters_dict['generalFlowDir'], self.parameters_dict['Kxyratio'], self.parameters_dict['Kxzratio'],
            self.parameters_dict['porosity'], scenario.ctm_ptx, scenario.ctm_pty, scenario.ctm_ptz, destination=scenario.graph_path,
            unique_edges=True, simplify=True, verb=False, res_type=scenario.res_type, dimensions = self.xxx.shape)
        if self.verb:
            print(f'{datetime.now()} GRAPH BUILT')
        if len(id_src) >1:
            print('WARNING : multiple-source found') #ignore
        
        scenario.id_src = id_src[0]
        scenario.id_tgt = id_tgt
        scenario.graph = G

    def load_graph(self, scenario):
        if scenario.graph is None:


            if self.verb:
                print(f'{(datetime.now())} - LOADING GRAPH')

            with open(scenario.graph_path, 'rb') as f:
                [G, id_src, id_tgt] = pickle.load(f)

            if len(id_src) >1:
                print('WARNING : multiple-source found')
        
            scenario.id_src = id_src[0]
            scenario.id_tgt = id_tgt
            scenario.graph = G



    def get_mass_data(self, scenario):
        """Computes several mass arrays, from the modflow concentration array.
        scenario.mass_time_1D = mass of contaminant flowing through the last layer as a function of time
        scenario.cumul_mass_time = cumulative array of scenario.mass_time_1D
        """
        volume = self.dx * self.dy * self.dz * self.parameters_dict['porosity'] #(m^3)
        scenario.mass_time_1D = np.sum(
            scenario.conc_time[:, :, :, -1], axis=(1, 2)) * volume
        if len(scenario.mass_time_1D.tolist()) <2: #for non convergent cases
            return False
        scenario.cumul_mass_time = np.array([np.sum(scenario.mass_time_1D[:k+1]) for k in range(len(scenario.mass_time_1D))])
        
        ### COMPUTING CARACTERISTIC TIMES
        scenario.t_first = get_t_first(scenario.cumul_mass_time, 
                                       scenario.times)
        scenario.t_car = get_t_car(scenario.cumul_mass_time, scenario.times)
        scenario.t_last = scenario.times[-1]

        if scenario.time_mode == 'first':
            scenario.t_sim = scenario.t_first
        if scenario.time_mode == 'car':
            scenario.t_sim = scenario.t_car
        if scenario.time_mode == 'last':
            scenario.t_sim = scenario.t_last


        time_index = np.argmin(np.abs(scenario.times-scenario.t_sim))
        scenario.mf_map = np.sum(
            scenario.conc_time[:time_index, :, :, -1]*self.parameters_dict['porosity'], axis=0)
        if np.max(scenario.mf_map) < 0.01:
            #if the mass is unsignificant
            return False
        return True

    def plot_plume_3D(self, scenario, t_sim=None, visu = 'browser'):
        """Plots with plotly the plume in 3D

        Args:
            scenario (_type_): _description_
            t_sim (_type_, optional): _description_. Defaults to None.
        """
        pio.renderers.default = visu
        isomin = 1E0
        isomax = 1E2
        self.load_hydro_data(scenario)
        self.get_mass_data(scenario)
        if t_sim is None:  # default
            t_sim = scenario.t_sim
        t_sim_idx = np.argmin(np.abs(scenario.times-t_sim))  # correction
        
        fig = go.Figure(data=[])
        values = np.moveaxis( scenario.conc_time[t_sim_idx, :, :, :],[0,2],[2,0])

        fig = go.Figure(data=go.Volume(
            x=self.xxx.flatten(),
            y=self.yyy.flatten(),
            z=self.zzz.flatten(),
            value=values.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=0.1,
    
            surface_count=21,
            colorscale='RdBu'
            ))
        title = f'Scenario : {scenario.id}, concentration at FTA' 
        fig.update_layout(scene_aspectmode='manual',
                        scene_aspectratio=dict(x=7, y=5, z=1), title_text = title)
        fig.show()
        print(f'{datetime.now()} -PLOTTING 3D')


    def main(self, scenario, build_graph=False, plot_plume_3D=False,
             compute_data_dict=False, compute_auto_thresh=False, plot=False, compute_similarity = False, plot_graph_3D = False,
             generate_K_field = False, run_MOFLOW = False,
             **kwargs):
        """Main function of the class. Used to carries out different methods/computations in a row. Some of them are not independent.

        Args:
            scenario (_type_): _description_
            build_graph (bool, optional): _description_. Defaults to False.
            plot_plume_3D (bool, optional): _description_. Defaults to False.
            compute_data_dict (bool, optional): _description_. Defaults to False.
            compute_auto_thresh (bool, optional): _description_. Defaults to False.
            plot (bool, optional): _description_. Defaults to False.
            compute_similarity (bool, optional): _description_. Defaults to False.
            plot_graph_3D (bool, optional): _description_. Defaults to False.
            generate_K_field (bool, optional): _description_. Defaults to False.
            run_MOFLOW (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if generate_K_field:
            self.generate_K_field(scenario)
        if run_MOFLOW:
            self.run_MODFLOW(scenario)
        self.load_hydro_data(scenario)
        validity = self.get_mass_data(scenario)
        if not validity:
            return {'id': scenario.id, 'success': False}
        if build_graph:
            print(f'{datetime.now()} - BUILDING GRAPH')
            self.build_graph(scenario)
        else:
            self.load_graph(scenario)

        
        self.compute_dijkstra(scenario)
        
 
        if compute_similarity:
            self.compute_similarity(scenario, plot=plot)

        if plot_plume_3D:
            self.plot_plume_3D(scenario)
        
        if compute_auto_thresh:
            self.compute_auto_thresh(scenario, plot=plot)
        if compute_data_dict:
            self.compute_data_dict(scenario, plot = plot)
            return scenario.data_dict
        
        if plot_graph_3D:
            self.plot_graph_3D(scenario)
            
        if self.verb:
            print(f'{datetime.now()} - END')

    def compute_data_dict(self, scenario, plot = True):
        """Returns a dict of scores for data analysis"""
        self.compute_similarity(scenario, plot = plot)
        
        scenario.data_dict = {'id': scenario.id,
                              'success': True,
                              'conc_max': scenario.conc_max,
                              'min_distance':np.min(scenario.distances),
                              't_sim': scenario.t_sim,
                              'ctm_x' : scenario.ctm_ptx,
                              'ctm_y': scenario.ctm_pty}
        scenario.data_dict.update(scenario.scores)
        sum_mass = np.sum(scenario.mf_map)
        sum_dist = np.sum(scenario.ig_map)
        scenario.data_dict['sum_mass'] = sum_mass
        scenario.data_dict['sum_dist'] = sum_dist

   

    def parallel_computation(self, list_of_scenario_id, **kwargs):
        """Calls Sim.main in parallel, with the same args, for the scenarios in list_of_scenario_id

        Args:
            list_of_scenario_id (_type_): List of int

        Returns:
            _type_: _description_
        """
        for id in list_of_scenario_id:
            self.instantiate_scenario(id, **kwargs)

        import multiprocessing
        from itertools import repeat
        kwargs_iter = list(repeat(kwargs, len(list_of_scenario_id)))
        kwargs_list = []
        for kwarg, scenario in zip(kwargs_iter, self.scenarios.values()):
            kwargs_list.append({**kwarg, **{'scenario': scenario}})
        with multiprocessing.Pool() as pool:
            result = starmap_with_kwargs(pool, self.main, kwargs_list)
        return result

    def compute_dijkstra(self, scenario:Scenario):
        """Computes dijkstra algorithm between source and target.

        Args:
            scenario (Scenario)
            layer (str, optional): _description_. Defaults to 'last'.

        Returns:
            _type_: _description_
        """        
        g = scenario.graph
        if self.verb:
            print(f'{datetime.now()} - COMPUTING DIJKSTRA WITH IGRAPH')
        target_layer = scenario.id_tgt

        shortest_paths = g.get_shortest_paths(
            scenario.id_src, to=target_layer, weights='weightK')
        distances = np.array(g.distances(
            source=scenario.id_src, target=target_layer, weights='weightK')[0])

        scenario.shortest_paths = shortest_paths
        scenario.distances_from_ig = distances
        scenario.ig_map = np.zeros((self.xxx.shape[2], self.xxx.shape[1]))
        for i in range(self.xxx.shape[1] * self.xxx.shape[2]):
            ig_id = target_layer[i]
            # if scenario.distances[i] ==np.inf:
            #     scenario.distances[i]
            scenario.ig_map[
                int(scenario.graph.vs[ig_id]['Z']//self.dz),
                int(scenario.graph.vs[ig_id]['Y']//self.dy)] = scenario.distances_from_ig[i]  # verified
        scenario.ig_map[scenario.ig_map > 1e308] = np.max(
            scenario.ig_map[scenario.ig_map < 1e308])  # remove inf values
        scenario.distances = scenario.ig_map.flatten()
        if self.verb:
            print(f'{datetime.now()} - DIJKSTRA COMPUTED')
        return scenario.ig_map
    
    
    
    
    ################################################################################
    ###   similarity study
    ################################################################################
    def compute_otsu(self,scenario, nb_classes = 2):
        """https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html

        Args:
            scenario (_type_): _description_
            nb_classes (int, optional): _description_. Defaults to 2.
        """
        bins = threshold_multiotsu(scenario.mf_map,nb_classes)
        image = np.digitize(scenario.mf_map, bins = bins)
        scenario.mf_m_otsu = image
    
    def compute_similarity(self, scenario:Scenario, plot=False):
        """Computes similarity score between cumulative mass (from MODFLOW) and graph distances distributions.

        Args:
            scenario (Scenario)
            plot (bool, optional): Plotting. Defaults to False.
        """        
        if scenario.similarity_masks is None:
            scenario.similarity_masks = dict()
            scenario.similarity_masks['mf'], otsu_thresh_value = compute_otsu(scenario.mf_map, nb_classes = 2)
            scenario.conc_max = np.max(scenario.mf_map)

            Y, Z = scenario.similarity_masks['mf'].nonzero()
            scenario.similarity_masks['mf_vec'] = np.concatenate(
                [Y[..., np.newaxis]*self.dy, Z[..., np.newaxis]*self.dz], axis=1)
            nb_pixels = int(np.sum(scenario.similarity_masks['mf']))
            scenario.similarity_masks['ig'],scenario.similarity_masks['ig_vec']  = compute_mask_from_nb(scenario.ig_map, nb_pixels, self.dy, self.dz)
            
            self.compute_scores(scenario)
            
        if plot:

            fig, axs = plt.subplots(1, 2, sharey = True)
            axs[1].imshow(scenario.similarity_masks['ig'], origin='lower')
            axs[1].set_title(r'Thresh. distance $X_{d}$')
            axs[1].set_xlabel(r'$y/\Delta y$')
            axs[0].imshow(scenario.similarity_masks['mf'], origin='lower')
            axs[0].set_title(r'Thresh. cumulative mass $X_{m}$ ')
            axs[0].set_xlabel(r'$y/\Delta y$')
            axs[0].set_ylabel(r'$z/\Delta z$')
            similarity = round(scenario.scores['similarity'],2)
            title =  r'$ \mu (X_{m}, X_{d}) =$' + f"{round(scenario.scores['similarity'],2)}"
            fig.suptitle(title,
                        y=0.80)

            os.makedirs(os.path.join(self.img_folder, 'similarity'), exist_ok = True)
            save_path = os.path.join(self.img_folder, 'similarity', f'{scenario.id}')
            fig.savefig(save_path,bbox_inches='tight', dpi = 300)
    
    def compute_scores(self, scenario):
        """Computes several scores between cumulative mass distribution and distance distribution. Returns the data as a dict.
        
        com_distance : Euclidean distance between center of masses,
        variance : Variance of the coordinates of the modflow data,
        jaccard_sim : Jaccard similarity index,
        wass : Wasserstein Distance,
        NWD : Normalized Wasserstein Distance,
        similarity : pondered metric, see paper,
        Spearman : Spearman coefficient,
        Pearson : Pearson coefficient.
        Args:
            scenario (Scenario)
        """        
        scenario.scores = {}
        com_distance = compute_com_distance(scenario.similarity_masks['ig_vec'],
                                  scenario.similarity_masks['mf_vec'])
        var = compute_variance(scenario.similarity_masks['mf_vec'])
        jaccard_sim = compute_jaccard_sim(scenario.similarity_masks['mf'],
                            scenario.similarity_masks['ig'])
        wass, NWD = compute_wass_distance(scenario.similarity_masks['ig_vec'],
                                  scenario.similarity_masks['mf_vec'])
        similarity = (NWD + jaccard_sim)/2
        self.compute_Spearman(scenario)
        self.compute_Pearson(scenario)
        scenario.scores = {
            'com_distance': com_distance  ,
            'var': var  ,
            'jaccard_sim': jaccard_sim  ,
            'wass': wass   ,
            'NWD': NWD,
            'similarity':similarity,
            'Spearman':scenario.Spearman,
            'Pearson':scenario.Pearson
            
        }


    def compute_Spearman(self, scenario:Scenario):
        from scipy import stats
        res = stats.spearmanr(-scenario.ig_map.flatten(),
                              scenario.mf_map.flatten())
        scenario.Spearman = res.statistic
        return res.statistic

    def compute_Pearson(self, scenario:Scenario):
        try:
            res = stats.pearsonr(-scenario.ig_map.flatten(),
                                 scenario.mf_map.flatten())
            scenario.Pearson = res.statistic
        except:
            scenario.Pearson = False

        return scenario.Pearson

    ################################################################################
    ###   Auto Thresholding
    ################################################################################    
    
    def compute_histogram(self, scenario:Scenario, bins=1000, plot=False):
        """Computes histogram of the distribution of distances. Softened by gaussian filter.

        Args:
            scenario (Scenario)
            bins (str, optional): _description_. Defaults to 'auto'.
            plot (bool, optional): _description_. Defaults to False.
        """
    
        from scipy.ndimage import gaussian_filter1d

        scenario.counts, scenario.bins = np.histogram(
            scenario.ig_map.flatten(), density=True, bins=bins)
        scenario.counts, scenario.bins = scenario.counts[:-2], scenario.bins[:-2]
        scenario.counts_b = gaussian_filter1d(scenario.counts, bins/200)
        if plot:
            x = scenario.bins[1:]
            y = scenario.counts
            y_b = scenario.counts_b
            fig, ax = plt.subplots()
            ax.plot(x[:-2], y_b[:-2])
            ax.set_xlabel('Distances')
            ax.set_ylabel('Density')
            ax.ticklabel_format(useMathText=True)
            os.makedirs(os.path.join(self.img_folder, 'histograms'), exist_ok = True)
            save_path = os.path.join(self.img_folder,'histograms/' + str(scenario.id))
            fig.savefig(save_path,bbox_inches='tight')

    def compute_reduced_density(self, scenario:Scenario, plot = False):
        """Computes density (with gaussian KDE) of the distance distribution restrained to the values before the caracteristic peak.

        Args:
            scenario (Scenario)
            plot (bool, optional)
        """
        self.compute_histogram(scenario,bins = 1000)
        y_b = scenario.counts_b[:-1]
        y = scenario.counts[:-1]
        x = scenario.bins[1:-1]
        half = y.shape[0]//2
        first_peak = np.argmax(y_b[0:half])
        value_peak = x[first_peak]
        flattened_map = scenario.ig_map[scenario.ig_map<value_peak].flatten()
        from scipy.stats import iqr
        std = np.std(flattened_map)
        IQR = iqr(flattened_map)        
        h = 1.06 *std * (flattened_map.shape[0])**(-1/5)
        # h = 0.9 * min( std,IQR) * (flattened_map.shape[0])**(-1/5)
        scenario.brandwidth = h
        
        X = flattened_map[:,None]
        from sklearn.neighbors import KernelDensity
        try :
            kernel = KernelDensity(kernel='gaussian', bandwidth=h)
            kernel.fit(X)
            X_d = np.linspace(np.min(X), np.max(X),1000)
            log_density = kernel.score_samples(X_d[:, None])
            scenario.x_density = X_d
            scenario.y_density = np.exp(log_density)
        except: #when there are not enough values to compute a density
            scenario.x_density = x
            scenario.y_density = y
        if plot:
            fig,ax = plt.subplots()
            ax.plot(scenario.x_density, scenario.y_density, alpha=0.5)
            ax.set_xlabel('Distances')
            ax.set_ylabel('Density')
            os.makedirs(os.path.join(self.img_folder, 'histograms'), exist_ok = True)
            save_path = os.path.join(self.img_folder,'histograms/' + str(scenario.id))
            fig.savefig(save_path)
    

    def compute_auto_thresh(self, scenario:Scenario, plot=True, method='otsu', parameters = None):
        """Computes an automatic thresholding to find threshold of significant data.

        Args:
            scenario (Scenario)
            plot (bool, optional): Defaults to True.
            method (str, optional): _description_. Defaults to 'otsu'.
            parameters (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        self.compute_reduced_density(scenario)
        id_peak = np.argmax(scenario.y_density)
        value_peak = scenario.x_density[id_peak]

                
        if method =='otsu': #works better
            nb_classes = 4
            if parameters is not None and 'nb_classes' in parameters.keys():
                nb_classes = parameters['nb_classes']
            n_map = scenario.ig_map[scenario.ig_map < value_peak]
            thresh_list = threshold_multiotsu(n_map,classes = nb_classes)
            

        if method == 'm_peak': #another temptative
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(-scenario.y_density[0:id_peak])
            nb_classes = 4
            if parameters is not None and 'nb_classes' in parameters.keys():
                nb_classes = parameters['nb_classes']
            thresh_list = [scenario.x_density[peak] for peak in peaks[0:nb_classes]]
        

        scenario.thresh_dict[method] = thresh_list
        if isinstance(thresh_list, float):
            bins = [thresh_list]
        else:
            bins = thresh_list
        binary_image = np.digitize(scenario.ig_map, bins = bins)
        binary_image = len(bins) - binary_image
        scenario.auto_masks['ig'] = binary_image
        
        bins_mf = threshold_multiotsu(scenario.mf_map,nb_classes)
        image_mf = np.digitize(scenario.mf_map, bins = bins_mf)
        scenario.auto_masks['mf'] = image_mf
        
        if plot:
            fig, axs = plt.subplots(1,2, layout = 'constrained', sharey= True)
            # fig.tight_layout()
            im = axs[0].imshow(scenario.mf_map, origin='lower', cmap = 'viridis')
            axs[0].set_title('Cumulative mass at FTA')
            axs[0].set_xlabel(r'$y/\Delta y$')
            axs[0].set_ylabel(r'$z/\Delta z$')
            fig.colorbar(im, ax = axs[0],shrink = 0.6, location = 'left')
            im2 = axs[1].imshow(scenario.auto_masks['ig'], origin='lower')
            axs[1].set_xlabel(r'$y/\Delta y$')
            fig.colorbar(im2, ax = axs[1],shrink = 0.6, location = 'right')
            axs[1].set_title('Distances with thresholding')
            
            
        if False: #another interesting visu
            fig, axs = plt.subplots(2, 2, layout = 'constrained')
            axs[0, 1].imshow(scenario.auto_masks['ig'], origin='lower')
            axs[0, 1].set_title('Distances (Otsu)')
            axs[0, 1].set_xlabel('x')
            axs[0, 1].set_ylabel('z')
            im = axs[1, 0].imshow(scenario.mf_map, origin='lower', cmap = 'viridis')
            fig.colorbar(im, ax = axs[1,0],shrink = 0.6, location = 'bottom')
            axs[1, 0].set_title('Cumulated mass')
            axs[1, 0].set_xlabel('x')
            axs[1, 0].set_ylabel('z')
            
            axs[0,0].imshow(scenario.auto_masks['mf'], origin = 'lower')
            axs[0,0].set_title('Cumulated mass (Otsu)')
        
                
            axs[1, 1].plot(scenario.x_density, scenario.y_density, c='black')
            axs[1, 1].set_title('Histogram (zoom & smooth)')
            
            axs[1, 1].vlines([thresh_list], ymin=0, ymax=np.max(
                scenario.y_density), color='red', linestyle='dotted', label = 'Thresholds')
            axs[1,1].legend()
            # fig.tight_layout()
            os.makedirs(os.path.join(self.img_folder, 'auto_threshold'), exist_ok = True)
            save_path = os.path.join(self.img_folder,'auto_threshold/' + str(scenario.id))
            fig.savefig(save_path)
        
  
        return scenario.auto_masks['ig'], value_peak, thresh_list

    ################################################################################
    ###   3D
    ################################################################################    
   
                
    def plot_graph_3D(self, scenario:Scenario, plot = True ,visu = 'browser'):
        """Computes the most visited nodes ny best paths for the 3D mesh/graph.

        Args:
            scenario (Scenario): Plotting
            plot (bool, optional): _description_. Defaults to True.
        """
        scenario.sum_3D_array = np.zeros(scenario.conc_time[0,:,:,:].shape)
        auto_mask, _, _ = self.compute_auto_thresh(scenario, plot = False)
        
        nb_indices = np.sum(auto_mask >= 1) #The number of paths considered is restricted to the ones who truly reach the significant spots.
        nb_indices = 100 #other possibility, used in the paper
        sorted_map = np.argsort(scenario.distances_from_ig)
        selected_indices = sorted_map[:nb_indices]
        
        for id_path in selected_indices:
            path = scenario.shortest_paths[id_path]
            for node_id in path:
                x_id = int(scenario.graph.vs[node_id]['X']//self.dx)
                y_id = int(scenario.graph.vs[node_id]['Y']//self.dy)
                z_id = int(scenario.graph.vs[node_id]['Z']//self.dz)
                scenario.sum_3D_array[z_id,y_id,x_id] += 1
        if plot:
            pio.renderers.default = visu
            fig = go.Figure(data=[])
            values = np.moveaxis(scenario.sum_3D_array,[0,2],[2,0])
            fig = go.Figure(data=go.Volume(
                x=self.xxx.flatten(),
                y=self.yyy.flatten(),
                z=self.zzz.flatten(),
                value=values.flatten(),
                isomin=nb_indices//10,
                isomax=nb_indices,
                opacity=0.7, 
                surface_count=21,
                colorscale='Reds',
                # showscale = False
                ))
            isomin = 1E0
            isomax = 1E2

            t_sim = scenario.t_sim
            t_sim_idx = np.argmin(np.abs(scenario.times-t_sim)) 

            
            values = np.moveaxis( scenario.conc_time[t_sim_idx, :, :, :],[0,2],[2,0])
            fig.add_trace(go.Volume(
                x=self.xxx.flatten(),
                y=self.yyy.flatten(),
                z=self.zzz.flatten(),
                value=values.flatten(),
                isomin=isomin,
                isomax=isomax,
                opacity=0.05,
                surface_count=21,
                colorscale='Viridis',
                showscale = False
                ))
            camera = dict(
                center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-5, z=-7), up = dict(x=-2,y=0,z=0)
)            
            title = 'Scenario  ' + str(scenario.id)
            fig.update_layout(scene_aspectmode='manual',title_text = title,
                             scene_aspectratio=dict(x=7, y=5, z=1),scene_camera = camera)
            
            fig.show()
            
            
        


class SimWithoutFault(GraphSim):
    """Child class for simulations with different MG realisations for each scenario.

    Args:
        GraphSim (_type_): _description_
    """
    
    def __init__(self, verb = True, data_path = '', si_factor = None, **kwargs):
        super().__init__(verb = verb, data_path = data_path, **kwargs)
        self.is_fault = False
        if si_factor is not None:
            self.parameters_dict['log10Ksi'] *= si_factor
        
    

    
    def generate_sim_data(self, **args):
        if not self.data_sim:
            self.nb_scenarios = 50

            # Generate 50 scenarios (50 different MG realisations)
        
            self.nmgs = 50
            mgs_list = []
            for k in range(self.nmgs):
                mgs_list += [k%self.nmgs, (k+1)%self.nmgs, (k+2)%self.nmgs]
            self.mgs_combinations = np.reshape(mgs_list, (50,3))
            
            
            with open(self.reggrid_path, 'rb') as f:
                content = pickle.load(f)
                [self.vx, self.vy, self.vz, self.nd_lithocodes, self.nd_topo_faults, self.ixflt1, self.ixflt2, self.ixflt3,
                self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz] = content[:20]

            self.xxx, self.yyy, self.zzz = np.meshgrid(
                self.vx, self.vy, self.vz, indexing='ij')  # build mesh
        self.data_sim = True
        return self.data_sim
            
    def generate_MGS_dict(self, scenario):
        """Specific function to retrieve certain data for Random MG simulations.

        Args:
            scenario (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.compute_dijkstra(scenario)
        self.compute_similarity(scenario)
        
        data_dict = {}
        
        flattened_index = np.argmax(scenario.mf_map)
        [data_dict['max_mf_y'], data_dict['max_mf_z']] = np.array(np.unravel_index(flattened_index, scenario.mf_map.shape))                                                                                                                                                                                                                                                                                                                                             
        flattened_index = np.argmin(scenario.ig_map)
        [data_dict['max_ig_y'], data_dict['max_ig_z']] = np.array(np.unravel_index(flattened_index, scenario.ig_map.shape))        
        
        data_dict['similarity'] =  scenario.scores['similarity']
        
        self.compute_auto_thresh(scenario)
        
        data_dict['size_mf'] = np.sum(scenario.auto_masks['mf'] > 0)
        
        data_dict['size_ig'] = np.sum(scenario.auto_masks['ig'] > 0)
        return data_dict
        
    def main(self, scenario, MGS_dict = False,  **kwargs):
        result = super().main(scenario, **kwargs)
        if result is not None:
            return result
            
        if MGS_dict:
            return self.generate_MGS_dict(scenario)
        
        
class SimWithFault(GraphSim):
    """Child class for simulations with faults.

    Args:
        GraphSim (_type_): _description_
    """
    def __init__(self, verb = True, data_path = '', **kwargs):
        super().__init__(verb = verb, data_path = data_path, **kwargs)
        self.is_fault = True
    
    
    
    def generate_sim_data(self, **args):
        if not self.data_sim: 
                    
            # Generate 80 scenarios (8 faults combination x 10 contaminant sources)
            
            self.mgs_combinations = np.zeros([80,3], dtype = int)
            self.mgs_combinations[:,0], self.mgs_combinations[:,1], self.mgs_combinations[:,2] = 0,1,2 #fixed MG
            with open(self.reggrid_path, 'rb') as f:
                content = pickle.load(f)
                [self.vx, self.vy, self.vz, self.nd_lithocodes, self.nd_topo_faults, self.ixflt1, self.ixflt2, self.ixflt3,
                self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz] = content[:20]

            self.xxx, self.yyy, self.zzz = np.meshgrid(
                self.vx, self.vy, self.vz, indexing='ij')  # build mesh
            self.nmgs = 3
        return self.data_sim
            

