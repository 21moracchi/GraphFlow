from .graph_simulation import SimWithFault, Scenario
import gstools as gs
import os 
import numpy as np
from .parameters import *
import pickle
import sys
sys.path.append("..")
from utils.utils import *
import matplotlib.colors as colors 
import multiprocessing
    
    
class CrossComparison(SimWithFault):
    def __init__(self, verb=False, **kwargs):
        super().__init__(verb=verb, **kwargs)
        self.df_results = pd.DataFrame()
        self.list_dict_results = []

        
        
    def compute_cross_similarity(self, scenario_ref:Scenario, scenario_test:Scenario):
        """Computes the similarity between the cumulative mass distribution of scenario_ref and the distance distribution of scenario_test.

        Args:
            scenario_ref (Scenario)
            scenario_test (Scenario)

        """
        self.generate_K_field(scenario_ref, [0,1,2])
        self.main(scenario_ref) #pas de generate_K car is on randomise on veut que le principal ne soit pas aleatoire
        self.compute_similarity(scenario_ref)
        self.main(scenario_test,generate_K = True)
        nb_pixels = int(np.sum(scenario_ref.similarity_masks['mf']))
        mask_test, vec_test = compute_mask_from_nb(
            scenario_test.ig_map, nb_pixels, self.dy, self.dz)
        jaccard_sim = compute_jaccard_sim(scenario_ref.similarity_masks['mf'],
                                          mask_test)
        _, NWD = compute_wass_distance(vec_test,
                                          scenario_ref.similarity_masks['mf_vec'])
        cross_similarity = (NWD + jaccard_sim)/2
        return cross_similarity, mask_test


    def show_variability(self, source_id, nb_pixels=30, show = 'mass'):
        fig, axs = plt.subplots(3, 3)
        for k in range(0, 8):
            try:
                id = 10*k + source_id
                scenario = self.instantiate_scenario(id)
                self.main(scenario)
                fig_id = np.unravel_index(k+1, (3, 3))
                if show == 'mass':
                    im_to_show = scenario.mf_map
                if show == 'distance':
                    im_to_show,_ = compute_mask_from_nb(
                    scenario.ig_map, nb_pixels, self.dy, self.dz)
                axs[fig_id].imshow(im_to_show)
                axs[fig_id].set_title(self.parameters_dict['fault_array'][k])
                axs[fig_id].set_xlabel('y')
                axs[fig_id].set_ylabel('z')
            except:
                pass
        axs[0, 0].imshow(np.sum(scenario.realK, axis=2),
                         origin='upper', norm=colors.LogNorm(), cmap='viridis')
        axs[0, 0].scatter(scenario.ctm_pty//self.dy,
                          scenario.ctm_ptx//self.dx, c='red', marker='x')
        axs[0,0].set_xlabel(r'$y$')
        axs[0,0].set_ylabel(r'$x$')
        fig.tight_layout()
        # os.makedirs(os.path.join(self.img_folder, 'variability'), exist_ok = True)
        # save_path = os.path.join(self.img_folder, 'variability', f'{source_id}_{show}')
        # fig.savefig(save_path)
        return fig,axs 
    
    def parallel_cross_comparison(self, list_ids=None):
        if list_ids is None:
            list_ids = list(range(80))  # default value
        function_to_use = self.compute_cross_comparison
        self.instantiate_multiple_scenarios(list_ids)
        for k in range(len(list_ids)//4):  # 4 by 4 not to crash the kernel
            inf, sup = 4*k, min(len(list_ids), 4*k+4)
            ids_to_run = list_ids[inf:sup]
            scenarios_to_run = [self.scenarios[i] for i in ids_to_run]
            with multiprocessing.Pool() as pool:
                print(' Processing ...' + str(inf) + '...'+ str(sup))
                self.list_dict_results += pool.map(
                    function_to_use, scenarios_to_run)
        self.to_csv()

    def to_csv(self):
        if len(self.list_dict_results )>0:
            df = pd.DataFrame(self.list_dict_results)
            path = os.path.join(self.results_folder, f'cross_comparison.csv')
            df.to_csv(path)
    
    def compute_cross_comparison(self, scenario_ref:Scenario, plot=True, score=True, figure = 'realK'):
        """Compares the cumulative mass distribution of a reference scenario (scenario_ref) to the distances of each of the 8 scenarios.
        Computes for each pair the cross similarity and returns it as a dict. 

        Args:
            scenario_ref (Scenario): _description_
            plot (bool, optional): _description_. Defaults to True.
            score (bool, optional): _description_. Defaults to True.
            figure (str, optional): _description_. Defaults to 'realK'.

        Returns:
            dict : Dict with the cross similarities
        """
        source_id = scenario_ref.id % 10
        cross_similarity_array = [0]*8
        scenario_dict = {}
        scenario_ref.cross_score = {}
        for flt_id in range(0, 8):
            id = 10*flt_id + source_id
            scenario_test = self.instantiate_scenario(id)
            cross_similarity, mask_test = self.compute_cross_similarity(
                scenario_ref, scenario_test)
            cross_similarity_array[flt_id] = cross_similarity
            scenario_dict[flt_id] = scenario_test, mask_test

        scenario_ref.cross_similarity_array = cross_similarity_array
        best_scenarios = np.argsort(cross_similarity_array)[::-1]

        fig, axs = plt.subplots(3, 3)
        if figure == 'realK':
            axs[0, 0].imshow(np.sum(scenario_ref.realK, axis=2),
                                origin='upper', norm=colors.LogNorm(), cmap='viridis')
            axs[0, 0].scatter(scenario_ref.ctm_pty//self.dy,
                                scenario_ref.ctm_ptx//self.dx, c='red', marker='x')
        if figure == 'distances':
            axs[0, 0].imshow(scenario_ref.mf_map,
                                origin='lower', )
        if plot:
            for rk_flt_id in range(0, 8):
                if figure == 'realK':
                    
                    flt_id = best_scenarios[rk_flt_id]
                else :
                    flt_id = rk_flt_id 
                scenario, mask = scenario_dict[flt_id]
                fig_id = np.unravel_index(rk_flt_id+1, (3, 3))
                ax: plt.axis = axs[fig_id]
                if figure == 'realK':
                    ax.imshow(np.sum(scenario.realK, axis=2), origin='upper',
                            norm=colors.LogNorm(), cmap='viridis')
                    ax.scatter(scenario.ctm_pty//self.dy, scenario.ctm_ptx //
                            self.dx, c='red', marker='x')
                    title = str(round(cross_similarity_array[flt_id], 2))
                    ax.set_title(title)
                if figure == 'distances':
                    ax.imshow(mask, origin='lower', cmap='viridis')
                    title = str(round(cross_similarity_array[flt_id], 2))
                    title = f'{tuple(scenario.f1_f2_f3)} : {round(cross_similarity_array[flt_id], 2)} '
                    ax.set_title(title)
            fig.tight_layout()
            os.makedirs(os.path.join(self.img_folder, 'cross_comparaison'), exist_ok = True)
            save_path = os.path.join(self.img_folder, 'cross_comparaison', f'{scenario_ref.id}')
            fig.savefig(save_path)

        if score:
            scenario_ref.cross_score = dict(zip(range(8),cross_similarity_array))
           

            return scenario_ref.cross_score
    