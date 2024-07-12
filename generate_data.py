###file to execute to generate both data/random_sim/results/general_data.csv and data/fault_sim/results/cross_comparison.csv
###can take a while to run.

from classes.graph_simulation import SimWithFault, SimWithoutFault
from classes.fault_recognition import CrossComparison
import pandas as pd
import numpy as np
import os
parameters = {'location_array':np.array([[2011.8216247 , 2950.46369633,  512.5       ],
       [1644.15961272, 2948.64944714,  512.5       ],
       [1811.83145201, 2423.32644897,  512.5       ],
       [2327.70259382, 2409.19913637,  512.5       ],
       [2049.59368767, 2027.55911324,  512.5       ],
       [2253.51310867, 2538.14331322,  512.5       ],
       [1829.7317165 , 2788.42870343,  512.5       ],
       [1803.19482929, 2453.49788948,  512.5       ],
       [1634.04169725, 2403.11298645,  512.5       ],
       [1703.45524068, 2262.31334044,  512.5       ]])}




df_destination = './data/fault_sim/results'
df_name = 'general_data.csv'
ids_list = list(range(0,80))
sim = CrossComparison(data_path='./data/fault_sim', parameters_dict =parameters)
data_list = sim.parallel_computation(ids_list, compute_data_dict = True)
df = pd.DataFrame(data_list)
df.to_csv(os.path.join(df_destination,df_name))

sim.parallel_cross_comparison(parallel = True, method = 'graph')
