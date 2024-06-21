import numpy as np
default_parameters_dict = {}
default_parameters_dict['ctm_location'] = np.array([[2011.8216247 , 2950.46369633,  512.5       ],
       [1644.15961272, 2948.64944714,  512.5       ],
       [1811.83145201, 2423.32644897,  512.5       ],
       [2327.70259382, 2409.19913637,  512.5       ],
       [2049.59368767, 2027.55911324,  512.5       ],
       [2253.51310867, 2538.14331322,  512.5       ],
       [1829.7317165 , 2788.42870343,  512.5       ],
       [1803.19482929, 2453.49788948,  512.5       ],
       [1634.04169725, 2403.11298645,  512.5       ],
       [1703.45524068, 2262.31334044,  512.5       ]])
default_parameters_dict['location_array'] = np.zeros((10,3))
default_parameters_dict['location_array'][:,0], default_parameters_dict['location_array'][:,1], default_parameters_dict['location_array'][:,2] = 1050, 2550, 512.5
# 8 COMBINATIONS OF 3 FAULTS ACTING AS PATH (1) OR BARRIER (0)
default_parameters_dict['fault_array'] = np.array([[0, 0, 0],
                                            [0, 0, 1],
                                            [0, 1, 0],
                                            [1, 0, 0],
                                            [0, 1, 1],
                                            [1, 1, 0],
                                            [1, 0, 1],
                                            [1, 1, 1]
                                            ])

        
        
# ****************************************************************************
# HYDROGEOLOGICAL PARAMETERS
# ****************************************************************************
default_parameters_dict['log10Kmu'] = np.log10(np.array([3.5E-5, 8E-4, 2E-5]))  # (m/s)
default_parameters_dict['log10Ksi'] = np.array([0.4, 0.5, 0.6])  # (m/s)
default_parameters_dict['porosity'] = 0.25
default_parameters_dict['fltlog10factor'] = 2
default_parameters_dict['generalFlowDir'] = np.array([1, 0, 0])
default_parameters_dict['Kxyratio'] = 1
default_parameters_dict['Kxzratio'] = 10
