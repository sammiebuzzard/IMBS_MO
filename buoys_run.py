'''Main driver program for processing IMB data'''

import buoys
import numpy as np

temp_series_base_positions = np.arange(0,1,.1)
temp_series_sfc_positions = np.arange(-.5,.5,.1)

base_layers_calc = [[0,.1],[0,.2],[0,.3],[0,.4],[0,.5],[0,.6],\
                    [.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],[.6,.7],\
                    [.1,.3],[.2,.4],[.3,.5],[.4,.6],[.5,.7],[.6,.8],\
                    [.1,.4],[.2,.5],[.3,.6],[.4,.7],[.5,.8],[.6,.9],\
                    [.1,.5],[.2,.6],[.3,.7],[.4,.8],[.5,.9],[.6,1.]]


buoy_list = buoys.buoylist()

for buoy_name in buoy_list:
    buoy_str = buoys.buoy(buoy_name)
    buoy_str.process_temp()
    buoy_str.calculate_elevation_series()
    buoy_str.position_rt()

    buoy_str.temp_series_base(temp_series_base_positions)
    for base_layer_calc in base_layers_calc:
        buoy_str.temp_statistics(layer_calc=base_layer_calc,mode='base')
    buoy_str.temp_series_sfc(temp_series_sfc_positions)
    #for sfc_layer_calc in sfc_layers_calc:
    #    buoy_str.temp_statistics(sfc_layer_calc,mode='interface')

    buoy_str.effective_thickness(ice_cond=2.03,snow_cond=0.33)
    buoy_str.adjusted_zgrad_snow(int_length=0.5,ice_cond=2.03,snow_cond=0.33,\
       mean_salinity=1.)
    buoy_str.save_rt_nc()
