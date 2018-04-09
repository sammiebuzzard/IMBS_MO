def calculate_elevation_series(quiet=True,blist=None):
    import numpy as np
    import buoys
    import data_series as ds

    buoy_list = buoys.buoylist()

    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    basic_variable_names = ['interface','snow depth','surface','bottom']
    if not quiet:
        print ('{:3s} '*4).format('sfc','snd','int','bot')

    for buoy_name in buoy_list:

        buoy = buoys.buoy(buoy_name)
        buoy.extract_temp()

        results = np.zeros(4,dtype='int32')
        for (ii,varname) in enumerate(basic_variable_names):
            buoy.extract_data(varname)
            if len(buoy.data[varname].data_list) != 0:
                if varname == 'surface' and len(buoy.data['interface'].data_list) > 0:
                    use_series = buoy.data['surface'].snap(buoy.data['interface_r'])
                else:
                    use_series = buoy.data[varname]
                buoy.data[varname+'_r'] = use_series.regularise()
                buoy.data[varname+'_rt'] = use_series.regularise_temp(buoy.temp.dates())
                results.itemset(ii,1)
            else:
                buoy.data[varname+'_r'] = ds.data_series(buoy_name,'',varname+'_r')
                buoy.data[varname+'_rt'] = ds.data_series(buoy_name,'',varname+'_rt')

        if len(buoy.data['interface'].data_list)==0 and len(buoy.data['surface'].data_list) > 0:
            buoy.interface_from_surface()
            buoy.data['interface_rt'] = buoy.data['interface_r'].regularise_temp(buoy.temp.dates())
            results.itemset(0,2)

        if len(buoy.data['surface'].data_list)==0 and len(buoy.data['interface'].data_list) > 0 \
               and len(buoy.data['snow depth'].data_list) > 0:
            buoy.surface_from_interface_and_snow()
            buoy.data['surface_rt'] = buoy.data['surface_r'].regularise_temp(buoy.temp.dates())
            results.itemset(2,2)

        
        buoy.data['ice_thickness'] = buoy.data['interface_rt'] - \
                                     buoy.data['bottom_rt']
        buoy.data['ice_thickness'].label = 'ice_thickness'
        buoy.data['snow_thickness'] = buoy.data['surface_rt'] - \
                                      buoy.data['interface_rt']
        buoy.data['snow_thickness'].label = 'snow_thickness'

        buoy.save_series()
        if not quiet:
            print ('{:6s}'+'{:3d} '*4).format(buoy_name,*results)



def lei_temp_series(quiet=True,blist=None,force_new=False):
    import buoys
    required_distances = [0.,.4,.7]
    buoy_list = buoys.buoylist()

    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print ' '
            print 'Calculating Lei temperature series for buoy '+buoy_name
            print '----------------------------'

        buoy = buoys.buoy(buoy_name)
        buoy.update_series()            
        for distance in required_distances:
            series_name = 'temp_{:4.3f}'.format(distance)
            if (not buoy.is_series(series_name)) or force_new:
                buoy.new_basal_temp_series(distance)
                if len(buoy.data[series_name].data_list) > 0:
                    result = 1
                else:
                    result = 0
                buoy.save_series()
            else:
                result = 2

            if not quiet:
                print '{:5.2f}{:2d}'.format(distance,result)


def sfc_temp_series(quiet=True,blist=None,force_new=False):
    import buoys
    required_distances = [-.5,-.3,-.2,-.1,0.,.1]
    buoy_list = buoys.buoylist()

    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print ' '
            print 'Calculating Lei temperature series for buoy '+buoy_name
            print '----------------------------'

        buoy = buoys.buoy(buoy_name)
        buoy.update_series()            
        for distance in required_distances:
            series_name = 'sftemp_{:4.3f}'.format(distance)
            if (not buoy.is_series(series_name)) or force_new:
                buoy.new_sfc_temp_series(distance)
                if len(buoy.data[series_name].data_list) > 0:
                    result = 1
                else:
                    result = 0
                buoy.save_series()
            else:
                result = 2

            if not quiet:
                print '{:5.2f}{:2d}'.format(distance,result)


def zgrad_top_snow_adjusted(blist=None,force_new=False,quiet=True,\
        int_length = 0.5, mean_salinity = 1):
    import buoys
    import numpy as np
    import data_series as ds
    import tprof

    buoy_list = buoys.buoylist()

    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print 'Calculating top gradient for buoy '+buoy_name
        
        buoy = buoys.buoy(buoy_name)
        buoy.update_series()
        buoy.process_temp()

        series_names = ['sftemp_'+'{:6.3f}'.format(0.-int_length),'sftemp_0.000']
        if set(series_names) & set(buoy.data.keys())==set(series_names):
            boundary_vars = series_names
        else:
            boundary_vars = None

        tdates = buoy.temp.dates()
        varname = 'zgrad_top_snow_adjusted_'+'{:6.3f}'.format(0.-int_length)
        zgrad_adjust_series = ds.data_series(buoy.name,'',varname)
        for tdate in tdates:
            snowpos = buoy.snowpos_rt(tdate)
            if snowpos is not None:
                zint = [snowpos[1]-int_length,snowpos[1]]

                complete_zprof = tprof.complete_zprof(buoy,tdate,zint,\
                    boundary_vars = boundary_vars, adjust_for_snow=True,\
                    salinity = mean_salinity)

                if complete_zprof is not None:
                    zgrad_adjust = np.polyfit(complete_zprof[0],complete_zprof[1],1)[0]
                    zgrad_adjust_series.data_list[tdate] = zgrad_adjust

        buoy.data[varname] = zgrad_adjust_series
        buoy.data[varname].type = 'regular_temp'
        buoy.save_series()


def lei_statistics(zint=None,quiet=True,blist=None,force_new=False,\
       relative_position = 'bottom'):
    import buoys
    import tprof
    import data_series as ds
    import numpy as np

    if relative_position == 'bottom':
        icepos_index = 0
        prefix = ''
        boundary_vars = ['temp_{:4.3f}'.format(zpos) for zpos in zint]
    elif relative_position == 'interface':
        icepos_index = 1
        prefix = 'int_'
        boundary_vars = None

    statistics = ['zmean','zgrad','z_recip_mean','z_recip_sq_mean']
    if zint is None:
        print 'Please specify z-interval'

    layer_label = '{:4.3f}_{:4.3f}'.format(*zint)
    varnames = [prefix + layer_label + '_' + stat for stat in statistics]

    buoy_list = buoys.buoylist()

    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print ' '
            print 'Calculating Lei statistics of '+\
                  str(zint[0])+'-'+str(zint[1])+' heat storage series '+\
                  'for buoy '+buoy_name
            print '----------------------------'

        buoy = buoys.buoy(buoy_name)
        not_there_log = np.array([(not buoy.is_series(varname)) \
                                      for varname in varnames])
        if np.sum(not_there_log) > 0 or force_new:
            buoy.update_series()          
            buoy.process_temp()  
    
            for varname in varnames:
                buoy.data[varname] = ds.data_series(buoy.name,'',varname)
                buoy.data[varname].type = 'regular_temp'
    
            tdates = buoy.temp.dates()
            for tdate in tdates:
                if tdate.day==1 and tdate.hour==1:
                    print tdate

                icepos = buoy.icepos_rt(tdate)
                if not icepos is None:
                    zint_transform = [zi+icepos[icepos_index] for zi in zint]
                    zprof = tprof.complete_zprof(buoy,tdate,zint_transform,\
                        boundary_vars=boundary_vars)
                    if not zprof is None:
                        zweights = tprof.zpt_weights(zprof[0])

                        zgrad = np.polyfit(zprof[0],zprof[1],1)[0]
                        zmean = np.sum(zweights * zprof[1])
                        z_recip_mean = np.sum(zweights * 1. / zprof[1])
                        z_recip_sq_mean = np.sum(zweights * 1. / (zprof[1])**2.)

                        data_series = [zmean,zgrad,z_recip_mean,\
                                           z_recip_sq_mean]
                        for (ii,varname) in enumerate(varnames):
                            buoy.data[varname].data_list[tdate] = \
                               data_series[ii]
    
            result = 1
            buoy.save_series()
        else:
            result = 2
    
        if not quiet:
            print result


def lei_heat_balance(ztri=[0,.4,.7],quiet=True,blist=None):
    import my_fluxes
    import buoys
    import numpy as np
    buoy_list = buoys.buoylist()

    writedir = '/data/cr1/hadax/PhD/Buoys/'
    filenames = ['my_conductive_flux.dat','my_heat_uptake.dat',\
                 'my_basal_change.dat',\
                 'my_ocean_heat_flux.dat']
    labels = ['Conductive flux','Heat uptake','Basal change','Ocean heat flux']
    files = [writedir + filename for filename in filenames]
    filehandles = [open(ffile,'w') for ffile in files]

    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for (label,fileh) in zip(labels,filehandles):
        fileh.write(label+'\n')
        fileh.write('{:6s},{:4s},{:6s},{:12s},{:12s},{:12s},{:4s}\n'.\
                    format('Buoy','Mon','Year','Value','Prec','Sal','RC'))
        fileh.write('-----------------------------\n')

    for buoy_name in buoy_list:
        if not quiet:
            print 'Calculating Lei heat balance for buoy '+buoy_name

        cf  = my_fluxes.f_conductive_flux(buoy_name,ztri[1:])
        hu  = my_fluxes.f_heat_uptake(buoy_name,ztri[:2])
        bc  = my_fluxes.f_basal_change(buoy_name)
        ohf = hu + bc - cf
        ohf.label = 'Ocean heat flux'

        list_all_my = list(set.union(*[set(item.my_list()) for item in\
                                               (cf,hu,bc,ohf)]))
        if len(list_all_my) > 0:
            list_all_my = my_fluxes.my_sort(np.array(list_all_my))
            list_all_my = [tuple(list_all_my[ii,:]) for ii in \
                  range(list_all_my.shape[0])]
            for my_point in list_all_my:
                for (iss,series) in enumerate((cf,hu,bc,ohf)):
                    try:
                        index = series.my_list().index(my_point)
                    except ValueError:
                        index = None
                    if not index is None:
                        ce = series.central_estimates[index]
                        pe = series.precision_error[index]
                        se = series.salinity_error[index]

                        out_str = '{:6s},{:4d},{:6d},{:12.3f},{:12.3f},{:12.3f}\n'.\
                                            format(buoy_name,my_point[0],\
                                            my_point[1],ce,pe,se)

                        filehandles[iss].write(out_str)

    for fileh in filehandles:
        fileh.close()
               

def calculate_ice_snow_thickness(quiet=True,blist=None,force_new=False):
    
    import buoys
    import numpy as np
    import data_series as ds

    buoy_list = buoys.buoylist()

    ice_series_label = 'Ice_thickness'
    snow_series_label = 'Snow_thickness'
    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:

        buoy = buoys.buoy(buoy_name)
        buoy.update_series()
        if (not buoy.is_series(ice_series_label)) or force_new:
            if 'interface_rt' in buoy.data.keys() and \
               'bottom_rt' in buoy.data.keys():
                ice_thickness = buoy.data['interface_rt'] - buoy.data['bottom_rt']
                ice_thickness.label = ice_series_label

                buoy.data[ice_thickness.label] = ice_thickness
                buoy.save_series()
                ice_result = 1
            else:
                buoy.data[ice_series_label] = ds.data_series(buoy_name,' ',\
                    ice_series_label)
                buoy.save_series()
                ice_result = 0
        else:
            ice_result = 2
    
        if (not buoy.is_series(snow_series_label)) or force_new:
            if 'interface_rt' in buoy.data.keys() and \
               'surface_rt' in buoy.data.keys():
                snow_thickness = buoy.data['surface_rt'] - \
                    buoy.data['interface_rt']
                snow_thickness.label = snow_series_label

                buoy.data[snow_thickness.label] = snow_thickness
                buoy.save_series()
                snow_result = 1
            else:
                buoy.data[snow_series_label] = ds.data_series(buoy_name,' ',\
                    snow_series_label)
                buoy.save_series()
                snow_result = 0
        else:
            snow_result = 2

        if not quiet:
            print '{:6s}{:4d}{:4d}'.format(buoy_name,ice_result,snow_result)


def write_spatial_codes(quiet=True,blist=None):
    import buoys
    import spatial
    import my_fluxes
    import numpy as np

    outfile = '/data/cr1/hadax/PhD/Buoys/spatial_codes.dat'
    outfileh = open(outfile,'w')

    outfileh.write('Codes to denote regions occupied by buoys during particular months\n')
    outfileh.write('1 = North Pole\n')
    outfileh.write('2 = Beaufort Sea\n')
    outfileh.write('0 = neither\n')
    outfileh.write('-1 = no lat or lon data available\n')
    outfileh.write('\n')
    outfileh.write('------------------------------------------------------------------\n')
    outfileh.write('{:6s},{:4s},{:6s},{:6s}\n'.format('Buoy','Mon','Year','Code'))
    
    buoy_list = buoys.buoylist()
    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print 'Calculating spatial codes for buoy '+buoy_name

        buoy = buoys.buoy(buoy_name)
        buoy.update_series()

        if np.sum(np.array([name not in buoy.data.keys() for \
            name in ['longitude','latitude']])) == 0:
            my_points = my_fluxes.valid_month_points([buoy.data[name] for \
                name in ['longitude','latitude']])
            for my_point in my_points:
                code = spatial.region_my(buoy,my_point)
                outfileh.write('{:6s},{:4d},{:6d},{:6d}\n'.format(\
                    buoy_name, my_point[0], my_point[1], code))

    outfileh.close()


def write_all_my_series(filename,funcname,quiet=True,blist=None,label='',**kwargs):
    import buoys

    ddir = '/data/cr1/hadax/PhD/Buoys/'
    ffile = ddir + filename
    fileh = open(ffile,'w')

    fileh.write(label+'\n')
    fileh.write('{:6s},{:4s},{:6s},{:12s},{:12s},{:12s},{:4s}\n'.\
                    format('Buoy','Mon','Year','Value','Prec','Sal','RC'))
    fileh.write('-----------------------------\n')

    buoy_list = buoys.buoylist()
    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        buoy = buoys.buoy(buoy_name)
        buoy.update_series()

        if not quiet:
            print 'Calculating series '+\
                  'for buoy '+buoy_name

        my_series = funcname(buoy_name,**kwargs)
        my_points = my_series.my_list()
        for (ii,my_point) in enumerate(my_points):
            ce = my_series.central_estimates[ii]
            pe = my_series.precision_error[ii]
            se = my_series.salinity_error[ii]

            out_str='{:6s},{:4d},{:6d},{:12.3f},{:12.3f},{:12.3f}\n'.\
                                        format(buoy_name,my_point[0],\
                                        my_point[1],ce,pe,se)

            fileh.write(out_str)

    fileh.close()


def effective_thickness(quiet=True,blist=None,snow_conductivity = 0.33):
    import buoys
    ice_conductivity = 2.03

    buoy_list = buoys.buoylist()
    
    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print 'Calculating effective thickness for buoy '+buoy_name
        buoy = buoys.buoy(buoy_name)
        buoy.update_series()

        series_label = 'effective_thickness.'+snow_conductivity.__str__()
        buoy.data[series_label] = \
            buoy.data['ice_thickness'] / ice_conductivity + \
            buoy.data['snow_thickness'] / snow_conductivity

        buoy.data[series_label].label = series_label
        buoy.save_series()


def full_tgrads(quiet=True,blist=None,tperiod=1.,force_new=False):
    import tprof
    import buoys

    buoy_list = buoys.buoylist()
    
    if not blist is None:
        buoy_list = [bb for bb in buoy_list if bb in blist]

    for buoy_name in buoy_list:
        if not quiet:
            print 'Calculating full depth tgrad in ice for buoy '+buoy_name
 
        buoy = buoys.buoy(buoy_name)
        buoy.update_series()
        buoy.process_temp()
        varname = 'weighted_total_ice_tgrad_tperiod_{:3.1f}'.format(tperiod)
        if force_new or not buoy.is_series(varname):
            total_tgrad_series = tprof.weighted_total_ice_tgrad(buoy,tperiod)
            buoy.data[total_tgrad_series.varname] = total_tgrad_series
            buoy.data[total_tgrad_series.varname].type = 'regular_temp'
            buoy.save_series()
