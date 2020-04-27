def estimate(buoy,ddate,zpt):
    ddates = buoy.temp.dates()
    if not ddate in ddates:
        print('Date is not within temperature set')
        return None

    profile = buoy.temp.mprofile_set[ddate]
    zpts = buoy.temp.zpoints()

    result = estimate_prof(zpts,profile,zpt)
    return result

def estimate_prof(zpts,profile,zpt):
    import numpy as np

    where_present = np.where(profile.mask == False)
    profile_present = profile[where_present]
    if profile_present.shape[0] == 0:
        print('No data for this date')
        return None

    zpts_present = zpts[where_present]
    
    if zpt > zpts_present[0] or zpt < zpts_present[-1]:
        print('Z-point is not within vertical set')
        return None

    pts_diff = zpts_present - zpt
    pts_above = np.where(pts_diff > 0)[0]
    pts_equal = np.where(pts_diff == 0)[0]
    pts_below = np.where(pts_diff < 0)[0]

    if pts_equal.shape[0] > 0:
        return_value = profile_present[pts_equal[0]]
    else:
        above_index = pts_above[-1]
        below_index = pts_below[0]
        
        above_point = zpts_present[above_index]
        below_point = zpts_present[below_index]
        
        above_value = profile_present[above_index]
        below_value = profile_present[below_index]

        gradient = (above_value - below_value) / \
                   (above_point - below_point)

        return_value = below_value + gradient * (zpt - below_point)

    return return_value


def complete_zprof(buoy,ddate,zint,boundary_vars=None,adjust_for_snow=False,\
        salinity = 1):
    import numpy as np

    use_zpt_index = np.where(np.logical_and(buoy.temp.zpoints() >= zint[0],\
        buoy.temp.zpoints() <= zint[1]))[0]
    nz = use_zpt_index.shape[0]

    main_zpts = [buoy.temp.zpoints()[ii] for ii in use_zpt_index]

    if adjust_for_snow:
        main_profile_source = snow_adjusted_zprof(buoy,ddate, \
          salinity = salinity)
        if main_profile_source is None:
            print('No temperature profile for this date')
            return None
    else:
        try:
            main_profile_source = buoy.temp.mprofile_set[ddate]
        except KeyError:
            print('No temperature profile for this date')
            return None

    main_profile = \
        [main_profile_source.data[ii] for ii in use_zpt_index]
    main_mask = \
        [main_profile_source.mask[ii] for ii in use_zpt_index]

    temp_top = estimate_prof(buoy.temp.zpoints(),main_profile_source,\
                    zint[1])
    temp_bot = estimate_prof(buoy.temp.zpoints(),main_profile_source,\
            zint[0])
                                
    zpt_bot = zint[0]
    zpt_top = zint[1]

    if np.sum(np.array(main_mask)) > 0:
        print('This profile contains missing data')
        return None

    if type(temp_top) == type(None):
        print('Could not find top temp value for this date')
        return None

    if type(temp_bot) == type(None):
        print('Could not find bottom temp value for this date')
        return None

    full_zpts = np.array([zpt_top]+main_zpts+[zpt_bot])
    full_profile = np.array([temp_top]+main_profile+[temp_bot])

    return (full_zpts,full_profile)

    
def zpt_weights(zpts):
    import numpy as np

    tiny = 1.e-5
    zpt_min = np.min(zpts)
    zpt_max = np.max(zpts)
    zrange = zpt_max - zpt_min
    npts = zpts.shape[0]
    zpa = np.argsort(zpts)

    weights = np.zeros(npts)
    
    for ii in range(npts):
        z_index = zpa[ii]
        point = zpts[z_index]
        if ii==0:
            last_point = zpts[zpa[0]]
        else:
            last_point = zpts[zpa[ii-1]]
 
        if ii==npts-1:
            next_point = zpts[zpa[-1]]
        else:
            next_point = zpts[zpa[ii+1]]

        interval_base = (last_point + point) / 2.
        interval_top = (point + next_point) / 2.
        weights.itemset(z_index,interval_top - interval_base)

    weights = weights / np.sum(weights)
    return weights


def weighted_total_ice_tgrad(buoy,tperiod):
    import data_series as ds

    zpts = buoy.temp.zpoints()
    tdates = buoy.temp.dates()

    varname = 'weighted_total_ice_tgrad_tperiod_{:3.1f}'.format(tperiod)
    main_data_series = ds.data_series(buoy.name,'','data')
    main_weight_series = ds.data_series(buoy.name,'','weights')

    for ddt in tdates:
        main_data_series.data_list[ddt] = 0.
        main_weight_series.data_list[ddt] = 0.

    for zpt in zpts:
        print(zpt)
        tgrad_data = tgrad_series(buoy,zpt,tperiod)
        intersection_data = intersection_weights(buoy,zpt)
        weighted_tgrad = tgrad_data * intersection_data
        for ddt in intersection_data.dates():
            if intersection_data.data_list[ddt] == 0.:
                weighted_tgrad.data_list[ddt] = 0.

        main_data_series = main_data_series + weighted_tgrad
        main_weight_series = main_weight_series + intersection_data

    mean_data = main_data_series# / main_weight_series
    mean_data.varname = varname
    mean_data.label = varname
    mean_data.name = main_data_series.name
    return mean_data
    
        
def intersection_weights(buoy,zpt,mode='ice'):
    import numpy as np
    import data_series as ds

    if mode not in ['ice','snow']:
        print('Mode should be \'ice\' or \'snow\'')
        return None
 
    tdates = buoy.temp.dates()
    zpts = buoy.temp.zpoints()
    if zpt not in zpts:
        print('Given zpt is not in the set')
        return None

    varname = 'intersection_weights_'+mode+'_zpt_{:4.2f}'.format(zpt)
    series = ds.data_series(buoy.name,'',varname)
    buoy.data[varname] = series

    npts = zpts.shape[0]
    zpt_index = np.where(zpts==zpt)
    last_index = np.max(np.array([0,zpt_index[0][0]-1]))
    next_index = np.min(np.array([npts-1,zpt_index[0][0]+1]))
    for tdate in tdates:
        if mode=='ice':
            reference_interval = buoy.icepos_rt(tdate)
        elif mode=='snow':
            reference_interval = buoy.snowpos_rt(tdate)

        if reference_interval is not None:
            upper_point = min(zpts[last_index],reference_interval[1])
            lower_point = max(zpts[next_index],reference_interval[0])

            point_interval = [(lower_point + zpt)/2.,(upper_point + zpt)/2.]
            interval_length = max(0,point_interval[1] - point_interval[0])

            buoy.data[varname].data_list[tdate] = interval_length

    return buoy.data[varname]


def zgrad(buoy,date,zint,regularise=False,verbose=False,smooth=True,smooth_temp_pts = None):
    import buoys
    import numpy as np

    period = buoy.temp.period()
    if date < period[0] or date > period[1]:
        print('Date is not within the reporting period for this buoy')
        return None

    zpts_buoy = buoy.temp.zpoints()

    index_use = np.where(np.logical_and(zpts_buoy >= zint[0],zpts_buoy <= zint[1]))[0]

    if len(index_use) == 0:
        print('Buoy does not intersect given interval')
        return None

    if regularise:
        temp_prof = buoy.temp.estimate(date)
    else:
        try: 
            temp_prof = buoy.temp.mprofile_set[date]
        except KeyError:
            print('Buoy does not have data for this date')
            return None

    temp_use = temp_prof[index_use]
    zpts_use = zpts_buoy[index_use]

    index_use2 = np.where(temp_use.mask == False)
    temp_use = temp_use.data[index_use2]
    zpts_use = zpts_use[index_use2]    

    if smooth:
        if type(smooth_temp_pts) != type(None):
            varname_top = 'temp_{:4.3f}'.format(smooth_temp_pts[0])
            varname_bot = 'temp_{:4.3f}'.format(smooth_temp_pts[1])
            tempval_top = buoy.data[varname_top][date]
            tempval_bot = buoy.data[varname_bot][date]
            missing1 = tempval_top.mask==True
            missing2 = tempval_bot.mask==True
        else:
            tempval_top = estimate(buoy,date,zint[0])
            tempval_bot = estimate(buoy,date,zint[1])
            missing1 = type(tempval_top) == type(None) 
            missing2 = type(tempval_bot) == type(None)

        missing3 = temp_use.shape[0] == 0
   
        if sum([missing1,missing2,missing3]) > 1:
            return None

        temp_use = np.array([tempval_top]+list(temp_use)+[tempval_bot])
        zpts_use = np.array([zint[0]]+list(zpts_use)+[zint[1]])
        
    zpts_use = [zpt for (zpt,temp) in zip(zpts_use,temp_use) if temp != None]
    temp_use = [temp for temp in temp_use if temp != None]

    if verbose:
        print(temp_use)
        print(zpts_use)

    if buoy.temp.mdi in temp_use:
        return None
    else:
        rgrad = np.polyfit(zpts_use,temp_use,1)[0]

    return rgrad  


def zgrad_scaled(buoy,date,zint,regularise=False):
    zgrad_actual = zgrad(buoy,date,zint,regularise=regularise)

    icepos = buoy.icepos_rt(date)
    if not icepos:
        return None

    temp_grad_scale = zgrad(buoy,date,buoy.icepos_rt(date),regularise=regularise)

    if zgrad_actual and temp_grad_scale:
        return zgrad_actual / temp_grad_scale
    else:
        return None


def depth_scaled(buoy,date,zpos,icepos=None):
    if type(icepos) == type(None):
        icepos = buoy.icepos_rt(date)

    zpos_scale = (zpos - icepos[1]) / (icepos[0] - icepos[1])
    return zpos_scale


def tgrad_series(buoy,zpt,tperiod,verbose=False,mdtol = .5):
    import numpy as np
    import numpy.ma as ma
    import functions
    import data_series as ds

    tdates = buoy.temp.dates()
    zpts_buoy = buoy.temp.zpoints()
    if zpt not in zpts_buoy:
        print('Z-pt is not within this vertical set')
        return None

    zpt_index = np.where(zpts_buoy == zpt)
    
    tnumbers = np.array([functions.datetime_to_float(tdate) \
        for tdate in tdates])
    datavals = np.array([buoy.temp.mprofile_set[tdate][zpt_index[0]].data \
        for tdate in tdates])
    maskvals = np.array([buoy.temp.mprofile_set[tdate][zpt_index[0]].mask \
        for tdate in tdates])

    varname = 'tgrad_period_{:2.1f}_zpt_{:3.2f}'.format(tperiod,zpt)
    series = ds.data_series(buoy.name,'',varname)
    buoy.data[varname] = series

    for (tnumber,tdate) in zip(tnumbers,tdates):
        index = np.where(np.logical_and(tnumbers >= tnumber - tperiod/2,\
                                        tnumbers <= tnumber + tperiod/2))
        sub_datavals = datavals[index]
        sub_maskvals = maskvals[index]
        sub_datavals_masked = ma.masked_array(sub_datavals, mask = \
                                              sub_maskvals)

        npts = sub_maskvals.shape[0]

        first_halfmask = sub_maskvals[:npts/2]
        second_halfmask = sub_maskvals[npts/2:]

        enough_data = (np.sum(first_halfmask) < mdtol * npts/2) and \
                      (np.sum(second_halfmask) < mdtol * npts/2)
    
        if enough_data:
            use_index = np.where(1- sub_maskvals)[0]
            result = np.polyfit(tnumbers[index][use_index],\
                                sub_datavals_masked[use_index],1)[0]
            buoy.data[varname].data_list[tdate] = result

    return buoy.data[varname]


def tgrad(buoy,date,zint,tperiod,verbose=False,smooth=True,smooth_temp_pts = None):
    import buoys
    import numpy as np
    import numpy.ma as ma
    import functions

    period = buoy.temp.period()
    if date < period[0] or date > period[1]:
        print('Date is not within the reporting period for this buoy')
        return None

    zpts_buoy = buoy.temp.zpoints()

    index_use = np.where(np.logical_and(zpts_buoy >= zint[0],zpts_buoy <= zint[1]))[0]

    if len(index_use) == 0:
        print('Buoy does not intersect given interval')
        return None

    if regularise:
        temp_prof = buoy.temp.estimate(date)
    else:
        try: 
            temp_prof = buoy.temp.mprofile_set[date]
        except KeyError:
            print('Buoy does not have data for this date')
            return None

    date_number = functions.datetime_to_float(date)
    
    date_period = \
    [functions.float_to_datetime(number) for number in [date_number-tperiod/2,date_number+tperiod/2]]

    use_dates = [ddt for ddt in buoy.temp.dates() if ddt >= date_period[0] and ddt <= date_period[1]]
    
    use_zpt_index = np.where(np.logical_and(buoy.temp.zpoints() >= zint[0],\
        buoy.temp.zpoints() <= zint[1]))
    nz = use_zpt_index.shape[0]

    avg_temp = {}

    if type(smooth_temp_pts) != type(None):
        varname_top = 'temp_{:4.3f}'.format(smooth_temp_pts[0])
        varname_bot = 'temp_{:4.3f}'.format(smooth_temp_pts[1])

    for ddt in use_dates:
        try:
            main_zpts = [buoy.temp.zpoints()[ii] for ii in use_zpt_index]
            main_profile = \
                [buoy.temp.profile_set[ddt][ii] for ii in use_zpt_index]
        except KeyError:
            print('No temperature profile for this date')
            return None

        if smooth:
            if smooth_temp_points is None:
                temp_top = estimate(buoy,ddt,zint[1])
                temp_bot = estimate(buoy,ddt,zint[0])
            else:
                try:
                    temp_top = buoy.data[varname_top][ddt]
                except KeyError:
                    temp_top = None
                try:
                    temp_bot = buoy.data[varname_bot][ddt]
                except KeyError:
                    temp_bot = None
                icepos = buoy.icepos_rt(ddt)
                zpt_bot = icepos[0]
                zpt_top = icepos[1]

        if type(tempval_top) == type(None):
            print('Could not find top temp value for this date')
            return None

        if type(tempval_bot) == type(None):
            print('Could not find bottom temp value for this date')
            return None

            full_zpts = np.array([zpt_bot]+main_zpts+[zpt_top])
            full_profile = np.array([tempval_bot]+main_profile+[tempval_top])

        else:
            full_zpts = np.array(main_zpts)
            full_profile = np.array(main_profile)
            full_weights = np.repeat(1.,nz) / nz

    tgrads = []
    for zpt in use_zpts:
        if zpt in zpts_buoy:
            index = np.where(zpts_buoy == zpt)[0]
            tseries = [buoy.temp.profile_set[ddt][index] for ddt in use_dates]
        else:
            tseries = [estimate(buoy,ddt,zpt) for ddt in use_dates]

        numbers = [functions.datetime_to_float(ddt) for ddt in use_dates]

        tseries_use = [item for item in tseries if item != None]
        numbers_use = [number for (number,item) in zip (numbers,tseries) if item != None]

        if verbose:
            print(zpt, tseries, numbers)

        if buoy.temp.mdi in tseries:
            tgrad = buoy.temp.mdi
        else:
            tgrad = np.polyfit(numbers_use,tseries_use,1)[0]

        tgrads.append(tgrad)


    np_tgrads = np.array(tgrads)
    mean_tgrad = ma.mean(ma.masked_array(tgrads,mask=(np_tgrads == buoy.temp.mdi)))

    if ma.is_masked(mean_tgrad):
        return None
    else:
        return mean_tgrad


def zgrad_show(buoy,period,zint_rel,verbose=False):
    import matplotlib.pyplot as plt
    ddates = buoy.temp.dates()
    ddates_use = [ddt for ddt in ddates if ddt >= period[0] and ddt <= period[1]]

    zgs = []
    zgs_dates = []
    for ddate in ddates_use:
        icepos = buoy.icepos_rt(ddate)
        if icepos:
            if icepos[1] >= icepos[0] + zint_rel[1]:
                zint_zgrad = [icepos[0]+zp for zp in zint_rel]
                if verbose:
                    print('\n')
                    print(ddate)

                zgrad_single = zgrad(buoy,ddate,zint_zgrad,verbose=verbose)
                
                zgs.append(zgrad_single)
                zgs_dates.append(ddate)

    plt.plot(zgs_dates,zgs)
    plt.show()


def tgrad_show(buoy,period,zint_rel,verbose=False,tperiod=1.):
    import matplotlib.pyplot as plt
    seconds_in_day = 3600. * 24.
    ddates = buoy.temp.dates()
    ddates_use = [ddt for ddt in ddates if ddt >= period[0] and ddt <= period[1]]

    tgs = []
    tgs_dates = []
    for ddate in ddates_use:
        icepos = buoy.icepos_rt(ddate)
        if icepos:
            if icepos[1] >= icepos[0] + zint_rel[1]:
                zint_tgrad = [icepos[0]+zp for zp in zint_rel]
                if verbose:
                    print('\n')
                    print(ddate)

                tgrad_single = tgrad(buoy,ddate,zint_tgrad,tperiod,verbose=verbose)
                if tgrad_single:
                    tgrad_single = tgrad_single / seconds_in_day
                    
                tgs.append(tgrad_single)
                tgs_dates.append(ddate)

    plt.plot(tgs_dates,tgs)
    plt.show()


def show_profiles(buoy,period):
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    ddates = buoy.temp.dates()
    ddates_use = [ddt for ddt in ddates if ddt >= period[0] and ddt <= period[1]]

    for ddt in ddates_use:
        zdata = buoy.temp.profile_set[ddt]
        mzdata = ma.masked_array(np.array(zdata),mask=np.array(zdata)==buoy.temp.mdi)
        plt.plot(mzdata,buoy.temp.zpoints())
    
    xlim = plt.gca().get_xlim()
    for ddt in ddates_use:
        icepos = buoy.icepos_rt(ddt)
        if icepos:
            plt.plot(xlim,[icepos[0],icepos[0]],color='k',linestyle='--')
            plt.plot(xlim,[icepos[1],icepos[1]],color='k',linestyle='--')
        
    plt.show()


def snow_adjusted_zprof(buoy,ddate,snow_conductivity=0.33,\
        fresh_ice_conductivity=2.03,salinity=1):
    import numpy as np
    import copy

    offset = -.05
    beta_k_coefficient = 0.117

    temps = buoy.temp.mprofile_set[ddate]
    zpoints = buoy.temp.zpoints()

    snowpos = buoy.snowpos_rt(ddate)
    if snowpos is None:
        return None

    int_temp = estimate(buoy,ddate,snowpos[0]+offset)
    if int_temp is None:
        return None

    above_int_ind = np.where(zpoints >= snowpos[0]+offset)

    new_temps = copy.copy(temps)

    ice_conductivity = fresh_ice_conductivity + \
        beta_k_coefficient * salinity / int_temp
    mu = snow_conductivity / ice_conductivity
    new_temps[above_int_ind] = temps[above_int_ind] * mu + \
                               int_temp * (1-mu)

    return new_temps
