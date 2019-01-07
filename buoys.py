'''Defines the main buoy class, a structure for holding data from CRREL 
ice mass balance buoys. Data is held in the tag .data, a dictionary of data
series objects, and in the tag .temp, a temperature series object'''


maindir = '/data/cr1/hadax/PhD/Buoys/' # This should be set to the main 
                                       # directory where the IMB data is held.
                                       # Subdirectories should have names 
                                       # equal to the IMB labels, e.g. '1997D',
                                       # '2012L', with all files for each IMB
                                       # placed under its directory
revision_number = 20107
buoydir = maindir

def buoylist(year=0):
    '''Returns list of all IMB labels'''
    import subprocess
    import data_series as ds

    command = ['ls',buoydir]    
    output = subprocess.Popen(command,stdout=subprocess.PIPE).communicate()[0]
    output = output.split('\n')
    
    if year != 0:
        restrict_output = [buoyname for buoyname in output if str(year) in buoyname]
    else:
        restrict_output = [buoyname for buoyname in output if len(buoyname)==5 and \
           ds.is_number(buoyname[0:4])]
	
    return restrict_output
    
	
def filelist(name):
    '''Returns list of all files associated with a given IMB'''
    import subprocess
    fulldir = buoydir + name + '/'
    command = ['ls',fulldir]
    
    output = subprocess.Popen(command,stdout=subprocess.PIPE).communicate()[0]
	
    output = output.split('\n')
    output.pop()
    return output


def dt_number(datetime):
    import datetime as dt
    date = dt.date(year=datetime.year,month=datetime.month,day=datetime.day)
    hr_float = datetime.hour / 24.
    number = date.toordinal() + hr_float
    
    return number
    
        

class buoy:
    def __init__(self,name):
        '''Creates a new empty buoy object'''
        self.name = name
        self.data = {}
        self.temp = None
        pass


    def buoy_file(self,varname):
        '''Identifies the file containing a given variable for a buoy'''
	import dictionaries
	fd = dictionaries.file_dic()

        bfiles = filelist(self.name)
	use_file = [bfile for bfile in bfiles if \
            sum([bfile.count(sstring) for sstring in fd[varname]]) > 0]
	if len(use_file)==0: return None

        return use_file[0]


    def find_file_ext_name(self,varname):
        '''Isolates the file extension name for the file containing variable
        varname for the given buoy, e.g. Temp, L3, clean'''
        use_file = self.buoy_file(varname)
        if use_file is None: return None
	file_ext_name = file_split(use_file)[1]
        return file_ext_name
    
    
    def extract_temp(self):
        '''Reads temperature data for a given buoy into the tag temp'''
    
        import linekey
	import dictionaries
	import csv
        import temp_series as tss
	
	fd = dictionaries.file_dic()
	
        bfiles = filelist(self.name)
	use_file = self.buoy_file('temp')
        if use_file is None: return None

	file_ext_name = self.find_file_ext_name('temp')	
        full_file = buoydir + self.name + '/' + use_file
	
        key = linekey.get_temp_linekey(full_file)
	
	ts = tss.temp_series(self.name,file_ext_name,key.phenomena_names)
	ts.read(full_file,key)
        self.temp = ts
	
        return ts
	
    
    def extract_data(self,varname,depth_marker=1):
        '''Reads data from variable varname into the data dictionary for the 
           given buoy. The data is stored as a data series object'''
        import numpy as np
	import dictionaries
        import data_series as ds

	td = dictionaries.title_dic()
	
	use_file = self.buoy_file(varname)
        if use_file is None: 
            series = ds.data_series(self.name,'',varname)
            self.data[varname] = series
            return series

	file_ext_name = self.find_file_ext_name(varname)	
		
        full_file = buoydir + self.name + '/' + use_file

        series = ds.data_series(self.name,file_ext_name,varname)
        series.read(full_file,varname)

        if len(series.data_list) > 0:
            if varname=='bottom' and np.sum(np.array(series.values()) > 0) > len(series.values())*0.75:
                for key in series.data_list.keys():
                    series.data_list[key] = 0. - series.data_list[key]
        else:
            series = ds.data_series(self.name,'',varname)

        self.data[varname] = series
	return series
   
    
    def estimate(self,varname,datetime):
        if not varname in self.data.keys():
            self.extract_data(varname)
        
        if len(self.data[varname].data_list) > 0:
            return_value = self.data[varname].estimate(datetime)
            return (1,return_value)
        else:
            print 'Data list is empty for this variable'
            return None
            


    def extract_datetime(self,varname):
        pass
    
    
    def depth_marker(self):    
        pass
	
	
    def map_track(self,mmap = False,show=True,axis=False,start_date=None,end_date=None,color=None,endpoints=True,legend=True):
        '''Produces a map of the track of a given buoy'''
        import datetime as dt
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
	
        if not start_date:
            start_date = dt.datetime(1900,1,1,1,1)
        
        if not end_date:
            end_date = dt.datetime(2100,1,1,1,1)

	if not mmap:
	    mmap = Basemap(width = 3.e6,height = 3.e6, resolution = 'l', projection='stere',lat_ts=0,lat_0=90,lon_0=0)
            mmap.fillcontinents(color='#A0FFE0')
            mmap.drawrivers(color='#000080')
            mmap.drawcoastlines()
            mmap.drawparallels(range(60,90,5),dashes=[2,2])
            mmap.drawmeridians(range(0,360,30),dashes=[2,2])

	latitude = self.extract_data('latitude')
	longitude = self.extract_data('longitude')

        latdates = latitude.dates()
        londates = longitude.dates()
        latlon_dates = list(set(latdates) & set(londates))
        latlon_dates.sort()
        latlon_values = [(latitude.data_list[date],longitude.data_list[date]) for date in latlon_dates if date >= start_date and date <= end_date]
	
	
        if len(latlon_values) > 0:
            longitude_values = [v[1] for v in latlon_values]
            latitude_values = [v[0] for v in latlon_values]
	    xpt, ypt = mmap(longitude_values,latitude_values)
	
	    if not color:
	        color = '#FF0000'
	
	    if not axis:
	        axis = plt.gca()
	    
            dates = latitude.dates()
	    axis.plot(xpt,ypt,label=self.name,color=color)
            if endpoints:
	        axis.plot(xpt[0],ypt[0],'o',color=color,label='Start ('+dates[0].strftime('%B %y')+')')
	        axis.plot(xpt[-1],ypt[-1],'o',color='#FFFFFF',label='Finish ('+dates[-1].strftime('%B %y')+')') 
            if legend:
                axis.legend(bbox_to_anchor = [.96,.8,.4,.2])
	
	    if show:
	        plt.show()
	
        #return mmap

	
    def show(self,varname_list,axis=0,show=True,start_date = None, \
              xlr = None, end_date = None, label=True,marker='+',\
              ylim = None, legend = True, colors = None, \
              legend_pos = [.1,.1,.8,.4]):
        '''Produces a timeseries plot of a list of data series variables 
           in a given buoy's .data tag (all on the same axes)'''
        import matplotlib.pyplot as plt
        import data_series as ds
        import numpy as np
        import datetime as dt
    
        if not axis:
	    axis = plt.axes()
	    
	plt.setp(axis.get_xticklabels(),visible=False)
    
        if colors is None:
            colors = 'r'*len(varname_list)

        for (ii,varname) in enumerate(varname_list):
            if not varname in self.data.keys():
                if varname[-2:]=='_r':
                    i_varname = varname[:-2]
                    self.extract_data(i_varname)
                    r_series = self.data[i_varname].regularise()
                    if not r_series:
                        r_series = ds.data_series(self.name,'Synthetic',varname)
                    self.data[r_series.label] = r_series
                else:
	            dds = self.extract_data(varname)
	    
	    if len(self.data[varname].data_list) > 0:
                dds = self.data[varname]
	        dates = dds.dates()
	        values = dds.values()
	        date_numbers = [date.toordinal() for date in dates]
	        axis.plot(dates,values,label=dds.label,marker=marker,ls='*',color=colors[ii])
	
	fig = plt.gcf()

        if legend:
	    axis.legend(bbox_to_anchor=legend_pos)

        ds_periods = [self.data[vn].period() for vn in varname_list \
             if not self.data[vn].period() is None]
        if len(ds_periods)==0:
            period_plot = [dt.date(1980,9,1),dt.date(1980,10,1)]
        else:
            period_plot = [np.min(np.array([dsp[0] for dsp in ds_periods])), \
                           np.max(np.array([dsp[1] for dsp in ds_periods]))]

        if not start_date is None:
            period_plot[0] = start_date
        if not end_date is None:
            period_plot[1] = end_date
	
        if not ylim is None:
            axis.set_ylim(ylim)

        ds.set_special_xaxis(axis,xlr=xlr,period_plot=period_plot,label=label)

	if show:
	    plt.show()
	    
        return axis


    def interface_from_surface(self):
        '''Calculates a likely snow-ice interface timeseries from the surface
           timeseries of a given buoy, assuming that the interface is 
           initially at elevation 0m, and remains constant until the surface
           elevation falls below its level, whereupon it decreases with the
           surface until the surface elevation begins to rise again.
           This method would fail in the presence of snow-ice
           formation.'''
        import datetime as dt
        import numpy as np
        import os
        import data_series as ds

        interface_series = ds.data_series(self.name,'Synthetic','interface_r')
        datafile = interface_series.data_file()

        if not 'surface' in self.data.keys():
            self.extract_data('surface')

        if not self.data['surface']:
            print 'No surface data to use as reference'
            return None

        if 'interface' in self.data.keys():
            if len(self.data['interface'].data_list) > 0:
                print 'This buoy already has an interface timeseries'
                return None

        sfc_period = self.data['surface'].period()
        sfc_period_dates = [dt.date(ddt.year,ddt.month,ddt.day) for ddt in sfc_period]
        sfc_period_numbers = [ddate.toordinal() for ddate in sfc_period_dates]

        int_min = 0.

        for number in range(sfc_period_numbers[0],sfc_period_numbers[1]+1):
            ddate = dt.date.fromordinal(number)
            ddatetime = dt.datetime(ddate.year,ddate.month,ddate.day,0,0)
            sfc_value = self.data['surface'].estimate(ddatetime)
            if sfc_value:
                int_min = np.min(np.array([int_min,sfc_value]))

            interface_series.data_list[ddatetime] = int_min

            interface_series.type = 'regular'
    
            
        self.data['interface_r'] = interface_series
        return interface_series


    def surface_from_interface_and_snow(self):
        '''Calculates surface timeseries from timeseries of snow depth and
           snow-ice interface.'''
        import datetime as dt
        import numpy as np
        import os
        import data_series as ds

        surface_series = ds.data_series(self.name,'Synthetic','surface_r')
        datafile = surface_series.data_file()

        for varname in ['interface','snow depth']:
            if not varname in self.data.keys():
                self.extract_data(varname)

            if not self.data[varname]:
                print 'No '+varname+' data to use as reference'
                return None

        if 'surface' in self.data.keys():
            if len(self.data['surface'].data_list) > 0:
                print 'This buoy already has a surface timeseries'
                return None

        int_period = self.data['interface'].period()
        int_period_dates = [dt.date(ddt.year,ddt.month,ddt.day) for ddt in int_period]
        int_period_numbers = [ddate.toordinal() for ddate in int_period_dates]

        snow_period = self.data['snow depth'].period()
        snow_period_dates = [dt.date(ddt.year,ddt.month,ddt.day) for ddt in snow_period]
        snow_period_numbers = [ddate.toordinal() for ddate in snow_period_dates]

        use_period = [max([int_period_numbers[0],snow_period_numbers[0]]),\
                      min([int_period_numbers[1],snow_period_numbers[1]])]

        if (use_period[0] > use_period[1]):
            print 'Can\'t estimate surface because periods for which we have'+\
                  'interface and snow data don\' overlap'
            return None

        for number in range(use_period[0]+1,use_period[1]+1):
            ddate = dt.date.fromordinal(number)
            ddatetime = dt.datetime(ddate.year,ddate.month,ddate.day,0,0)
            int_value = self.data['interface'].estimate(ddatetime)
            snow_value = self.data['snow depth'].estimate(ddatetime)

            sfc_value = int_value + snow_value
            surface_series.data_list[ddatetime] = sfc_value

        surface_series.type = 'regular'
    
            
        self.data['surface_r'] = surface_series
        return surface_series


    def icecond(self,ddatetime,iz_used=None):
        import numpy as np
        kice = 2.11
        
        for varname in ['interface','bottom']:
            if not varname in self.data.keys():
                self.extract_data(varname)
            if len(self.data[varname].data_list) == 0:
                print 'No interface or bottom data for this buoy'
                return None

        if not self.temp:
            print 'Buoy must have objective temperature series already associated'
            return None

        check_periods = [self.data[varname].period() for varname in \
                         ['interface','bottom']]
        check_periods.append(self.temp.period())
        too_early = np.any(np.array([ddatetime < cp[0] for cp in check_periods]))
        too_late = np.any(np.array([ddatetime > cp[1] for cp in check_periods]))
        if too_early or too_late:
            print 'Can\'t estimate ice conduction for this datetime'
            return None
        
        profile_top = self.data['interface'].estimate(ddatetime)
        profile_bot = self.data['bottom'].estimate(ddatetime)
        temp_profile = np.array(self.temp.estimate(ddatetime))
        zpoints = np.array(self.temp.zpoints())
        use_index = np.where(np.logical_and(zpoints > profile_bot,zpoints < profile_top))
 
        temp_ice = temp_profile[use_index[0]]
        z_ice = zpoints[use_index[0]]

        gradient = np.polyfit(z_ice,temp_ice,1)[0]
        icecond = gradient * kice
        iz_used = use_index[0]
        return icecond
        

    def all_pos_r(self,temp=True):
        '''The buoy elevation 'standardisation' method.
           Calculates regular timeseries of ice base elevation, ice thickness,
           snow-ice interface, snow depth and snow surface from all available 
           data, deducing as far as possible timeseries for which no data is 
           available from those for which there is data.
           If keyword 'temp' is set, additional timeseries are calculated
           with times-of-observation equal to those of the temperature data.'''
        import temp_series as tss

        if temp:
            self.extract_temp()
            tdates = self.temp.dates()
            if self.temp.classify()=='Subjective':
                self.temp.subjective_to_objective(dictionary=tss.standard_ztemp_subj_obj_dic())

        for var in ['surface','interface','bottom','ice thickness','snow depth']:
            self.extract_data(var)
            if self.data[var]:
                self.data[var+'_r'] = self.data[var].regularise()
                
                if temp:
                    self.data[var+'_rt'] = self.data[var].regularise_temp(tdates)
            else:
                self.data[var+'_r'] = None
                if temp:
                    self.data[var+'_rt'] = None

        if self.data['surface'] and not self.data['interface']:
            self.interface_from_surface()
            if temp:
                self.data['interface_rt'] = self.data['interface_r'].regularise_temp(tdates)
            else:
                self.data['interface_rt'] = None
                

        if self.data['interface'] and self.data['snow depth'] and not self.data['surface']:
            self.surface_from_interface_and_snow()
            if temp:
                self.data['surface_rt'] = self.data['surface_r'].regularise_temp(tdates)
            else:
                self.data['surface_rt'] = None
        
        

    def snowpos_rt(self,ddatetime):
        '''Returns elevation of the top and base of the snow layer for a given
           datetime which is a time-of-observation of temperature'''

        try:
            sseries = self.data['surface_rt']
        except KeyError:
            print 'No temp-regularised surface data available for '+self.name
            return None

        try:
            iseries = self.data['interface_rt']
        except KeyError:
            print 'No temp-regularised interface data available for '+self.name
            return None

        if (not sseries) or (not iseries):
            print 'Temperature regularised surface and interface timeseries required'
            return None

        try:
            spos = self.data['surface_rt'].data_list[ddatetime]
        except KeyError:
            return None

        try:
            ipos = self.data['interface_rt'].data_list[ddatetime]
        except KeyError:
            return None
        
        return [ipos,spos]
        

    def icepos_rt(self,ddatetime):
        '''Returns elevation of the top and base of the ice layer for a given
           datetime which is a time-of-observation of temperature'''

        try:
            bseries = self.data['bottom_rt']
        except KeyError:
            print 'No temp-regularised bottom data available for '+self.name
            return None

        try:
            iseries = self.data['interface_rt']
        except KeyError:
            print 'No temp-regularised interface data available for '+self.name
            return None

        if (not bseries) or (not iseries):
            print 'Temperature regularised bottom and interface timeseries required'
            return None

        try:
            bpos = self.data['bottom_rt'].data_list[ddatetime]
        except KeyError:
            return None

        try:
            ipos = self.data['interface_rt'].data_list[ddatetime]
        except KeyError:
            return None
        
        return [bpos,ipos]


    def average_snow_temp(self,restrict_hour=None):
        '''Returns data series of the average temperature of the snow layer,
           with times-of-observation equal to those of the temperature'''
        import tprof
        import data_series as ds
        import numpy as np
        tdates = self.temp.dates()
        varname = 'average_snow_temp'
        self.data[varname] = ds.data_series(self.name,'',varname)
        if not 'surface_rt' in self.data.keys() and \
           not 'interface_rt' in self.data.keys():
            print 'Not enough information provided'
            return self.data[varname]

        if self.data['surface_rt'].period() is None or \
           self.data['interface_rt'].period() is None:
            return self.data[varname]

        for tdate in tdates:
            if restrict_hour is None or tdate.hour == restrict_hour:
                if tdate >= max([self.data['surface_rt'].period()[0],\
                                 self.data['interface_rt'].period()[0]]) and \
                   tdate <= min([self.data['surface_rt'].period()[1],\
                                 self.data['interface_rt'].period()[1]]):
                    zint = [self.data['surface_rt'].data_list[tdate],\
                            self.data['interface_rt'].data_list[tdate]]

                    zprof = tprof.complete_zprof(self,tdate,zint)
                    if zprof is not None:
                        weights = tprof.zpt_weights(zprof[0])
                        mean_temp = np.sum(zprof[1] * weights) / np.sum(weights)
                        self.data[varname].data_list[tdate] = mean_temp

        self.data[varname].type = 'regular'
        return self.data[varname]


    def average_ice_temp(self,layer=1,nlayer=4,restrict_hour = None):
        '''Returns data series of the average temperature of given sections of 
          the ice layer, with times-of-observation equal to those of the 
          temperature.
          e.g. layer=1, nlayer=1 returns the average temp of the whole ice layer
               layer=1, nlayer=4 returns the average temp of the top 
               quarter of the ice layer
               layer=4, nlayer=4 returns the average temp of the bottom quarter
               of the ice layer'''
        import tprof
        import data_series as ds
        import numpy as np
        tdates = self.temp.dates()
        varname = 'average_ice_temp.ilayer.'+str(layer)+'.nlayer.'+str(nlayer)
        self.data[varname] = ds.data_series(self.name,'',varname)
        if not 'bottom_rt' in self.data.keys() and \
           not 'interface_rt' in self.data.keys():
            print 'Not enough information provided'
            return self.data[varname]

        if self.data['bottom_rt'].period() is None or \
           self.data['interface_rt'].period() is None:
            return self.data[varname]

        for tdate in tdates:
            if restrict_hour is None or tdate.hour == restrict_hour:
                if tdate >= max([self.data['bottom_rt'].period()[0],\
                                 self.data['interface_rt'].period()[0]]) and \
                   tdate <= min([self.data['bottom_rt'].period()[1],\
                                 self.data['interface_rt'].period()[1]]):
                    zint_f = np.array([self.data['interface_rt'].data_list[tdate],\
                            self.data['bottom_rt'].data_list[tdate]])
                    prop_start = float(layer-1) / float(nlayer)
                    prop_end = float(layer) / float(nlayer)
                    zint = [zint_f[0] + prop_start * (zint_f[1] - zint_f[0]),\
                            zint_f[0] + prop_end * (zint_f[1] - zint_f[0])]

                    zprof = tprof.complete_zprof(self,tdate,zint)
                    if zprof is not None:
                        weights = tprof.zpt_weights(zprof[0])
                        mean_temp = np.sum(zprof[1] * weights) / np.sum(weights)
                        self.data[varname].data_list[tdate] = mean_temp

        self.data[varname].type = 'regular'
        return self.data[varname]


    def new_sfc_temp_series(self,distance_from_sfc):
        '''Calculates a new data series of estimated temperature of an elevation
           which remains a constant distance from the top snow surface'''
        import data_series as ds
        import tprof

        self.process_temp()
        ddates = self.temp.dates()
        varname = 'sftemp_{:4.3f}'.format(distance_from_sfc)
        self.data[varname] = ds.data_series(self.name,'',varname)
        for ddate in ddates:
            snowpos = self.snowpos_rt(ddate)
            if not snowpos is None:
                temperature = tprof.estimate(self,ddate,snowpos[1]+distance_from_sfc)
                if not temperature is None:
                    self.data[varname].data_list[ddate] = temperature

        self.data[varname].type='regular_temp'
        return self.data[varname]


    def new_basal_temp_series(self,distance_from_base):
        '''Calculates a new data series of estimated temperature of an elevation
           which remains a constant distance from the ice base'''
        import data_series as ds
        import tprof

        self.process_temp()
        ddates = self.temp.dates()
        varname = 'temp_{:4.3f}'.format(distance_from_base)
        self.data[varname] = ds.data_series(self.name,'',varname)
        for ddate in ddates:
            icepos = self.icepos_rt(ddate)
            if not icepos is None:
                temperature = tprof.estimate(self,ddate,icepos[0]+distance_from_base)
                if not temperature is None:
                    self.data[varname].data_list[ddate] = temperature

        self.data[varname].type='regular_temp'
        return self.data[varname]


    def is_series(self,name):
        '''Returns True if a given data series has been saved to the main
           data directory.'''
        import os
        datadir = '/data/cr1/hadax/PhD/Buoys/'+self.name+'/'
        datafile = datadir + name + '.dat'
        return os.path.exists(datafile)


    def save_series(self):
        '''Saves all data series to the main data directory'''
        for series_name in self.data.keys():
            self.data[series_name].file_write()


    def update_series(self):
        '''Loads all data series saved in the main data directory to the .data
           tag of the given buoy'''
        import glob
        import data_series as ds
        datadir = buoydir+self.name+'/'
        files = glob.glob(datadir+'*.dat')
        for ffile in files:
            filename = ffile.split('/')[-1]
            varname = '.'.join(filename.split('.')[:-1])
            dds = ds.data_series(self.name,'',varname)
            dds.file_read()
            self.data[varname] = dds
            
    
    def process_temp(self):
        '''Reads and provides an initial 'clean' of ice and snow temperature
           data.'''
        import temp_series as tss
        self.extract_temp()
        if self.temp.classify()=='Subjective':
            self.temp.subjective_to_objective(dictionary=tss.standard_ztemp_subj_obj_dic())

        self.temp.mask()
        self.temp.translate()


    def translate_elevations(self):
        translation_file = '/data/cr1/hadax/PhD/Buoys/elevation_translations.txt'

        fileh = open(translation_file)
        line = fileh.readline()
        while line.strip()[1:-1] != self.name and \
            line != '':
            line = fileh.readline()

        line = fileh.readline()
        while line.strip() != '' and line != '':
            arguments = line.split(' ')
            if len(arguments) != 2:
                print 'Cannot parse translation file for buoy '+self.name
                return None

            series_name = arguments[0]
            try:
                series_translation = float(arguments[1])
            except ValueError:
                print 'Invalid translation value for buoy '+self.name+', series '+series_name

            if series_name in self.data.keys():
                self.data[series_name] = self.data[series_name] + series_translation

            line = fileh.readline()


    def surface_for_energy(self):
        import datetime as dt
        import numpy as np
        import data_series as ds
        ''' This method calculates a regular timeseries similar to the surface,
            but in which the rate of change is decreased by a factor R when the
            surface is located in the snow, where 
            R = (ice density)/(snow density)'''

        density_ice = 917.
        density_snow = 330.
        threshold = .015
        self.data['surface_for_energy_r'] = ds.data_series(self.name,\
            '','surface_for_energy_r')

        if 'surface_r' not in self.data.keys() or 'interface_r' not in \
              self.data.keys():
            print 'Cannot calculate surface_for_energy series for this buoy'
            return None

        sdates = list(set(self.data['surface_r'].dates()) & \
                      set(self.data['interface_r'].dates()))
        sdates.sort()

        if len(sdates) == 0:
             return None
           

        self.data['surface_for_energy_r'] = ds.data_series(self.name,\
            '','surface_for_energy_r')
        
        self.data['surface_for_energy_r'].data_list[sdates[0]] = \
                self.data['surface_r'].data_list[sdates[0]]

        for ii in range(len(sdates)-1):
            main_surf_last = self.data['surface_r'].data_list[sdates[ii]]
            main_surf_now  = self.data['surface_r'].data_list[sdates[ii+1]]
            
            change = main_surf_now - main_surf_last

            ice_surf_last = self.data['interface_r'].data_list[sdates[ii]]
            ice_surf_now = self.data['interface_r'].data_list[sdates[ii+1]]
            ice_portion = np.mean(np.array([\
                np.abs(ice_surf_last- main_surf_last) < threshold , \
                np.abs(ice_surf_last - main_surf_last) < threshold\
                    ]).astype('float'))
            scaled_change = change * ice_portion + \
                 (density_snow / density_ice) * change * (1 - ice_portion)

            self.data['surface_for_energy_r'].data_list[sdates[ii+1]] = \
                 self.data['surface_for_energy_r'].data_list[sdates[ii]] + \
                 scaled_change


def buoy_data_grid(varlist=[],file_out=None):
    import os
    nvars = len(varlist)
    buoy_names = buoylist()
    datadir = '/data/cr1/hadax/PhD/Buoys/'
    out_string = ''
    out_string = out_string + '{:12s}'.format('Buoy name')+\
          (nvars*'{:>14s}').format(*varlist) + '\n'
    for buoy in buoy_names:
        out_line = '{:12s}'.format(buoy)
        for var in varlist:
            subdir = datadir + buoy + '/'
            series_file = subdir + var + '.dat'
            result = os.path.exists(series_file)
            out_line = out_line + '{:14d}'.format(result)
        
        out_string = out_string + out_line + '\n'

    if file_out is None:
        print out_string
    else:
        fileh = open(file_out,'w')
        fileh.write(out_string)
        fileh.close()
            

def filelist(name):
    import subprocess
    buoydir = '/data/cr1/hadax/PhD/Buoys/'
    fulldir = buoydir + name + '/'
    command = ['ls',fulldir]
    
    output = subprocess.Popen(command,stdout=subprocess.PIPE).communicate()[0]
	
    output = output.split('\n')
    output.pop()
    return output


def file_split(filename):
    import re
    cpts = re.split('[._]',filename)
    return cpts    
