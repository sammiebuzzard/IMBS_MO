def standard_ztemp_subj_obj_dic():
    # Provides a standard set of rules for converting a series of z points to depth in the absence of labels
    # by assuming measurements to be made at 10cm intervals and the 7th measurement to lie at zero.
    
    return_value = {}
    for index in range(50):
        return_value[index] = 60. - float(index)*10.
	
    return return_value

class temp_series:
    def __init__(self,buoy_name,file_ext_name,znames):
        self.name = buoy_name
	self.fen = file_ext_name
        self.znames=znames
	self.mdi = -999.
	self.profile_set = {}
	
    def read(self,full_file,key):
        import csv
        import data_series as ds
	
	max_zlist = len(key.value_index)		
            
	fileh = open(full_file)
        rows = csv.reader(fileh)
        for row in rows:
	    if len(row)>0:
	        date_string = row[key.date_index]
		date = ds.process_dates(date_string,self.name,self.fen)
		
		if date != 0:
		
		    temp_list = []
		    for index in key.value_index:
                        if index >= len(row):
                            temp_value = self.mdi
                        else:
			    temp_string = row[index]
			    if len(temp_string) != 0 and \
                                 ds.is_number(temp_string):
			        temp_value = float(temp_string)
			    else:
			        temp_value = self.mdi
			
			temp_list.append(temp_value)
			
	            if len(temp_list) != max_zlist:
		        print 'Number of temp instances does not match key for date ', date
		        return 0
			
	            if temp_list.count(self.mdi) != len(temp_list):
		        self.profile_set[date] = temp_list

        fileh.close()


    def mask(self):
        import numpy as np
        import numpy.ma as ma        
        defined_masks = temp_mask(self.name)

        ddates = self.dates()

        mprofiles = {}
        for ddate in ddates:
            profile = self.profile_set[ddate]
            new_profile = ma.masked_array(np.array(profile),mask=(np.array(profile)==self.mdi))
            mprofiles[ddate] = new_profile

        for mask in defined_masks:
            ddates_mask = [ddt for ddt in ddates if ddt >= mask.period[0] and ddt <= mask.period[1]]
            for ddate in ddates_mask:
                mprofiles[ddate].mask[mask.zint[0]:mask.zint[1]] = True

        self.mprofile_set = mprofiles

    
    def classify(self):
	ttypes = [zname[1] for zname in self.znames]
        if ttypes.count('S')>0:
            return 'Subjective'
	else:
	    return 'Objective'
        
    
    def dates(self):
        date_values = self.profile_set.keys()
	date_values.sort()
        return date_values


    def values(self,position):
        
	if (position > len(self.znames)):
	    print 'No data available for this position'
	    return 0
	
	ddates = self.dates()
	vvalues = [self.profile_set[date][position] for date in ddates]
	return vvalues


    def mvalues(self,position):
        
	if (position > len(self.znames)):
	    print 'No data available for this position'
	    return 0
	
	ddates = self.dates()
	maskvalues = [self.mprofile_set[date].mask[position] for date in ddates]
	return maskvalues
	
	
    def zpoints(self):
        import numpy as np
	if self.classify()=='Subjective':
            print 'Can\'t define zpoints as temperature levels are not all objectively labelled.'
	    return 0
	    
	points = np.array([float(zname[2:]) for zname in self.znames]) / 100.
	
	return points
	
	
    def period(self):
	dates = self.dates()
        return (dates[0],dates[-1])
	
	
    def zshow(self,ddt,show=True,regularise=False,zint=None):
        import matplotlib.pyplot as plt
        zpts = self.zpoints()
        
        if not check_period(ddt,self.period()):
            print 'Requested datetime is not within the period for this data'
            return None

        if regularise:
            zvals = self.estimate(ddt)
        else:
            if ddt not in self.dates():
                print 'Data is not available for this datetime'
                return None
            else:
                zvals = self.profile_set[ddt]

        plt.plot(zvals,zpts)

        if zint:
            for zpt in zint:
                plt.plot(plt.gca().get_xlim(),(zpt,zpt),color='g',linestyle='--')

        if show:
            plt.show()


    def show(self,position,show=True,start_date = None, end_date = None,color = None, xlr=None, label=True, ylim = None):
        import numpy as np    
        import matplotlib.pyplot as plt
        import data_series as ds
	
        ddates = self.dates()
	vvalues = self.values(position)
        maskvalues = self.mvalues(position)
	
	vthere = [(date,value,mask) for (date,value,mask) in zip(ddates,vvalues,maskvalues) if value != self.mdi]
	unzip_vthere = (np.array([date for (date,value,mask) in vthere]),\
                        np.array([value for (date,value,mask) in vthere]),\
                        np.array([mask for (data,value,mask) in vthere]))

        show_index = np.where(1-unzip_vthere[2])
	plt.plot(unzip_vthere[0][show_index],unzip_vthere[1][show_index])

        period = self.period()
	if not start_date:
	    start_date = period[0]
	if not end_date:
	    end_date = period[1]
	    
        period_plot = [start_date,end_date]
	
	axis = plt.gca()
        ds.set_special_xaxis(axis,xlr=xlr,period_plot=period_plot,label=label)
        if show:
	    plt.show()


    def below_surface(self,sfc,date):
    
        if self.classify()=='Subjective':
	    print 'Cannot carry out this operation for subjectively labelled zpoints'
	    return 0
	            
        value = sfc.estimate(date)
	return [value >= zpt for zpt in self.zpoints()]
	

    def above_bottom(self,bot,date):
    
        if self.classify()=='Subjective':
	    print 'Cannot carry out this operation for subjectively labelled zpoints'
	    return 0
	            
        value = bot.estimate(date)
	return [value <= zpt for zpt in self.zpoints()]
	

    def values_2D(self):
        import numpy as np
	import numpy.ma as ma
        
	ddates = self.dates()
	nt = len(ddates)
	nz = len(self.znames)
	
	return_temp_array = np.zeros((nz,nt))
	return_temp_array_ma = ma.masked_array(return_temp_array,mask=(return_temp_array==self.mdi))
	
	for (i,date) in enumerate(ddates):
	    values = np.array(self.profile_set[date])
	    
	    return_temp_array[:,i] = values
	    
	return_temp_array_ma = ma.masked_array(return_temp_array,mask=(return_temp_array==self.mdi))
	
	return return_temp_array_ma


    def contour_subj(self,show=True):
        import matplotlib.pyplot as plt
	import numpy as np
        import data_series as ds
	
	nz = len(self.znames)
        
	ttypes = [zname[1] for zname in self.znames]
	subj = [ttype=='S' for ttype in ttypes]
        if subj.count(False)>0:
	    print 'Note that not all temperature levels are subjectively labelled'
	    
	v2 = self.values_2D()
	ddates = self.dates()
	points = np.array([int(zname[2:]) for zname in self.znames])
	points = 0 - points
	
	# Strip out dates without any data otherwise the plot will look rubbish
	dlogical = [sum(v2.mask[:,i]) < nz for (i,ddate) in enumerate(ddates)]
	use_dates = [date for (i,date) in enumerate(ddates) if dlogical[i]]
	use_indices = [i for (i,date) in enumerate(ddates) if dlogical[i]]
	use_values = v2[:,use_indices]
	#use_values = use_values[::-1,:]
	
	use_date_numbers = [date.toordinal() for date in use_dates]
        plt.contourf(use_date_numbers,points,use_values)
	axis = plt.gca()
	period_plot = self.period()
	ds.set_special_xaxis(axis,period_plot=period_plot)
	
	if show:
	    plt.show()
	    

    def contour_obj(self,show=True,levels=None,special_xaxis=True):
        import matplotlib.pyplot as plt
	import numpy as np
	import monty
        import data_series as ds
        cmap = monty.clr_cmap('/home/h01/hadax/IDL/colour_tables/temps.clr')

	nz = len(self.znames)
        
	ttypes = [zname[1] for zname in self.znames]
	obj = [ttype=='O' for ttype in ttypes]
        if obj.count(False)>0:
	    print 'Error: not all temperature levels are objectively labelled'
	    return 0
	    
	v2 = self.values_2D()
	ddates = self.dates()
	points = np.array([float(zname[2:]) for zname in self.znames]) / 100.
	
	# Strip out dates without any data otherwise the plot will look rubbish
	dlogical = [sum(v2.mask[:,i]) < nz for (i,ddate) in enumerate(ddates)]
	use_dates = [date for (i,date) in enumerate(ddates) if dlogical[i]]
	use_indices = [i for (i,date) in enumerate(ddates) if dlogical[i]]
	use_values = v2[:,use_indices]
	#use_values = use_values[::-1,:]
	
	use_date_numbers = [date.toordinal() for date in use_dates]
        clev = plt.contourf(use_date_numbers,points,use_values,cmap=cmap,levels=levels)
	axis = plt.gca()
	period_plot = self.period()
	if special_xaxis:
            ds.set_special_xaxis(axis,period_plot=period_plot)
        axis.set_ylabel('Depth (m')
	
	plt.colorbar(clev)
	if show:
	    plt.show()
	    
        return plt.gcf()

	    
    def subjective_to_objective(self,dictionary=None):
        nz = len(self.znames)
	
	new_znames = []
	for zname in self.znames:
	    if zname[1]=='S':
	        number = int(zname[2:])
            else:
                print 'Error: this temp series appears to be at least'+\
                      ' partially objective'
                return None
		
	    objective_value = dictionary[number]
	    new_znames.append('TO'+str(objective_value))
        
	self.subj_znames = self.znames
	self.znames = new_znames
	

    def extract_series(self,zname):
        import data_series as ds
        series = ds.data_series(self.name,'Temp',zname)
        zi = self.znames.index(zname)
        for date in self.dates():
            series.data_list[date] = self.profile_set[date][zi]

        series.type = 'irregular'
        return series


    def estimate(self,datetime):
        profile = []
        for zpt in self.znames:
            profile.append(self.extract_series(zpt).estimate(datetime))
        return profile


    def apply_translation(self,period,translate_value):
        '''This method takes a portion of the temperature data, in the interval
        specified by \'period\' and moves it up or down relative to the z-points
        by the requested value.  Required as some buoys appear to have their
        temperature reference levels reset at certain points (e.g. 2006C).'''
        import numpy as np
        import numpy.ma as ma

        if not hasattr(self,'mprofile_set'):
            print 'Temperature data must be masked to apply this method'
            return None

        if self.classify()=='Subjective':
            print 'Temperature series must be objectively labelled to apply this method'
            return None

        tiny = 1.e-5
        translate_series = temp_series(self.name,self.fen,self.znames)
        translate_series.mprofile_set = {}
        zpts = self.zpoints()
        ztrans_dic = {}
        for zpt in zpts:
            translate_from_zpt = zpt - translate_value
            if translate_from_zpt < zpts[-1] or translate_from_zpt > zpts[0]:
                ztrans_dic[zpt] = {}
            else:
                if sum(np.abs(zpts - translate_from_zpt) < tiny) > 0:
                    zind = np.where(np.abs(zpts - translate_from_zpt) < tiny)[0]
                    ztrans_dic[zpt] = {zind.item():1}
                else:
                    distance_array = zpts - translate_from_zpt
                    upper_ind = np.where(distance_array > 0.)
                    lower_ind = np.where(distance_array < 0.)
                    zind_upper = upper_ind[0][np.argmin(distance_array[upper_ind])]
                    zind_lower = lower_ind[0][np.argmin(0.-distance_array[lower_ind])]

                    zdist_upper = np.abs(translate_from_zpt - zpts[zind_upper])
                    zdist_lower = np.abs(translate_from_zpt - zpts[zind_lower])

                    weight_upper = zdist_lower / (zdist_lower + zdist_upper)
                    weight_lower = 1. - weight_upper

                    ztrans_dic[zpt] = {zind_upper.item():weight_upper, zind_lower.item():weight_lower}

        tdates = self.dates()
        tdates_apply = [tdate for tdate in tdates if (tdate >= period[0] and tdate <= period[1])]
        
        for tdate in tdates:
            if tdate >= period[0] and tdate <= period[1]:

                temp_array = np.zeros(len(zpts))
                mask_array = np.zeros(len(zpts),dtype='bool')

                for (ii,zpt) in enumerate(zpts):
                    zinds = ztrans_dic[zpt].keys()
                    if len(zinds) == 0:
                        mask_array.itemset(ii,True)
                    else:
                        zweights = [ztrans_dic[zpt][zind] for zind in zinds]
                        temp_points = [self.mprofile_set[tdate].data[zind] for zind in zinds]
                        mask_points = [self.mprofile_set[tdate].mask[zind] for zind in zinds]
                        if True in mask_points:
                            mask_array.itemset(ii,True)
                        else:
                            temp_value = np.sum(np.array([temp*weight for (temp,weight) in \
                                            zip(temp_points,zweights)]))
                            temp_array.itemset(ii,temp_value)

                mtemp_array = ma.masked_array(temp_array,mask=mask_array)

                translate_series.profile_set[tdate] = temp_array
                translate_series.mprofile_set[tdate] = mtemp_array
    
            else:
                translate_series.profile_set[tdate] = self.profile_set[tdate]
                translate_series.mprofile_set[tdate] = self.mprofile_set[tdate]
                
        return translate_series


    def translate(self):
        import datetime as dt
        import copy

        tr_file = '/data/cr1/hadax/PhD/Buoys/temp_translations.txt'
        fileh = open(tr_file)
        line = fileh.readline()
        while line.strip()[1:-1] != self.name and \
            line != '':
            line = fileh.readline()

        line = fileh.readline()
        use_series = copy.deepcopy(self)
        while line.strip() != '' and line != '':
            arguments = line.split(' ')
            if len(arguments) != 3:
                print 'Cannot parse translation file for buoy '+self.name
                return None

            try:
                startdate = dt.datetime(*([int(item) for item in \
                                           arguments[0].split('-')]))
                enddate   = dt.datetime(*([int(item) for item in \
                                           arguments[1].split('-')]))
            except TypeError:
                print 'Date arguments are in the wrong format for buoy '+self.name
                return None

            try:
                translation = float(arguments[2])
            except ValueError:
                print 'Translation argument is in the wrong format for buoy '+\
                    self.name

            use_series = use_series.apply_translation([startdate,enddate],\
                                       translation)

            line = fileh.readline()
    
    
        for tdate in self.dates():
            self.profile_set[tdate] = use_series.profile_set[tdate]
            self.mprofile_set[tdate] = use_series.mprofile_set[tdate]
    
    


def check_period(ddt,period):
    return ddt > period[0] and ddt < period[1]


class single_temp_mask:
    def __init__(self,period,zint):
        self.period = period
        self.zint = zint
    

def temp_mask(buoy_name):
    import datetime as dt
    mask_file = 'temp_mask.txt'
    fileh = open(mask_file)

    found_buoy = False
    while not found_buoy:
        line = fileh.readline()
        if len(line) >= 6:
            found_buoy = line[1:6] == buoy_name

    if not found_buoy:
        print 'No reference to this buoy in mask file'
        return None

    reading_entries = True
    entries = []
    while reading_entries:
        line = fileh.readline()
        if line.rstrip():
            elements = line.rstrip().split(' ')
            print elements
            if len(elements) < 4:
                reading_entries = False
                break
    
            start_date_elts = elements[0].split('-')
            end_date_elts = elements[1].split('-')

            try:
                start_date_elts = [int(elt) for elt in start_date_elts]
                end_date_elts   = [int(elt) for elt in end_date_elts]
                start_zint = int(elements[2])
                end_zint = int(elements[3])
            except ValueError:
                print 'Incorrect format of date:'
                print line
                return 0

            period = [dt.datetime(*start_date_elts),dt.datetime(*end_date_elts)]
            zint = [start_zint,end_zint]

            entry = single_temp_mask(period,zint)
            entries.append(entry)
    
        else:
            reading_entries = False

    return entries
