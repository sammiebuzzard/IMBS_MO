'''Defines the data series object, a structure for holding a single data 
time series from a CRREL ice mass balance buoy. Data is held in .data_list, 
a dictionary in which the keys are Python datetime objects. Data series can 
be added, subtracted etc.'''

class data_series:
    def __init__(self,buoy_name,file_ext_name,varname):
        '''Creates a new empty data series object'''
        self.name = buoy_name
        self.fen = file_ext_name
        self.label = varname
        self.data_list = {}
        self.type = ''


    def __add__(self,other,name=None,label=None):
        '''Adds two data series objects together, with data points created only
           at coincident date times. Alternatively, adds
           a constant to the given data series.'''
        new_series = self.combine(other,name=name,label=label)

        ddates1 = self.dates()
        for ddate in ddates1:
            if isinstance(other,data_series):
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] + other.data_list[ddate]
                except KeyError:
                    pass
            else:
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] + other
                except TypeError:
                    print('Unable to add this object')

        return new_series


    def __sub__(self,other,name=None,label=None):
        '''Subtracts one data series object from another, with data points 
           created only at coincident date times. Alternatively, subtracts 
           a constant from the given data series.'''
        new_series = self.combine(other,name=name,label=label)

        ddates1 = self.dates()
        for ddate in ddates1:
            if isinstance(other,data_series):
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] - other.data_list[ddate]
                except KeyError:
                    pass
            else:
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] - other
                except TypeError:
                    print('Unable to subtract this object')

        return new_series


    def __mul__(self,other,name=None,label=None):
        '''Multiplies two data series objects together, with data points 
           created only at coincident date times. Alternatively, multiplies
           the given data series by a constant.'''
        new_series = self.combine(other,name=name,label=label)

        ddates1 = self.dates()
        for ddate in ddates1:
            if isinstance(other,data_series):
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] * other.data_list[ddate]
                except KeyError:
                    pass
            else:
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] * other
                except TypeError:
                    print('Unable to multiply by this object')

        return new_series


    def __truediv__(self,other,name=None,label=None):
        '''Divides one data series object by another, with data points 
           created only at coincident date times. Alternatively, divides
           the given data series by a constant.'''
        new_series = self.combine(other,name=name,label=label)

        ddates1 = self.dates()
        for ddate in ddates1:
            if isinstance(other,data_series):
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] / other.data_list[ddate]
                except KeyError:
                    pass
            else:
                try:
                    new_series.data_list[ddate] = \
                        self.data_list[ddate] / other
                except TypeError:
                    print('Unable to divide by this object')

        return new_series
    
        
    def combine(self,other,name=None,label=None):
        if isinstance(other,data_series):
            if not label is None:
                new_label = label
            elif self.label == other.label:
                new_label = self.label
            else:
                new_label = None

            if not name is None:
                new_name = name
            elif self.name == other.name:
                new_name = self.name
            else:
                new_name = None

            if self.type == other.type:
                new_type = self.type
            else:
                new_type = ' '

            new_series = data_series(new_name,' ',new_label)
            new_series.type = new_type
        else:
            import copy
            new_series = copy.deepcopy(self)
            new_series.label = 'New series'
    
        return new_series

        
    def read(self,data_file,varname):
        '''Given an IMB source file, reads the data into a data series object'''
    
        import csv
        import linekey
        import functions
        
        key = linekey.get_linekey(data_file,[varname],self.name)

        vscale_vars = ['surface','interface','bottom','snow depth','ice thickness']
        
        if (key is None or key.value_index.count(-1) > 0):
            print('Could not find variable '+varname)
            self.data_list = {}
            return None
            
        fileh = open(data_file)
        rows = csv.reader(fileh)
        
        for row in rows:
            if len(row) > 0:
                date_string = row[key.date_index]
                date = process_dates(date_string,self.name,self.fen)

                if (date is not None):

                    if key.value_index[0] < len(row):
                        value_string = row[key.value_index[0]]
                        if functions.is_number(value_string):
                            value = float(value_string) 

                            if (key.lat_flip_ns[0] and key.lat_flip_ns[1]==key.value_index[0]):
                                ns_value = row[key.lat_flip_ns[2]]
                                if (ns_value == 'S'):
                                    value = 0. - value

                            if (key.lon_flip_ew[0] and key.lon_flip_ew[1]==key.value_index[0]):
                                ew_value = row[key.lon_flip_ew[2]]
                                if (ew_value == 'W'):
                                    value = 0. - value

                            if key.fliplon and varname=='longitude':
                                value = 0. - value

                            if varname in vscale_vars:
                                value = value * key.vertical_scale

                            self.data_list[date] = value

                        else:
                            if varname in ['latitude','longitude']:
                            
                                first_part = value_string[:-2]
                                second_part = value_string[-2:]
                                if functions.is_number(first_part) and second_part.strip() in ['N','S','E','W']:
                                    value = float(first_part)
                                    if second_part.strip() in ['S','W']:
                                        value = 0. - value
                                    self.data_list[date] = value
        
        if len(self.data_list) > 0:
            self.type = 'irregular'
             
        fileh.close()
            
    def show(self,show=True,start_date = None, end_date = None,color = None, xlr=None, label=True, ylim = None, 
        estimate_date_list=[],ecolor = '#ff0000'):
        '''Produces time series plot of a given data series'''
        import matplotlib.pyplot as plt
        import datetime as dt
        import numpy as np
        import functions
        
        axis = plt.axes()
            
        plt.setp(axis.get_xticklabels(),visible=False)

        dates = self.dates()
        date_numbers = [functions.datetime_to_float(ddt) for ddt in dates]
        if not date_numbers:
            date_numbers = [1]
        
        period = self.period()
        if not start_date:
            start_date = period[0]
        if not end_date:
            end_date = period[1]
            
        values = list(self.values())
        
        if not color:
            color = '#0000FF'
            
        axis.plot(date_numbers,values,'g+',markersize = 8,markeredgewidth=3,color = color)
        
        if len(estimate_date_list) > 0:
            values = [self.estimate(ddate) for ddate in estimate_date_list]
            axis.plot(estimate_date_list,values,marker='o',markersize=10,markeredgewidth=4,color = ecolor)
        
        axis = plt.gca()

        if ylim:
            axis.set_ylim(ylim)
            
        period_plot = [start_date,end_date]
        set_special_xaxis(axis,xlr=xlr,period_plot=period_plot,label=label)
                
        axis.set_ylabel('Depth (m)')
        axis.set_title(self.name)
        if show:
            plt.show()
        
        
    def dates(self):
        '''Returns ordered list of datetime objects corresponding to times of 
        observation for a given data series'''
        ddates = list(self.data_list.keys())
        ddates.sort()
        return ddates
                
                             
    def values(self):
        '''Returns list of data values for a given data series. The list is
        ordered by time of observation'''
        dates = self.dates()
        values = [self.data_list[date] for date in dates]
        return values
        
        
    def period(self):
        '''Returns first and last datetime object for a given data series
        (i.e. period of validity)'''
        dates = list(self.data_list.keys())
        dates.sort()

        if len(dates) > 0:
            return (dates[0],dates[-1])
        else:
            return None


    def contains_period(self,start_date,end_date):
        dsp = self.period()

        return dsp[0] <= start_date and dsp[1] >= end_date
        
        
    def classify(self,date_examine):
        '''Used in data series regularisation; decides whether a data series 
        is 'sparse' or 'dense' at any point in time within the period of
        validity'''
        import numpy as np
        import datetime as dt
    
        dates = list(self.data_list.keys())
        dates.sort()

        if len(dates)==0:
            print('Data series is empty')
            return None        

        number_examine = date_examine.toordinal() + date_examine.hour/24. + date_examine.minute/(24.*60.)
    
        if isinstance(dates[0],dt.datetime):
            numbers = [date.toordinal() + date.hour/24. + date.minute/(24.*60.) for date in dates]
        else:
            numbers = [date.toordinal() for date in dates]
            
        result = 'indeterminate'
        
        numbers_before = [number for number in numbers if number<=number_examine]
        numbers_after  = [number for number in numbers if number>=number_examine]
        
        if len(numbers_before)==0 or len(numbers_after)==0:
            result = 'outside'
            return result
        
        numbers_within_day_before = [number for number in numbers if number>=number_examine-1 and 
                                                                     number<=number_examine]                                                                 
        numbers_within_day_after  = [number for number in numbers if number>=number_examine and 
                                                                     number<=number_examine+1]
                                                                     
        n_before_and_after = (len(numbers_within_day_before),len(numbers_within_day_after))
        
        if sum(n_before_and_after) <= 2:
            result = 'sparse'
        
        elif n_before_and_after[0] <=3 and n_before_and_after[1] <=3:
            numbers_within = numbers_within_day_before + numbers_within_day_after
            gaps = np.array(numbers_within[1:]) - np.array(numbers_within[:-1])
            
            if np.mean(gaps) < 0.95:
                result = 'dense'

        else:
            result = 'dense'
            
        return result   
                
                             
    def estimate(self,date_examine):
        '''Estimates the value of given data series at any point in time
           within the period of validity. The estimate is calculated using 
           either linear interpolation (if the data series is classified as
           sparse at that point) or binomial mean (if the data series is 
           classified as dense at that point.'''
    
        if self.classify(date_examine)=='outside':
            result = None
        elif self.classify(date_examine)=='dense' and self.type=='irregular':
            result = self.estimate_binomial_mean(date_examine,radius=1.)
        elif self.classify(date_examine)=='sparse' or self.type=='regular':
            result = self.estimate_interpolate(date_examine)
        elif self.type=='regular_temp':
            try:
                result = self.data_list[date_examine]
            except KeyError:
                result = self.estimate_interpolate(date_examine)
        else:
            result = None
            
        return result
            
            
    def estimate_binomial_mean(self,date_examine,radius=1.):
        '''Estimates the value of given data series at any point in time
           within the period of validity using binomial mean. The mean is taken
           over the period centred on date_examine, with length (in days) of
           radius * 2'''

        import functions
        import datetime as dt

        sigma = radius * 60. * 60. * 24. * radius
        dates = self.dates()
        if isinstance(dates[0],dt.datetime):
            numbers = [date.toordinal() + date.hour/24. + date.minute/(24.*60.) for date in dates]
            
            result = functions.erf_average(dates,list(self.values()),date_examine,sigma)
            
        else:
            print('Dense data should have datetime coordinates, not just dates.  Something is wrong.')
            result = None
        
        return result
        pass


    def restrict(self,start_date = None, end_date = None):
        import copy
        import datetime as dt

        if start_date == None:
            start_date = dt.datetime(1900,1,1,0,0)
        if end_date == None:
            end_date = dt.datetime(2100,1,1,0,0)

        new_series = copy.deepcopy(self)

        ddates = self.dates()
        ddates_use = [ddate for ddate in ddates if ddate >= start_date and \
                      ddate <= end_date]
        
        new_series.data_list = {ddt:self.data_list[ddt] for ddt in \
                                ddates_use}
        return new_series


    def estimate_interpolate(self,date_examine):
        '''Estimates the value of given data series at any point in time
           (date_examine) within the period of validity using interpolation'''
        import numpy as np
        import datetime as dt

        number_examine = date_examine.toordinal() + date_examine.hour/24. + date_examine.minute/(24.*60.)
    
        dates = list(self.data_list.keys())
        dates.sort()
        if isinstance(dates[0],dt.datetime):
            numbers = [date.toordinal() + date.hour/24. + date.minute/(24.*60.) for date in dates]
        else:
            numbers = [date.toordinal() for date in dates]
        
        numbers_before = [number for number in numbers if number<=number_examine]
        numbers_after  = [number for number in numbers if number>=number_examine]
        
        if len(numbers_before) == 0 or len(numbers_after) == 0:
            return None
         
        closest_before,closest_after = (max(numbers_before),min(numbers_after))
        
        if closest_before != closest_after:
            ind_before,ind_after = (numbers.index(closest_before),numbers.index(closest_after))

            value_before,value_after = (self.data_list[dates[ind_before]],self.data_list[dates[ind_after]])
            return_value = np.interp(number_examine,[closest_before,closest_after],[value_before,value_after])
        else:
            ind_before = numbers.index(closest_before)
            value_before = self.data_list[dates[ind_before]]
            return_value = value_before
            
        return return_value


    def snap(self,reference_series):
        import numpy as np
        '''This function creates new data points for the input series equal to data points
           of reference_series at all regular points for which reference_series is decreasing.
           It is designed for use with surface and interface series, where these are both provided,
           as whenever the interface is decreasing in elevation, it is extremely likely that the 
           surface is coincident with the interface.'''
                        
        if reference_series.type != 'regular':
            print('Reference series must be regular')
            return None
            
        new_series = data_series(self.name,'',self.label)
        new_series.type = 'irregular'
        for ddt in self.dates():
            new_series.data_list[ddt] = self.data_list[ddt]

        rdates = reference_series.dates()
        rvalues = list(reference_series.values())

        tiny = -1.e-5
        for (ii,ddt) in enumerate(rdates[1:-1]):
            v_array = rvalues[ii:ii+3]
            gradient = np.polyfit(list(range(0,3)),v_array,1)
            if gradient[0] < tiny:
                new_series.data_list[ddt] = reference_series.data_list[ddt]

        return(new_series)            


    def regularise(self):
        '''Produce a modified version of a given data series, with times of
        observation at regular intervals (midnight daily) using the estimate()
        method'''
        import os
        import datetime as dt
        if self.type == 'regular':
            print('Series is already regular')
            return None
        
        series_name = self.label+'_r'

        if self.type == '':
            print('Series does not appear to have any data')
            return data_series(self.name,'Synthetic',series_name)

        new_series = data_series(self.name,'Synthetic',series_name)
        datafile = new_series.data_file()

        period = self.period()
        numbers = [dt.date(ddt.year,ddt.month,ddt.day).toordinal() for ddt in period]
        for number in range(numbers[0]+1,numbers[1]):
            ddate = dt.date.fromordinal(number)
            ddatetime = dt.datetime(ddate.year,ddate.month,ddate.day,0,0)
            value = self.estimate(ddatetime)
            if value != None:
                new_series.data_list[ddatetime] = value

        new_series.type = 'regular'

        return new_series
                                    

    def regularise_temp(self,tdates):
        '''Produce a modified version of a given data series, with times of
        observation at specified list of datetime points (tdates)'''
        import os
        import datetime as dt
        
        if self.label[-2:] == '_r':
            series_name = self.label[:-2] + '_rt'
        else:
            series_name = self.label+'_rt'

        if self.type == '':
            print('Series does not appear to have any data')
            return data_series(self.name,'Synthetic',series_name)

        new_series = data_series(self.name,'Synthetic',series_name)
        datafile = new_series.data_file()

        period = self.period()
        numbers = [dt.date(ddt.year,ddt.month,ddt.day).toordinal() for ddt in period]
        for ddatetime in tdates:
            value = self.estimate(ddatetime)
            if value != None:
                new_series.data_list[ddatetime] = value
        new_series.type = 'regular_temp'

        return new_series


    def daily_estimate(self,day,month,year):
        import datetime as dt
        regular_logical = (self.label[-2:] == '_r')
        estimate_date = dt.datetime(year,month,day,0,0)
        if regular_logical:
            data_point = self.data_list[estimate_date]
        else:
            data_point = self.estimate(estimate_date)

        return data_point


    def full_month_estimates(self,month,year):
        regular_logical = (self.label[-2:] == '_r')
        
        import datetime as dt
        running_date = dt.datetime(year,month,1,0,0)
        rd_number = running_date.toordinal()
        
        data_points = []
        while running_date.month == month:
            if regular_logical:
                data_point = self.data_list[running_date]
            else:
                data_point = self.estimate(running_date)
            rd_number = rd_number + 1
            running_date = dt.datetime.fromordinal(rd_number)
            
            data_points.append(data_point)
            
        if regular_logical:
            data_point = self.data_list[running_date]
        else:
            data_point = self.estimate(running_date)
        data_points.append(data_point)
        return data_points


    def period_estimates(self,start_date,end_date):
        import datetime as dt
        regular_logical = (self.label[-2:] == '_r')
        start_number = start_date.toordinal()
        end_number = end_date.toordinal()
        sp = self.period()

        data_points = []
        for number in range(start_number,end_number+1):
            ddate = dt.date.fromordinal(number)
            ddatetime = dt.datetime(ddate.year,ddate.month,ddate.day,0,0)

            if ddatetime >= sp[0] and ddatetime <= sp[1]:
                if regular_logical:
                    data_point = self.data_list[ddatetime]
                else:
                    data_point = self.estimate(ddatetime)
                data_points.append(data_point)

        return data_points

    def estimate_decrease(self,month,year):
        import numpy as np
        
     
        month_estimates = self.full_month_estimates(month,year)
        if None in month_estimates:
            print('There was at least one faulty estimate in this month')
            return None

        array_estimates = np.array(month_estimates)
        
        differences = array_estimates - np.roll(array_estimates,1)
        differences.itemset(0,0.)
        
        negative_differences = differences * (differences < 0)
        
        return sum(negative_differences)


    def estimate_increase(self,month,year):
        import numpy as np
        
     
        month_estimates = self.full_month_estimates(month,year)
        if None in month_estimates:
            print('There was at least one faulty estimate in this month')
            return None

        array_estimates = np.array(month_estimates)
        
        differences = array_estimates - np.roll(array_estimates,1)
        differences.itemset(0,0.)
        
        positive_differences = differences * (differences > 0)
        
        return sum(positive_differences)
            

    def estimate_total(self,month,year):
        increase = self.estimate_increase(month,year)
        decrease = self.estimate_decrease(month,year)
        return increase + decrease


    def aggregated_value(self,period,time_function=None,data_function=None):
        import numpy as np
       
        if time_function is None:
            time_function = np.mean

        dt_period = date_int_to_datetime_int(period)
        if self.contains_period(*dt_period):
            values = np.array(self.period_estimates(*period))
            if not data_function is None:
                values = data_function(values)
            daily_values = (values[1:] + values[:-1]) / 2.

            aggregated_value = time_function(daily_values)

            return aggregated_value
        else:
            print('Requested period does not fall within period of buoy '+
                  'operation')
            return None



    def data_file(self):
        import filepaths
        
        datadir = filepaths.filepaths()['series_dir']+self.name+'/'
        datafile = datadir + self.label + '.dat'
        return datafile        


    def file_write(self):
        datafile = self.data_file()

        fileh = open(datafile,'w')
        fileh.write('IMB buoy data series: '+self.label+'\n')
        fileh.write('Read from buoy: '+self.name+'\n')
        fileh.write('Data series type: '+self.type+'\n')
        fileh.write('Derived from file: '+self.fen+'\n')
        fileh.write('\n')
        
        for date in self.dates():
            strdate = date.strftime('%Y/%m/%d %H.%M')
            fileh.write(strdate+':  '+'%12.5f'%self.data_list[date]+'\n')

        fileh.close()


    def file_read(self):
        import datetime as dt
        datafile = self.data_file()

        fileh = open(datafile)
        fileh.readline()
        fileh.readline()
        typeline = fileh.readline()
        fenline = fileh.readline()
        fileh.readline()

        self.type = typeline.split(':')[1].strip()
        self.fen = fenline.split(':')[1].strip()

        keyline = fileh.readline()
        while len(keyline.strip()) > 0:
            date, value = keyline.split(':')

            try:
                year = int(date[0:4])
                month = int(date[5:7])
                day = int(date[8:10])
                hour = int(date[11:13])
                minute = int(date[14:16])
            except ValueError:
                print('Date appears to be in the wrong format')
                return None

            date = dt.datetime(year,month,day,hour,minute)

            try:
                value = float(value)
            except ValueError:
                print('Value does not appear to be a float')
                return None

            self.data_list[date] = value
            keyline = fileh.readline()


    def validate(self,threshold=10,period=1):
        import numpy as np
        import functions
        import scipy.signal

        ddates = self.dates()
        numbers = np.array([functions.datetime_to_float(ddt) for ddt in ddates])
        values = np.array(list(self.values()))
        return_value = {}
        for (ii,ddt) in enumerate(ddates):
            number = numbers[ii]
            interval = [number-period,number+period]
            index = np.where(np.logical_and(numbers>=interval[0],\
                 numbers<=interval[1]))
            lindex = list(index[0])
            lindex.remove(ii)
            rindex = (np.array(lindex),)
    
            detrend_values = scipy.signal.detrend(values[rindex])
            vmean = np.mean(values[rindex])
            vstd = np.std(detrend_values)
            v_anomalousness = np.abs((vmean - values[ii])) / vstd
            return_tuple = (v_anomalousness,v_anomalousness <= threshold,\
                vmean)

            return_value[ddt] = return_tuple

        return return_value


    def rate_of_change(self,period=1):
        import numpy as np
        import datetime as dt
        import functions
        
        seconds_in_day = 24.*3600.
        new_varname = self.label + '_ddt_period_'+str(period)
        new_series = data_series(self.name,self.fen,new_varname)

        dates = self.dates()
        values = np.array(list(self.values()))
        numbers = np.array([functions.datetime_to_float(ddt) for ddt in dates])
        for (number,ddt) in zip(numbers,dates):
            index_period = np.where(np.logical_and(\
                numbers >= number-period,numbers <= number+period))
            numbers_period = numbers[index_period]
            values_period = values[index_period]
            rate_of_change = np.polyfit(numbers_period,values_period,1)[0]/\
                              seconds_in_day
            new_series.data_list[ddt] = rate_of_change

        new_series.type = self.type
        return new_series


    def scatter(self,other,xlim=None,ylim=None,show=True):
        import numpy as np
        import matplotlib.pyplot as plt

        xx = []
        yy = []

        dates1 = self.dates()
        dates2 = other.dates()
        dates_both = list(set(dates1) & set(dates2))
        for ddt in dates_both:
            xx.append(self.data_list[ddt])
            yy.append(other.data_list[ddt])

        axx = np.array(xx)
        ayy = np.array(yy)
                
        plt.plot(axx,ayy,marker='*',linestyle=' ')
        if xlim is not None:
            plt.gca().set_xlim(xlim)
  
        if ylim is not None:
            plt.gca().set_ylim(ylim)

        if show:
            plt.show()


def datefunc1(date_cpts,buoy_name,file_ext_name):
    '''This function calculates the datetime corresponding to a single
       record in an IMB file in the case that the date entry is given as a
       single number. This number represents the number of days since 1st 
       January of the year of deployment of an IMB (which is represented 
       in the IMB name)'''
    import datetime
    buoy_year = int(buoy_name[0:4])
    aux_date = datetime.date(year = buoy_year, month = 1, day = 1)
    aux_date_number = aux_date.toordinal()
    actual_date_number = aux_date_number + int(date_cpts[0]) - 1
    actual_date = datetime.date.fromordinal(actual_date_number)
    estimated_datetime = datetime.datetime(year=actual_date.year,
                                           month=actual_date.month,
                                           day = actual_date.day,
                                           hour = 12,
                                           minute = 0)
    return estimated_datetime


def datefunc3(date_cpts,buoy_name,file_ext_name):
    '''This function calculates the datetime corresponding to a single
       record in an IMB file in the case that the date entry is given as 
       three numbers: year, month and day. Hour and minute are both 
       assumed to be zero.'''
    import datetime
    
    buoy_year = int(date_cpts[2])
    if buoy_year < 1900:
        buoy_year = buoy_year + 2000
        
    if mday_flag(buoy_name,file_ext_name):
       buoy_day = int(date_cpts[0])
       buoy_month = int(date_cpts[1])
    else:
       buoy_day = int(date_cpts[1])
       buoy_month = int(date_cpts[0])
    
    actual_date = datetime.datetime(year = buoy_year, 
                                month = buoy_month,
                                day = buoy_day,
                                hour = 12,
                                minute = 0)
    return actual_date


def datefunc5(date_cpts,buoy_name,file_ext_name):
    '''This function calculates the datetime corresponding to a single
       record in an IMB file in the case that the date entry is given as 
       five numbers: year, month, day, hour and minute.'''
    import datetime
    
    buoy_year = int(date_cpts[2])
    if buoy_year < 1900:
        buoy_year = buoy_year + 2000
        
    if mday_flag(buoy_name,file_ext_name):
       buoy_day = int(date_cpts[0])
       buoy_month = int(date_cpts[1])
    else:
       buoy_day = int(date_cpts[1])
       buoy_month = int(date_cpts[0])
        
    actual_date = datetime.datetime(year = buoy_year, 
                                month = buoy_month,
                                day = buoy_day,
                                hour = int(date_cpts[3]),
                                minute = int(date_cpts[4]))
    return actual_date

def process_dates(date_string,buoy_name,file_ext_name):
    import functions

    date_functions_d = {1:datefunc1,3:datefunc3,5:datefunc5}

    import re
    date_cpts = re.split('[:/ ]',date_string)
    bad_cpts = [cpt for cpt in date_cpts if not functions.is_number(cpt)]
    
    if len(bad_cpts) > 0:
        return None
    else:
    
    
        ncpts = len(date_cpts)

        use_function = date_functions_d[ncpts]

        processed_date = use_function(date_cpts,buoy_name,file_ext_name)
        return processed_date


def mday_flag(buoy_name,file_ext_name):
    import filepaths
    mday_fileh = open(filepaths.filepaths()['mday_file'])
    for line in mday_fileh.readlines():
        if (line[0:8].rstrip()==buoy_name and line[8:len(line)].rstrip()==file_ext_name):
            mday_fileh.close()
            return True
            
    mday_fileh.close()
    return False


def set_special_xaxis(axis,xlr=None,period_plot=None,label=True):
    import numpy as np
    import datetime as dt
    

    start_date = period_plot[0]
    end_date = period_plot[1]
    
    start_date_number = start_date.toordinal()
    end_date_number = end_date.toordinal()
    axis.set_xlim(start_date_number,end_date_number)
    
    ylim = axis.get_ylim()
    if label:
        ytick = [ylim[0],ylim[0]+.01*(ylim[1]-ylim[0])]

        td = period_plot[1] - period_plot[0]
        if xlr:
            rotation = xlr
        else:
            if td.days > 150:
                rotation = 0. - np.arctan((td.days - 150)/100.) * 180. / np.pi
            else:
                rotation = 0.

        write_dates = td.days < 30

        for dn in range(start_date_number,end_date_number):
            date = dt.date.fromordinal(dn)
            axis.plot([dn,dn],ytick,color = '#A0A0A0')

            if (dn==start_date_number) and (date.day > 16):
                axis.text(dn,ylim[0]-.02*(ylim[1]-ylim[0]),date.strftime('%B'),rotation = rotation,va='top',ha='center')

            if (date.day==16):
                axis.text(dn,ylim[0]-.02*(ylim[1]-ylim[0]),date.strftime('%B'),rotation = rotation,va='top',ha='center')

            if (date.day==1):
                axis.plot([dn,dn],ylim,color='#A0A0A0',linewidth = 1 + 2*(date.month==1))

            if write_dates:
                axis.text(dn+.5,ylim[0]-.02*(ylim[1]-ylim[0]),date.strftime('%d'),ha='center')

    axis.set_ylim(ylim)
    axis.set_xticks([])

        
def validated_version(series,threshold=10,period=1):
    validation_key = series.validate(threshold=threshold,\
        period=period)
    new_series = data_series(series.name,series.fen,series.label+'_vv.'+\
        str(threshold)+'.'+str(period))

    for ddt in series.dates():
        if not validation_key[ddt][1]:
            new_series.data_list[ddt] = validation_key[ddt][2]
        else:
            new_series.data_list[ddt] = series.data_list[ddt]

    return new_series

def date_int_to_datetime_int(date_int):
    import datetime as dt
    end_number = date_int[1].toordinal() + 1
    end_date = dt.date.fromordinal(end_number)
    
    start_ddt = dt.datetime(date_int[0].year,date_int[0].month,\
                            date_int[0].day,0,0)
    end_ddt   = dt.datetime(end_date.year,end_date.month,\
                            end_date.day,0,0)

    return [start_ddt,end_ddt]
