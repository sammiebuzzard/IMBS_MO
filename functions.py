'''Defines various auxiliary functions which are used in CRREL ice mass balance
buoy data processing'''

#!/usr/local/sci/bin/python2.7
import numpy as np

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def erf_average(datetimes,values,central_datetime,sigma):

    cutoff = 4. * sigma

    timedeltas = [test_datetime - central_datetime for test_datetime in
                  datetimes]
    differences = [td.days * 86400. + td.seconds for td in timedeltas]

    datetimes_use   = [dt for (i,dt) in enumerate(datetimes) if abs(differences[i])<=cutoff]
    values_use      = [value for (i,value) in enumerate(values) if abs(differences[i])<=cutoff]
    differences_use = [difference for difference in differences if abs(difference)<=cutoff]

    if (len(datetimes_use))==0:
        return None
    else:
        np_differences = np.array(differences_use)
        np_values = np.array(values_use)
        weightings = np.exp(-np_differences**2./(2.*sigma**2.)) 
        wvalues = weightings*np_values

        average_value = np.sum(weightings*np_values) / np.sum(weightings)
        return average_value


def datetime_to_float(datetime):
    import datetime as dt
    ddate = dt.date(datetime.year,datetime.month,datetime.day)
    fractional_part = datetime.hour / 24.  +  datetime.minute / (24.*60.)
    integer_part = ddate.toordinal()

    return integer_part + fractional_part


def float_to_datetime(number):
    import numpy as np
    import datetime as dt

    tiny = 1.e-5

    integer_part = np.int(np.floor(number))
    fractional_part = number - integer_part
    
    ddate = dt.date.fromordinal(integer_part)
    hours = np.int(np.floor(fractional_part * 24. + tiny))
    minutes = np.int(np.floor((fractional_part*24. + tiny - hours) * 60. + tiny))

    ddatetime = dt.datetime(ddate.year,ddate.month,ddate.day,hours,minutes)
    return ddatetime


def seconds_in_month(month,year):
    import datetime as dt
    seconds_in_day = 24. * 3600.

    date_begin = dt.date(year,month,1)
    day_counter = date_begin.toordinal()
    date_running = date_begin

    while date_running.month == date_begin.month:
        day_counter = day_counter + 1
        date_running = dt.date.fromordinal(day_counter)

    days_in_month = day_counter - date_begin.toordinal()
        
    return days_in_month * seconds_in_day


def buoynumber(input_value):
    import buoys

    bl = buoys.buoylist()

    try:
        return bl[input_value]
    except (TypeError,IndexError) as e:
        try:
            return bl.index(input_value.strip())
        except ValueError:
            print('Invalid buoy number/identifier')
            return None


def buoylist():
    import filepaths
    buoylist_file = filepaths.filepaths()['buoy_list']
    output_value = []
    fileh =open(buoylist_file)
    for line in fileh.readlines():
        buoy_name = line.strip()
        output_value.append(buoy_name)

    return output_value
