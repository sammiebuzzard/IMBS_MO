'''Two dictionaries which are used to aid reading of CRREL ice mass balance
buoy source files:
  - file_dic: filenames which may be associated with particular variables
  - title_dic: CSV file titles which may be associated with particular variables
'''


def file_dic():
    output = {}
    output['air temperature'] = ['Met' ,'L2_therm','L3','clean']
    output['air pressure']    = ['Met' ,'L2_therm','L3','clean']
    output['surface']         = ['Mass','L2_therm','L3','clean']
    output['bottom']          = ['Mass','L2_therm','L3','clean']
    output['interface']       = ['Mass','L2_therm','L3','clean']
    output['snow depth']      = ['Mass','L2_therm','L3','clean']
    output['ice thickness']   = ['Mass','L2_therm','L3','clean']
    output['latitude']        = ['Pos','pos','L3','clean']
    output['longitude']       = ['Pos','pos','L3','clean']
    output['temp']            = ['Temp','L2_therm','temp','L3','clean']    
    return output

def title_dic():
    output = {}
    output['air temperature'] = ['Air Temp','Air Temp (C)']
    output['air pressure']    = ['Air Pressure','Air Pressure (mb)']
    output['snow depth']      = ['Snow Depth','Snow Depth (m)','Snow depth (cm)']
    output['surface']         = ['Surface','Snow Surface Position (m)','Surface position / Cumulative surface melt','SNOW','Top']
    output['bottom']          =  ['Bottom','Bottom of Ice Position',\
    'Ice Bottom Position(m)','Bottom position (cm)','UNDERICE','Under Ice Distance']
    output['interface']       = ['Top of Ice Position',\
    'Ice Surface Position (m)']
    output['ice thickness']   = ['Ice Thickness','Ice Thickness(m)']
    output['latitude']        = ['Latitude','Latitude (N)','Latitude (degrees)','LATITUDE']
    output['longitude']       = ['Longitude','Longitude (W)','Longitude (E)','Longitude (degrees)','LONGITUDE']
    return output
