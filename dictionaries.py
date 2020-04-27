'''Two dictionaries which are used to aid reading of CRREL ice mass balance
buoy source files:
  - file_dic: filenames which may be associated with particular variables
  - title_dic: CSV file titles which may be associated with particular variables
'''


def file_dic():
    output = {
            'air temperature' : ['Met' ,'L2_therm','L3','clean'],
            'air pressure'    : ['Met' ,'L2_therm','L3','clean'],
            'surface'         : ['Mass','L2_therm','L3','clean'],
            'bottom'          : ['Mass','L2_therm','L3','clean'],
            'interface'       : ['Mass','L2_therm','L3','clean'],
            'snow depth'      : ['Mass','L2_therm','L3','clean'],
            'ice thickness'   : ['Mass','L2_therm','L3','clean'],
            'latitude'        : ['Pos','pos','L3','clean'],
            'longitude'       : ['Pos','pos','L3','clean'],
            'temp'            : ['Temp','L2_therm','temp','L3','clean'] 
        }   
    return output

def title_dic():
    output = {
            'air temperature' : ['Air Temp','Air Temp (C)'],
            'air pressure'    : ['Air Pressure','Air Pressure (mb)'],
            'snow depth'      : ['Snow Depth','Snow Depth (m)','Snow depth (cm)'],
            'surface'         : ['Surface','Snow Surface Position (m)',
                                 'Surface position / Cumulative surface melt',
                                 'SNOW','Top'],
            'bottom'          : ['Bottom','Bottom of Ice Position',
                                 'Ice Bottom Position(m)',
                                 'Bottom position (cm)',
                                 'UNDERICE',
                                 'Under Ice Distance'],
            'interface'       : ['Top of Ice Position',
                                 'Ice Surface Position (m)'],
            'ice thickness'   : ['Ice Thickness','Ice Thickness(m)'],
            'latitude'        : ['Latitude','Latitude (N)',
                                 'Latitude (degrees)','LATITUDE'],
            'longitude'       : ['Longitude','Longitude (W)','Longitude (E)',
                                 'Longitude (degrees)','LONGITUDE']
        }
    return output
