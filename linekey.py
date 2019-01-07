''' A module for defining and producing the linekey object, which is used 
to determine and store information about data format in a CRREL 
ice mass balance buoy.'''

def is_number(string):
    try:
        float(string)
	return True
    except ValueError:
        return False
	
class linekey:
    def __init__(self,date_index = 0):
	self.date_index = date_index
	self.value_index = []
	self.phenomena_names = []
	self.lon_flip_ew = (False,-1,-1)
	self.lat_flip_ns = (False,-1,-1)
        self.vertical_scale = 1.
        self.fliplon = False

    def add_value_index(self,phenomenon,index):
        self.value_index.append(index)
        self.phenomena_names.append(phenomenon)

    def ns(self,index_flippee,index_flipper):
        self.lat_flip_ns = (True,index_flippee,index_flipper)

    def ew(self,index_flippee,index_flipper):
        self.lon_flip_ew = (True,index_flippee,index_flipper)
	
	
def get_temp_linekey(data_file):
    fileh = open(data_file)
    
    import csv
    rows = csv.reader(fileh)
    found_key = False
    found_date = False
    
    for row in rows:

	for (i,strtest) in enumerate(row):
    	    if ('Date' in strtest) or ('DATE' in strtest):
		key = linekey(date_index = i)
		found_date = True			
		break

        if found_date:
	    
	    temp_codes = {}
	    
	    temp_type = ''
	    for (i,strtest) in enumerate(row):
	        result = classify_temp_header(strtest)
		
		if result[0]==1:
		    if temp_type == 'subjective':
		        print 'Unable to determine temperature type'
			return 0
		    temp_type = 'objective'
		    prefix = 'TO'
		    
		if result[0]==2:
		    if temp_type == 'objective':
		        print 'Unable to determine temperature type'
			return 0
		    temp_type = 'subjective'
		    prefix = 'TS'
		    
	        temp_codes[i] = classify_temp_header(strtest)

                if result[0]!=0:
		    key.add_value_index(prefix+str(result[1]),i)
		  
	    break
	    
    	      
    return key
    

def get_linekey(data_file,variable_list,buoy_name):

    fileh = open(data_file)

    import dictionaries
    import csv
    rows = csv.reader(fileh)
    found_key = False
    found_date = False
    
    td = dictionaries.title_dic()
    variable_keys_list = [td[variable_name] for variable_name in variable_list]
    vertical_scale = 1. 
    fliplon = False   

    for row in rows:
	if not found_key:

	    for (i,strtest) in enumerate(row):
    		if ('Date' in strtest) or ('DATE' in strtest):
		    key = linekey(date_index = i)
		    found_date = True			
		    break

	    if found_date:

                for (varno,variable_keys) in enumerate(variable_keys_list):
		    found_key = False
                    for string in variable_keys:
			for (i,strtest) in enumerate(row):
			    if (string == strtest.strip()):
				key.add_value_index(variable_list[varno],i)
				found_key = True
				i_key = i

                                if '(cm)' in string:
                                    vertical_scale = 0.01
                                if '(m)' in string:
                                    vertical_scale = 1.
                                
                                if string=='Longitude (W)':
                                    fliplon = True

                    if not found_key:
		        key.add_value_index(variable_list[varno],-1)

		    if variable_list[varno]=='latitude':
			for (i,strtest) in enumerate(row):
			    if (strtest == 'N/S'):
		        	key.ns(i_key,i)

		    if variable_list[varno]=='longitude':
			for (i,strtest) in enumerate(row):
			    if (strtest == 'E/W'):
		        	key.ew(i_key,i)

        if True in [('units are cm') in item for item in row]:
            vertical_scale = 0.01

        if 'E/W' in row and 'longitude' in key.phenomena_names:
            i_flipper = row.index('E/W')
            i_flippee = key.value_index[key.phenomena_names.index('longitude')]
            key.ew(i_flippee,i_flipper)

        if 'N/S' in row and 'latitude' in key.phenomena_names:
            i_flipper = row.index('N/S')
            i_flippee = key.value_index[key.phenomena_names.index('latitude')]
            key.ew(i_flippee,i_flipper)
            

    if not found_date:
	print 'Could not find date'
	fileh.close()
	return 0	
		
    key.vertical_scale = vertical_scale
    key.fliplon = fliplon
    fileh.close()
    return key

def classify_temp_header(string):
    if is_number(string):
        number = float(string)
        return (1,number)
    
    elif string[0:1]=='T' and string[-3:]=='(C)' and is_number(string[1:-3]):
        number = int(string[1:-3])
        return (2,number)

    elif string[0:1]=='T' and is_number(string[1:]):
        number = int(string[1:])
        return (2,number)

    elif len(string) >= 4:
        if string[0:4]=='TEMP' and is_number(string[4:]):
            number = int(string[4:])
            return (2,number)
        elif string[0:5]=='Temp ' and is_number(string[5:]):
            number = int(string[5:])
            return (2,number) 
	else:
            return (0,0)
    else:
        return (0,0)
