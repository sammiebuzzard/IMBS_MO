''' Defines files and directories used in stage 1 of the IMB calculation'''

def filepaths():

    output = {
        'source_dir' :                 # This should be set to the main 
                                       # directory where the IMB data is held.
                                       # Subdirectories should have names 
                                       # equal to the IMB labels, e.g. '1997D',
                                       # '2012L', with all files for each IMB
                                       # placed under its directory

        'series_dir' :                 # Directory where all IMB data series (raw and derived) can be saved in ASCII form
        'output_dir' :                 # Directory where all IMB data series, regularised to temperature measurements points, can be saved in netCDF format, for reading by second stage of the code
        'mday_file' :                  # File containing a list of all IMBs for which dates are in UK format (most are in US format). Provided with the code.
        'temp_masks_file' :            # File containing a list of blocks of IMB temperature data known to be spurious. Provided with the code
        'translation_file':            # File containing a list of IMB elevation data series thought be in error by a fixed displacement. Provided with the code
        'temp_translations_file':      # File containing a list of IMB temperature data blocks thought to be in error by a fixed displacement. Provided with the code
        'temp_ref_elev_file' :         # File with details of where the top temperature measurement point is located relative to the initial snow-ice interface, for each IMB (if not given, the point is assumed to be 60cm above the interface). Provided with the code
    }
    return output
