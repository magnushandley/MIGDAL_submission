import glob

"""
This contains a series of scenarios to select the correct filepath for the simulated tracks. If not being run locally these will need redoing.
The binning is a parameter used when plotting an event in 3D, such that for high deposition events it is more clear what is beign viewed. If in doubt leave as 1.
"""
def file_select(case):
    if case == 'migdal':
        TIFF_files = sorted(glob.glob("/Users/magnus/Downloads/TIFF_250_keV_NR_10_keV_ER 2/*_gem_out.tiff"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Downloads/250_keV_NR_10_keV_ER_ITO/raw/*"))
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Downloads/250_keV_NR_10_keV_ER_ITO/conv/*"))
        degrad_files = sorted(glob.glob("/Users/magnus/Downloads/250_keV_NR_10_keV_ER(1)/*"))
        binning = 1

    elif case == 'fe55':
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*.tiff"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*ITO_raw.txt"))
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*ITO_conv_ml.txt"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectron_5.25keV-inputs/*"))
        binning = 1

    elif case == 'artificial1':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6kev_uniform_ITO_TIFF/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6kev_uniform_ITO_TIFF/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6kev_uniform_ITO_TIFF/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_uniform_inputs/*"))
        binning = 1

    elif case == 'artificial1_new':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicalrange_line_outputs/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicalrange_line_outputs/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicalrange_line_outputs/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicialrange_line_inputs/*"))
        binning = 1

    elif case == 'artificial2':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250kev_uniform_ITO_tiff/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250kev_uniform_ITO_tiff/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250kev_uniform_ITO_tiff/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250keV_uniform_inputs/*"))
        binning = 20

    elif case == 'artificial3':
        convl_ITO_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/outputs/offset_v/v_shaped_inclined_third_kink_gem_out_ITO_conv_ml.txt"]
        raw_ITO_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/outputs/offset_v/v_shaped_inclined_third_kink_gem_out_ITO_raw.txt"]
        TIFF_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/outputs/offset_v/v_shaped_inclined_third_kink_gem_out.tiff"]
        degrad_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/inputs/v_shaped_inclined_third_kink.txt"]
        binning = 1

    elif case == 'artificial_inclined':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/outputs/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/outputs/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/outputs/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/inputs/*"))
        binning = 1

    elif case == 'curved_vertical':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_vertical_outputs/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_vertical_outputs/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_vertical_outputs/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_vertical_inputs/*"))
        binning = 1

    elif case == 'curved_flat':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_flat_outputs/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_flat_outputs/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_flat_outputs/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_flat_inputs/*"))
        binning = 1

    elif case == 'curved_inclined':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_inclined_outputs/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_inclined_outputs/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_inclined_outputs/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/curved_test/curved_inclined_inputs/*"))
        binning = 1

    elif case == 'artificial_migdal':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/artificial_migdal/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/artificial_migdal/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/artificial_migdal/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/artificial_migdal_inputs/*"))
        binning = 1

    elif case == 'z_separation':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_test/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_test/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_test/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_test_inputs/*.txt"))
        binning = 1

    elif case == 'y_separation':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/y_separation_test/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/y_separation_test/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/y_separation_test/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/y_separation_test_inputs/*.txt"))
        binning = 1

    elif case == 'z_sep_high_dep':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_high_dep/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_high_dep/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_high_dep/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_high_dep_inputs/*.txt"))
        binning = 1

    elif case == 'u_shaped':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_y_separations/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_y_separations/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_y_separations/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_y_separations_inputs/*.txt"))
        binning = 1

    elif case == 'u_shaped_high_dep':
        convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_high_diff_outputs/*ITO_conv_ml.txt"))
        raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_high_diff_outputs/*ITO_raw.txt"))
        TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_high_diff_outputs/*.tiff"))
        degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_high_diff_inputs/*.txt"))
        binning = 1
        
    return(TIFF_files,raw_ITO_files,convl_ITO_files,degrad_files,binning)
