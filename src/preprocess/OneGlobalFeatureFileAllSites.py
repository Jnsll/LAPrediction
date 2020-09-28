import pandas as pd
import sys
import os
# sys.path.append(os.path.split(os.path.dirname(__file__))[0])
# from helpers import helpers


mainAppRepo = "/DATA/These/Projects/modflops/docker-simulation/modflow/"
def get_site_name_from_site_number(site_number):
    sites = pd.read_csv(mainAppRepo + 'data/study_sites.txt',
                        sep=',', header=0, index_col=0) #\\s+
    site_name = sites.index._data[site_number]
    return site_name

def get_csv_file_with_indicator_for_a_site(site_number, chronicle, approx, permeability, folder):
    indicator = "H"
    site_name = get_site_name_from_site_number(site_number)
    file_name = "Exps_" + indicator + "_Indicator_" + "Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_" + site_name + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
    indicator_file = folder + "/" + site_name + "/" + file_name
    #dfp = pd.DataFrame()
    try:
        dfp = pd.read_csv(indicator_file, sep=",")
        
    except:
        print("File for site " + site_name + " (number : " + str(site_number) + " does not exist")
        dfp = pd.DataFrame()
        
    #print("dfp", dfp)
    return dfp

def create_csv_file_for_all_sites(chronicle, approx, permeability, folder):
    df_all_sites = pd.DataFrame()

    for site_number in range(1,31):
        site_name = get_site_name_from_site_number(site_number)
        print("site n°", site_number, " : ", site_name)
        df = get_csv_file_with_indicator_for_a_site(site_number, chronicle, approx, permeability, folder)
        #print(df)
        
        df_all_sites = pd.concat([df_all_sites, df])
    

    output_file_name = "Exps_" + indicator + "_Indicator_" + "Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_" + "All_Sites" + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
    print(folder + "/" + output_file_name)
    df_all_sites.to_csv(folder + "/" + output_file_name, index=False)


folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"
indicator = "H"
chronicle=0
approx=0
permeability=27.32
create_csv_file_for_all_sites(chronicle, approx, permeability, folder)