import pandas as pd
import sys
import os
# sys.path.append(os.path.split(os.path.dirname(__file__))[0])
# from helpers import helpers



# SITE NAME
mainAppRepo = "/DATA/These/Projects/modflops/docker-simulation/modflow/"
def get_site_name_from_site_number(site_number):
    sites = pd.read_csv(mainAppRepo + 'data/study_sites.txt',
                        sep=',', header=0, index_col=0) #\\s+
    site_name = sites.index._data[site_number]
    return site_name

def get_csv_file_with_indicator_for_a_context(site_number, chronicle, approx, folder):
    indicator = "H"
    site_name = get_site_name_from_site_number(site_number)
    file_name = "Exps_" + indicator + "_Indicator_" + site_name + "_Chronicle"+ str(chronicle) + "_Approx" + str(approx) + "_K_" + str(permeability) + "_BVE_CVHV_Extend.csv"
    indicator_file = folder + "/" + site_name + "/" + file_name
    try:
        dfp = pd.read_csv(indicator_file, sep=",")
    except:
        print("File does not exist")
        dfp = pd.DataFrame()
    
    return dfp

def get_csv_file_with_steady_features_for_a_context(site_number, chronicle, permeability, folder):
    site_name = get_site_name_from_site_number(site_number)
    model_name = "model_time_0_geo_0_thick_1_K_" + str(permeability) + "_Sy_0.1_Step1_site" + str(site_number) + "_Chronicle" + str(chronicle) + "_SteadyState"
    file_name = model_name + "_extracted_features.csv"
    steady_file = folder + "/" + site_name + "/" + model_name + "/" + file_name
    try:
        df = pd.read_csv(steady_file, sep=";")
    except:
        print("steady_file: ", steady_file)
        print("File for site " + site_name + " (number : " + str(site_number) + " & chronicle " + str(chronicle) + ") does not exist")
        df = pd.DataFrame()
    
    return df


folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"
site_number = 1
indicator = "H"
#for site_number in range(1,31):

approx=0
chronicle=0
permeability=27.32

for site_number in range(1,31):
    site_name = get_site_name_from_site_number(site_number)
    df_all_per_site = pd.DataFrame()

    df = get_csv_file_with_indicator_for_a_context(site_number, chronicle, approx, folder)
    if not df.empty:
        taille = len(df.index)
        df = df.drop(df.columns[[0]], axis=1)
        df.rename(columns={'Approximation':'Rate'}, inplace=True)
        df_site = pd.DataFrame(data=[site_number]*taille, index=df.index, columns=['Site_number'])
        df_approx = pd.DataFrame(data=[approx]*taille, index=df.index, columns=['Approx'])
        df_chr = pd.DataFrame(data=[chronicle]*taille, index=df.index, columns=['Chronicle'])
            #steady features 
        df_steady_uni = get_csv_file_with_steady_features_for_a_context(site_number, chronicle, permeability, folder)
        df_steady = pd.concat([df_steady_uni]*taille, ignore_index=True)
        dff = pd.concat([df_site, df_approx, df_chr, df, df_steady], axis=1)        

        df_all_per_site = pd.concat([df_all_per_site, dff])

    output_file_name = "Exps_" + indicator + "_Indicator_" + "Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_" + site_name + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
    print(output_file_name)
    df_all_per_site.to_csv(folder + "/" + site_name + "/" + output_file_name, index=False)
    