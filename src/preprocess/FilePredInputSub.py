import os
import numpy as np
import pandas as pd
import argparse
import csv


mainAppRepo = "/DATA/These/Projects/modflops/docker-simulation/modflow/"

def get_model_name(site_number, chronicle, approx, rate, ref, steady, permeability=86.4):
    model_name = "model_time_0_geo_0_thick_1_K_"+ str(permeability) + "_Sy_0.1_Step1_site" + str(site_number) + "_Chronicle" + str(chronicle)
    if steady:
        model_name += "_SteadyState"
    elif not ref:
        model_name += "_Approx" + str(approx)
        if approx == 0 or approx == 2:
            model_name += "_Period" + str(rate)
        elif approx==1:
            model_name += "_RechThreshold" + str(rate)
    return model_name

def get_site_name_from_site_number(site_number):
    sites = pd.read_csv(mainAppRepo + 'data/study_sites.txt',
                        sep=',', header=0, index_col=0) #\\s+
    site_name = sites.index._data[site_number]
    return site_name



def create_global_input_file_for_prediction_with_subcath(chronicle=0, approx=0, permeability=27.32):
    approximations = [1.0, 2.0, 7.0, 15.0, 21.0, 30.0, 45.0, 50.0, 60.0, 75.0, 90.0, 100.0, 125.0, 150.0, 182.0, 200.0, 250.0, 300.0, 330.0, 365.0, 550.0, 640.0, 730.0, 1000.0, 1500.0, 2000.0, 2250.0, 3000.0, 3182.0, 3652.0]
    HIndGlob = pd.DataFrame(columns=['Site', 'SubCatch', 'HError', 'Rate'])
    for site in range(1, 41):
        site_name = get_site_name_from_site_number(site)
        ref_name = get_model_name(site, chronicle, None, None, ref=True, steady=False, permeability=permeability)
        print(site_name)
        crits = pd.read_csv("/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + "Geomorph_Features_All_Sites_Saturation_SubCatch.csv", sep=",")
        
        for appr in range(len(approximations)):
            if appr == 0:
                simu_name = get_model_name(site, chronicle, approx, approximations[appr], ref=True, steady=False, permeability=permeability)
            else:
                simu_name = get_model_name(site, chronicle, approx, approximations[appr], ref=False, steady=False, permeability=permeability)
            filename = simu_name + "_Ref_" + ref_name + "_errorsresult_H_BVE_SUB.csv"
            try:
                with open("/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/Igrida" + "/" + filename, 'r') as f: #"/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + str(site_name) + "/" + simu_name
                    HindSubs = pd.read_csv(f, sep=";", encoding='utf_8')
            except:
                print("Pb file: ", "Site: ", site,"Rate:", approximations[appr])
                continue
            HindSubs.columns = ['SubCatch', 'HError']
            HindSubs["Site"] = [site]*len(HindSubs.index)
            HindSubs["Rate"] = [approximations[appr]]*len(HindSubs.index)
            HIndGlob = pd.concat([HIndGlob, HindSubs], sort=False)
    glob = pd.merge(crits, HIndGlob, how="left", on=["Site", "SubCatch"])
    glob.to_csv("/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + "DataInputPred_SUB.csv", index=False)
    print("File: ", "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + "DataInputPred_SUB.csv", "created!")



if __name__ == "__main__":
    create_global_input_file_for_prediction_with_subcath()