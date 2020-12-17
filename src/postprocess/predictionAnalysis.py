import pandas as pd 
import argparse
import numpy as np

INPUT_DIR = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/ZLearning/"

def get_input_data_of_Hvalues_predictions_with_subcatch(approx=0, chronicle=0, permeability=27.32):

    input_data = pd.read_csv(INPUT_DIR + "Prediction_HErrorValues_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv", sep=",")
    return input_data

def clean_data(input_data):

    input_data_cleaned = input_data.dropna()
    return input_data_cleaned


def get_discrepancy_between_Hvalues_real_and_predicted(cleaned_data):
    cleaned_data["DiffHreal-Hpred"] = -9999
    for row in range(len(cleaned_data)):
        cleaned_data.iloc[row, 7] = cleaned_data.iloc[row, 5] - cleaned_data.iloc[row, 6]
    return cleaned_data

def save_updated_data_with_diff_Hvalues(data_with_diff_H_values, approx=0, chronicle=0, permeability=27.32):
    data_with_diff_H_values.to_csv(INPUT_DIR + "Prediction_HErrorValues_DiscrepancyRealPred_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv", sep=";")
    print("File created: " + INPUT_DIR + "Prediction_HErrorValues_DiscrepancyRealPred_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv")

def update_data_with_diff_Hvalues():
    input_data = get_input_data_of_Hvalues_predictions_with_subcatch()
    cleaned_data = clean_data(input_data)
    updated_data = get_discrepancy_between_Hvalues_real_and_predicted(cleaned_data)
    save_updated_data_with_diff_Hvalues(updated_data)


def get_HValue_discrepancy_for_pmax(approx=0, chronicle=0, permeability=27.32):
    input_data_pmax_pedictions = get_input_data_of_pmax_predictions()
    input_data_pmax_pedictions_cleaned = input_data_pmax_pedictions.dropna()
    Hvalues_data = get_input_data_of_Hvalues_predictions_with_subcatch()
    input_data_pmax_pedictions_cleaned['H Discrepancy Pmax Real-Pred'] = np.nan
    for row in range(len(input_data_pmax_pedictions_cleaned)):
        pmax_pred = input_data_pmax_pedictions_cleaned.iloc[row, 7]
        pmax_real = input_data_pmax_pedictions_cleaned.iloc[row, 6]
        site = input_data_pmax_pedictions_cleaned.iloc[row, 2]
        subcatch = input_data_pmax_pedictions_cleaned.iloc[row, 3]

        H_real_pmax_real = Hvalues_data.loc[(Hvalues_data['Test Site']== site) & (Hvalues_data['SubCatchment']== subcatch) & (Hvalues_data['Rate']== pmax_real)]['H Error Real']
        H_real_pmax_pred = Hvalues_data.loc[(Hvalues_data['Test Site']== site) & (Hvalues_data['SubCatchment']== subcatch) & (Hvalues_data['Rate']== pmax_pred)]['H Error Real']
        #print(site, subcatch, pmax_real)
        if H_real_pmax_real.empty is False and H_real_pmax_pred.empty is False:
            index_df = input_data_pmax_pedictions_cleaned.loc[(input_data_pmax_pedictions_cleaned['Test Site']== site) & (input_data_pmax_pedictions_cleaned['SubCatchment']== subcatch)].index.values
            #print(index_df)
            #print(input_data_pmax_pedictions_cleaned.at[index_df[0], 'H Discrepancy Pmax Real-Pred'])
            input_data_pmax_pedictions_cleaned.at[index_df[0], 'H Discrepancy Pmax Real-Pred'] = float(H_real_pmax_real.iloc[0] - H_real_pmax_pred.iloc[0])
            #print(input_data_pmax_pedictions_cleaned.loc[(input_data_pmax_pedictions_cleaned['Test Site']== site) & (input_data_pmax_pedictions_cleaned['SubCatchment']== subcatch)]['H Discrepancy Pmax Real-Pred'])
    #print(input_data_pmax_pedictions_cleaned['H Discrepancy Pmax Real-Pred'])
    input_data_pmax_pedictions_cleaned.to_csv(INPUT_DIR + "Prediction_PMax_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV_HDiscrepancy_PmaxReal-Pred.csv", sep=";")




def get_input_data_of_pmax_predictions(approx=0, chronicle=0, permeability=27.32):
    input_data = pd.read_csv(INPUT_DIR + "Prediction_PMax_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv", sep=",")
    return input_data



if __name__ == "__main__":
    # update_data_with_diff_Hvalues()
    get_HValue_discrepancy_for_pmax()
