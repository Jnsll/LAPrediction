import pandas as pd 
import argparse

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



if __name__ == "__main__":
    update_data_with_diff_Hvalues()
