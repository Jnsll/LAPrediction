import pandas as pd 

INPUT_DIR = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/ZLearning/"

def get_quality_indicator_for_predictions(approx=0, chronicle=0, permeability=27.32):
    input_data_pmax_pedictions = get_input_data_of_pmax_predictions()
    #print(input_data_pmax_pedictions)
    input_data_pmax_pedictions_cleaned = input_data_pmax_pedictions.dropna()

    score_pred = []
    for row in range(len(input_data_pmax_pedictions_cleaned)):
        if input_data_pmax_pedictions_cleaned.iloc[row, 7] == input_data_pmax_pedictions_cleaned.iloc[row, 6]:
            score_pred.append(5)
        elif input_data_pmax_pedictions_cleaned.iloc[row, 7] > input_data_pmax_pedictions_cleaned.iloc[row, 6]:
            score_pred.append(-10)
        else:
            score_pred.append(1)
    global_quality_score = sum(score_pred) / len(input_data_pmax_pedictions_cleaned)
    print(global_quality_score)



def get_input_data_of_pmax_predictions(approx=0, chronicle=0, permeability=27.32):

    input_data = pd.read_csv(INPUT_DIR + "Prediction_PMax_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv", sep=",")
    return input_data


if __name__ == '__main__':
    get_quality_indicator_for_predictions()