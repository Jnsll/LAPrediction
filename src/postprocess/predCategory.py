import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import argparse

# Get data
# Prediction variables
# chronicle = 0
# approx = 0
# permeability = 27.32
# test_site = 2

result_folder = (
    "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/ZLearning"
)


def predCategory(approx, chronicle, permeability, test_site):

    dataset_filename = (
        "Prediction_HError_Basic"
        + "_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_BVE_CVHV_Geomorph_Sat_H_values_Extend.csv"
    )

    df = pd.read_csv(result_folder + "/" + "Approx" + str(approx) + "/" + "Chronicle" + str(chronicle) +"/" + "SiteTest" + str(test_site) + "/" + dataset_filename, sep=",")

    #DATA
    y_pred = df["H Error Predict"]
    y_real = df["H Error Real"]
    x = df["Rate"]
    rates_interp=[]
    rate_ref = x.tolist()
    H_pred = y_pred.tolist()
    H_real = y_real.tolist()


    # Create list of rates for which to interpolate the H ind values
    for i in range(2,len(rate_ref)):    
        rates_interp += [*range(rate_ref[i-1]+1, rate_ref[i])]

    # Interpolation
    tck_pred = interpolate.splrep(x, y_pred, s=0)
    tck_real = interpolate.splrep(x, y_real, s=0)

    ynew_pred = interpolate.splev(rates_interp, tck_pred, der=0)
    ynew_real = interpolate.splev(rates_interp, tck_real, der=0)

    vals_real, vals_pred = create_vals_dicts(rate_ref, rates_interp, ynew_real, ynew_pred, H_real, H_pred)

    graph_name_vals = result_folder + "/" + "Approx" + str(approx) + "/" + "Chronicle" + str(chronicle) +"/" + "SiteTest" + str(test_site) + "/" + "ValuesH_Real_Pred.jpg"
    save_graph_vals(vals_real, vals_pred, graph_name_vals, test_site)

    c = compute_c(vals_real, vals_pred)
    print(len(c))

    per_c_pos = get_positions_c_pos(c)

    graph_name = result_folder + "/" + "Approx" + str(approx) + "/" + "Chronicle" + str(chronicle) +"/" + "SiteTest" + str(test_site) + "/" + "IndicatorC_Localisation_DangerousSituationPrediction.jpg"
    create_figure_c_ind(graph_name, per_c_pos, c, test_site)
    print("Graph saved in " + graph_name)


def save_graph_vals(vals_real, vals_pred, graph_name, test_site):
    plt.plot(*zip(*sorted(vals_real.items())), label="H Ind Real")
    plt.plot(*zip(*sorted(vals_pred.items())), label="H Ind Pred")
    plt.legend()
    plt.xlabel("Rates")
    plt.ylabel("H Ind")
    plt.title("Test Site n°" + str(test_site))
    plt.savefig(graph_name)
    print("Fait: " + graph_name)
    plt.close()


def create_vals_dicts(rate_ref, rates_interp, ynew_real, ynew_pred, H_real, H_pred):
    vals_pred = {}
    vals_real = {}

    for i in range(len(rates_interp)):
        vals_pred[rates_interp[i]] = ynew_pred[i]
        vals_real[rates_interp[i]] = ynew_real[i]
    
    for z in range(len(rate_ref)):
        vals_pred[rate_ref[z]] = H_pred[z]
        vals_real[rate_ref[z]] = H_real[z]

    return vals_real, vals_pred

def compute_c(vals_real, vals_pred):
    num = []
    denom = []
    c = []

    for r in range(1,3653):
        num.append(vals_pred[r]-0.1)
        denom.append(vals_real[r]-0.1)
        #print(num[r-1], denom[r-1])
        try:
            val = num[r-1]/denom[r-1]
            c.append(-val)
        except:
            print("not possible")

    return c

def get_positions_c_pos(c):
    per_c_pos = []
    positive = False
    for i in range(len(c)):
        if c[i] > 0:
            if positive is False:
                per_c_pos.append(i+1)
                positive = True
        
        else:
            if positive:
                per_c_pos.append(i+1)
                positive = False
    return per_c_pos

def create_figure_c_ind(graph_name, per_c_pos, c, test_site):
    plt.rcParams["figure.figsize"] = (20,5)
    plt.plot([*range(1,3653)], c)
    plt.plot([1, 3652], [0, 0], 'k-', lw=2)
    plt.axis([1, 3652, -10, 15])
    plt.xlabel("Rate")
    plt.ylabel("C Ind Value")

    for p in range(0, len(per_c_pos), 2):
        #plt.axvline(x=p, linewidth=1, color='r', label=str(p))
        plt.axvspan(per_c_pos[p], per_c_pos[p+1], alpha=0.5, color='red', label="Rates " + str(per_c_pos[p])+"-"+str(per_c_pos[p+1]))
        if len(per_c_pos)%2 !=0 and p==len(per_c_pos)-1:
            plt.axvspan(per_c_pos[p], 3562, alpha=0.5, color='red', label="Rates " + str(per_c_pos[p])+"-"+str(3652))
    plt.legend()
    plt.title("Test Site n°" + str(test_site))
    plt.savefig(graph_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)

    args = parser.parse_args()

    site_number = args.sitenumber

    predCategory(test_site=site_number, chronicle=0, approx=0, permeability=27.32)