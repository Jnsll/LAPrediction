import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import argparse
import csv

folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"


def prep_data_for_pred(test_site, chronicle=0, approx=0, permeability=27.32):
    df_features_t = pd.read_csv(
        folder
        + "/"
        + "Features_Prediction_Pmax_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_BVE_all_sites.csv"
    )

    Y_t = df_features_t.filter(["Site", "Pmax"], axis=1)
    X_t = df_features_t.drop(["Pmax"], axis=1)

    y_test = Y_t[Y_t.Site == test_site]
    del y_test["Site"]

    y_train = Y_t.drop(Y_t[Y_t.Site == test_site].index)
    del y_train["Site"]
    y_train

    X_test = X_t[X_t.Site == test_site]
    del X_test["Site"]

    X_train = X_t.drop(X_t[X_t.Site == test_site].index)
    del X_train["Site"]

    x_t = X_train.iloc[:, :-4].values
    y_t = y_train.iloc[:, :].values

    return x_t, y_t, X_test, y_test


def pred_dtclass(x_t, y_t):

    clf = tree.DecisionTreeClassifier(min_samples_leaf=2)
    clf = clf.fit(x_t, y_t)

    feats = [
        "Slope",
        "Elevation",
        "LC",
        "CW",
        "Area",
        "Coastal Vulnerability",
        "Hydrological Vulnerability",
    ]

    plt.figure()
    tree.plot_tree(
        clf, filled=True, feature_names=feats, class_names=["30", "90", "182", "365"]
    )
    plt.savefig(
        "../../outputs/DT_tree_min_leaf2_Features_CVHV_Geomorph_testSite"
        + str(site_test)
        + ".pdf",
        format="pdf",
        bbox_inches="tight",
    )

    return clf, tree


def check_if_correct_predict(test_site, clf, X_test, y_test, permeability):
    sample_one_pred = int(
        clf.predict(
            [
                [
                    X_test.iloc[0, 0],
                    X_test.iloc[0, 1],
                    X_test.iloc[0, 2],
                    X_test.iloc[0, 3],
                    X_test.iloc[0, 4],
                    X_test.iloc[0, 5],
                    X_test.iloc[0, 6],
                ]
            ]
        )
    )
    if sample_one_pred == y_test.iloc[0, 0]:
        print(
            "Correct prediction for test site number",
            test_site,
            "with real pmax =",
            y_test.iloc[0, 0],
            "and predicted pmax is",
            sample_one_pred,
        )
    else:
        print(
            "Incorrect prediction for test site number",
            test_site,
            "with real pmax =",
            y_test.iloc[0, 0],
            "and predicted pmax is",
            sample_one_pred,
        )

    with open(
        folder
        + "/ZLearning/"
        + "Approx"
        + str(approx)
        + "/"
        + "Chronicle"
        + str(chronicle)
        + "/"
        + "SiteTest"
        + str(site_test)
        + "/"
        + "Prediction_pmax_DT_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_BVE_CVHV_Geomorph.csv",
        "w",
    ) as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "Approx",
                "Chronicle",
                "Test Site",
                "P Real",
                "P pred",
            ]
        )
        writer.writerow(
            [
                approx,
                chronicle,
                test_site,
                y_test.iloc[0, 0],
                sample_one_pred,
            ]
        )


def concat_pred_pmax(chronicle, approx, permeability):
    MYDIR = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/ZLearning/"

    sites = range(1, 31)
    frames = []
    for test_site in sites:
        file = (
            MYDIR
            + "Approx"
            + str(approx)
            + "/"
            + "Chronicle"
            + str(chronicle)
            + "/"
            + "SiteTest"
            + str(test_site)
            + "/"
            + "Prediction_pmax_DT_Chronicle"
            + str(chronicle)
            + "_Approx"
            + str(approx)
            + "_K"
            + str(permeability)
            + "_BVE_CVHV_Geomorph.csv"
        )
        try:
            dfp = pd.read_csv(file, sep=";")
        except:
            print("File ", file, "not found!")
            continue
        frames.append(dfp)
        df = pd.concat(frames)
    df.to_csv(
        MYDIR
        + "Pred_Pmax_DT_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_AllSites_BVE_CVHV_Geomorph.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--testsite", type=int, required=False)
    parser.add_argument("-chr", "--chronicle", type=int, required=False)
    parser.add_argument("-approx", "-approx", type=int, required=False)
    parser.add_argument("-perm", "--permeability", type=float, required=False)
    parser.add_argument("-concat", "--concat", action="store_true")

    args = parser.parse_args()

    site_test = args.testsite
    chronicle = args.chronicle
    approx = args.approx
    perm = args.permeability
    concat = args.concat

    if concat:
        concat_pred_pmax(chronicle, approx, perm)
    else:
        if chronicle is None and approx is None and perm is None:
            x_t, y_t, X_test, y_test = prep_data_for_pred(site_test)
        else:
            x_t, y_t, X_test, y_test = prep_data_for_pred(
                site_test, chronicle, approx, perm
            )
        clf, tree = pred_dtclass(x_t, y_t)
        check_if_correct_predict(site_test, clf, X_test, y_test, perm)
