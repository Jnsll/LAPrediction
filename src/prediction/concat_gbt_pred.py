import pandas as pd
import matplotlib.pyplot as plt

MYDIR = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/ZLearning/"

sites = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
    30,
]
approximations = [0]
chronicles = [0]
permeability=27.32
# df = pd.DataFrame(columns=["Site", "ExecTimeSum"])
frames = []
for approx in approximations:
    for chronicle in chronicles:
        for test_site in sites:
            file = (
                MYDIR
                + "Approx"
                + str(approx)
                + "/Chronicle"
                + str(chronicle)
                + "/SiteTest"
                + str(test_site)
                + "/"
                + "Prediction_HError_GBT_Chronicle"
                + str(chronicle)
                + "_Approx"
                + str(approx)
                + "_K"
                + str(permeability)
                + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
            )
            try:
                dfp = pd.read_csv(file, sep=";")
            except:
                continue
            frames.append(dfp)
            df = pd.concat(frames)
df.to_csv(MYDIR + "Pred_GBT_Chronicle" + str(chronicle) + "_Approx"
                + str(approx)
                + "_K"
                + str(permeability) + "_AllSites__BVE_CVHV_Geomorph_Sat_Extend.csv", index=False)
print(
    "File '"
    + "Pred_GBT_Chronicle" 
    + str(chronicle) 
    + "_Approx"
    + str(approx)
    + "_K"
    + str(permeability) + "_AllSites__BVE_CVHV_Geomorph_Sat_Extend.csv"
    + "' has been created."
)

# df_sort=df.sort_values(by=['R2 Test Test Site'])
# print(df_sort)
df.plot(kind="scatter", x="Test Site", y="R2 Test")
plt.savefig(
    MYDIR + "/All_Relation_GBTPred_Site_Deter_coef_Chronicle" + str(chronicle) 
    + "_Approx"
    + str(approx)
    + "_K"
    + str(permeability) 
    + "_BVE_CVHV_Geomorph_Sat_Extend.png"
)
plt.clf()
