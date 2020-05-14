import pandas as pd
import matplotlib.pyplot as plt
MYDIR= "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + "ZLearning/" 

sites= [1,4,5,6,7,8,10,11,12,17,18,19]
approximations=[0]
chronicles=[0]
#df = pd.DataFrame(columns=["Site", "ExecTimeSum"])
frames=[]
for approx in approximations:
    for chronicle in chronicles:
        for test_site in sites:
            file = MYDIR + "Approx"+ str(approx) + "/Chronicle" + str(chronicle) + "/SiteTest" + str(test_site) + "/" + 'Prediction_HError_Basic.csv'
            dfp = pd.read_csv(file, sep=";")
            frames.append(dfp)
            df=pd.concat(frames)
df.to_csv(MYDIR+"Pred_All.csv")
print("File '" + "Pred_All.csv" +"' has been created.")

#df_sort=df.sort_values(by=['R2 Test Test Site'])
#print(df_sort)
df.plot(kind='scatter', x='Test Site', y='R2 Test')
plt.savefig(MYDIR + '/All_Relation_Site_Deter_coef.png')
plt.clf()