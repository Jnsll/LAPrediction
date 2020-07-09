import pandas as pd
import matplotlib.pyplot as plt
MYDIR= "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + "ZLearning/" 

sites= [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16, 17,18,20, 21, 22, 23, 24, 25, 26, 27, 28,29,30]
approximations=[0]
chronicles=[0]
#df = pd.DataFrame(columns=["Site", "ExecTimeSum"])
frames=[]
for approx in approximations:
    for chronicle in chronicles:
        for test_site in sites:
            file = MYDIR + "Approx"+ str(approx) + "/Chronicle" + str(chronicle) + "/SiteTest" + str(test_site) + "/" + 'Prediction_HError_Basic_Chronicle' + str(chronicle) + '_BVE_CVHV.csv'
            dfp = pd.read_csv(file, sep=";")
            frames.append(dfp)
            df=pd.concat(frames)
df.to_csv(MYDIR+"Pred_Chronicle"+ str(chronicle) + "All_BVE_CVHV.csv", index=False)
print("File '" + "Pred_Chronicle"+ str(chronicle) + "All_BVE_CVHV.csv" +"' has been created.")

#df_sort=df.sort_values(by=['R2 Test Test Site'])
#print(df_sort)
df.plot(kind='scatter', x='Test Site', y='R2 Test')
plt.savefig(MYDIR + '/All_Relation_Site_Deter_coef_Chronicle'+ str(chronicle) + '_BVE_CVHV.png')
plt.clf()