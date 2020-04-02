import pandas as pd
MYDIR= "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/" + "ZLearning/" 

sites= [1,4,5,6,7,8,10,11,12,17,18,19]
approximations=[0]
chronicles=[0]
#df = pd.DataFrame(columns=["Site", "ExecTimeSum"])
frames=[]
for approx in approximations:
    for chronicle in chronicles:
        for test_site in sites:
            file = MYDIR + "Approx"+ str(approx) + "/Chronicle" + str(chronicle) + "/SiteTest" + str(test_site) + "/" + 'Stats.csv'
            dfp = pd.read_csv(file, sep=";")
            frames.append(dfp)
            df=pd.concat(frames)
df.to_csv(MYDIR+"Stats_All.csv")
print("File '" + "Stats_All.csv" +"' has been created.")