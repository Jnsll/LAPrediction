from sklearn import preprocessing
import pandas as pd
import prediction_of_H_indicator_with_subCatchmentData as prediction
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_input_crits_for_list_of_selected_crits(crits_all, list_crits):
    return crits_all[list_crits]

def get_k_means_for_a_nb_of_cluster(nb_cluster, crits):
    kmeans = KMeans(n_clusters = nb_cluster, init= 'k-means++')  #, n_init = 100, random_state = 99, algorithm="full"
    clusters = kmeans.fit_predict(crits)
    #print(clusters)
    taille = 444
    count_in_cluster = {}
    for cluster in range(nb_cluster):
        count_in_cluster[cluster] = list(clusters).count(cluster)
        print("Number of sub-catchment areas in cluster n°", str(cluster), ":", count_in_cluster[cluster], "=> ", (count_in_cluster[cluster]/taille)*100, "%")
    return clusters, count_in_cluster, kmeans

def get_repartition_category_for_clusters(merged_cut, nb_clusters):
    rep_category_by_cluster = {}
    rep_category_by_cluster_percentage = {}
    for cluster in range(nb_clusters):
        merged_cluster = merged_cut[merged_cut["clusters"]==cluster]
        rep_category_by_cluster[cluster] = {'P':0, 'G':0, 'D':0}
        rep_category_by_cluster_percentage[cluster] = {'P':0, 'G':0, 'D':0}
        for index in range(len(merged_cluster)):
            rep_category_by_cluster[cluster][merged_cluster.iloc[index, 2]] +=1
        sum_nb_site_in_cluster = sum(rep_category_by_cluster[cluster].values())
        for category in rep_category_by_cluster[cluster]:
            if sum_nb_site_in_cluster != 0:
                rep_category_by_cluster_percentage[cluster][category] = '{:.1%}'.format(float(rep_category_by_cluster[cluster][category] / sum_nb_site_in_cluster))
        
            
    #print(rep_category_by_cluster)
    print(rep_category_by_cluster_percentage)

def clustering(nb_clusters):
    input_data = prediction.import_input_data()
    # Remove redundant data!!!
    crits_all = input_data.iloc[:,:9]
    crits_all.drop_duplicates(inplace=True)
    list_crits = ['Slope', 'Elevation', 'LC', 'SAR', 'Area', 'CV', 'HV'] #'Slope', 'Elevation', 'LC', 'SAR', 'Area', 'CV', 'HV'
    crits = crits_all[list_crits]

    scaler = preprocessing.StandardScaler().fit(crits.values)
    X_scaled = scaler.transform(crits.values)
    crits_std = pd.DataFrame(X_scaled,columns=["Slope","Elevation", "LC","SAR","Area","CV","HV"])
    
    print("Without Standardization")
    clusters, count_in_cluster, kmeans = get_k_means_for_a_nb_of_cluster(nb_clusters, crits)
    print("-----------------")
    print("With Standardization")
    clusters_std, count_in_cluster_std, kmeans_std = get_k_means_for_a_nb_of_cluster(nb_clusters, crits_std)
    
    crits_all_std = crits_all.copy()
    crits_all["clusters"] = clusters
    crits_all_std["clusters"] = clusters_std
    crits_all.to_csv("output/" + "Clusters_Sites_SubCatch" + str(nb_clusters) + ".csv", index=False)
    crits_all_std.to_csv("output/" + "Clusters_Sites_SubCatch" + str(nb_clusters) + "_std" + ".csv", index=False)
    
    category = pd.read_csv("data/" + "Prediction_PMax_SubCatch_Chronicle0_Approx0_K27.32_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV_Category_meeting.csv")
    category = category.rename(columns = {'Test Site':'Site'})
    category = category.rename(columns = {'SubCatchment':'SubCatch'})
    
    merged = category.merge(crits_all)
    merged_std = category.merge(crits_all_std)
    merged_cut = merged[['Site', 'SubCatch', 'Category', 'clusters']]
    merged_cut_std = merged_std[['Site', 'SubCatch', 'Category', 'clusters']]
    merged.to_csv("output/" + "clusters_categories" + str(nb_clusters) + ".csv", index=False)
    merged_cut.to_csv("output/" + "clusters_categories" + str(nb_clusters) + "_cut.csv", index=False)
    merged_std.to_csv("output/" + "clusters_categories" + str(nb_clusters) + "_std.csv", index=False)
    merged_cut_std.to_csv("output/" + "clusters_categories" + str(nb_clusters) + "_cut_std.csv", index=False)
    
    get_repartition_category_for_clusters(merged_cut, nb_clusters)
    print("--------------------------")
    get_repartition_category_for_clusters(merged_cut_std, nb_clusters)