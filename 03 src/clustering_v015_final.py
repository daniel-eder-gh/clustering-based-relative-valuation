# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:27:33 2023

@author: Dani
"""

# Importing libraries 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import kneed
from kneed import KneeLocator
import random
import warnings
import time

warnings.filterwarnings('ignore') 


# METHOD TO IMPORT DATA AND SPLIT IT
def input(g_or_v, path):
    #Importing data
    dataset = pd.read_csv(path)
    
    if g_or_v == "Growth":
        dataset = dataset[dataset.loc[:,"Universe"] == "Growth"]
    elif g_or_v == "Value":
        dataset = dataset[dataset.loc[:,"Universe"] == "Value"]
    else:
        dataset = dataset
    
    #duplicating industry group col to keep as original
    dataset["Industry_Group_orig"] = dataset["Industry Group"]
        
    #for testing purposes
    #dataset = dataset.iloc[:1000,:]
    #dataset = dataset[dataset["exticker"].isin(list(dataset["exticker"].unique())[:500])]
    
    for column in dataset.columns:
        dataset = dataset.dropna(subset=[column], how='all', axis=0, inplace=False)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.dropna(subset=[column], axis=0, inplace=False)
        

    # Split the data into several dataframes for each corresponding FQ
    quarters = dataset["Quarter"].unique()
    dataframe_all_quarters = {}
    
    ### all quarters!
    
    for quarter in quarters:
        dataframe_one_quarter = dataset[dataset['Quarter'] == quarter]
        dataframe_all_quarters[quarter] = dataframe_one_quarter
        

    # Encoding the data
    label_encoder = LabelEncoder()

    # Columns to encode
    columns_to_encode = ['HQ', 'Primary Industry', 'Industry Sector', 'Industry Group', 'Industry']
    
    encoded_dataframe_all_quarters = {}
    # Encoding every df
    for quarter, df in dataframe_all_quarters.items():
        # Apply label encoding to the columns
        quarter_copy = df.copy()
        for column in columns_to_encode:
            # quarter_copy.loc[:, column] = label_encoder.fit_transform(quarter_copy[column])       (older python version)
            quarter_copy[column] = label_encoder.fit_transform(quarter_copy[column])
            
        # Updating new dictionary with encoded dataframe
        encoded_dataframe_all_quarters[quarter] = quarter_copy

    return encoded_dataframe_all_quarters

def first_clustering(encoded_dataframe_all_quarters):
    # Select the columns to use for clustering
    #   'GV_Key', 'Date_cal', 'Ticker Symbol', 'GV_Key', 'Quarter',
    #   'Fiscal Data Year and Quarter', 'Final Date', 'Date_rep', 'Assets',
    #   'Cash', 'Long-Term Debt', 'Invested Capital', 'Liabilities',
    #   'Net Income', 'EBIT', 'EBITDA', 'Revenue', 'Stockholders Equity', 'HQ',
    #   'Share_Price', 'Share_Price_BD', 'Div_Per_Share_Ann', 'Div_Yield',
    #   'Market_Cap', 'Beta_1Y', 'Beta_2Y', 'Beta_5Y', 'Rev_1y_Growth',
    #   'Rev_2y_Growth', 'Primary Industry', 'Industry Sector',
    #   'Industry Group', 'Industry', 'Shares_out', 'EBIT_Margin',
    #   'EBITDA_Margin', 'Profit_Margin', 'Leverage', 'Debt/Assets',
    #   'Cash/Assets', 'ROE', 'P/B', 'P/Sales', 'P/EBITDA', 'P/EBIT', 'P/E',
    #   'Universe'

    columns_to_cluster = ['Industry Group']

    cluster_dataframes = {}

    # For every FQ
    for quarter in encoded_dataframe_all_quarters:
        
        # First clustering based on Industry
        # Subset the dataframe with the selected columns
        clustering_input = encoded_dataframe_all_quarters[quarter][columns_to_cluster]

        # Apply weights to the data, This chan be changed
        weights = [1.00]
        
        # Calculate the pairwise distances between data points
        print('Passing Breakpoint')
        distances = pdist(clustering_input, metric='euclidean', w=weights) # weighted

        # Perform hierarchical clustering using Ward linkage method
        linkage_matrix = linkage(distances, method='ward')

        # In the first run, only 165 Clusters will be done since the weight is 100% on Primary Industry, which was 165 unique values
        max_cluster_size = 500
        cluster_labels = fcluster(linkage_matrix, max_cluster_size, criterion='maxclust')

        # Assign the cluster labels to the DataFrame
        encoded_dataframe_all_quarters[quarter]['Industry Cluster'] = cluster_labels

        # Print resulting clusters
        grouped_encoded_dataframe = encoded_dataframe_all_quarters[quarter].groupby('Industry Cluster')

        # Iterate over each group and print the companies
        for cluster_id, group in grouped_encoded_dataframe:
            companies = group['Company Name'].tolist()
            print(f"Companies in Cluster {cluster_id}: {', '.join(companies)}")

        cluster_counts = pd.Series(cluster_labels).value_counts()
        largest_cluster_size = cluster_counts.max()
        smallest_cluster_size = cluster_counts.min()
        avg_cluster_size = cluster_counts.mean()

        unique_clusters = encoded_dataframe_all_quarters[quarter]['Industry Cluster'].unique()

        clusters_one_quarter = {}
        
        # Iterate over each cluster label
        for cluster_label in unique_clusters:
            # Filter the original DataFrame based on the cluster label
            cluster_data = encoded_dataframe_all_quarters[quarter][encoded_dataframe_all_quarters[quarter]['Industry Cluster'] == cluster_label].copy()
            
            # Assign the filtered data to the current cluster label
            clusters_one_quarter[cluster_label] = cluster_data
        
        # Assign the clusters for the current quarter to the overall cluster_dataframes dictionary
        cluster_dataframes[quarter] = clusters_one_quarter
        
        # To access cluster c from quarter fq:
        # cluster_dataframes['fq'][c]

        cluster_counts = pd.Series(cluster_labels).value_counts()
        largest_cluster_size = cluster_counts.max()
        largest_cluster_index = cluster_counts.idxmax()
        smallest_cluster_size = cluster_counts.min()
        avg_cluster_size = cluster_counts.mean()

        print("Size of the largest cluster:", largest_cluster_size)
        print("Index of the largest cluster:", largest_cluster_index)
        print("Size of the smallest cluster:", smallest_cluster_size)
        print("Size of the average cluster:", avg_cluster_size)
        print("Number of clusters:", len(cluster_counts))

    print("")
    print("Primary Clustering done")
    return cluster_dataframes

# METHOD FOR CLUSTERING
# INPUT: dictionary that consists of dataframes as values for every quarter as key, list of columns to cluster of as well as weights for these columns
# OUTPUT: double linked dictionary that consists of clusters for quarters

# EXAMPLE: Cluster the data based on weights x
# SYNTAX: clustering(dataframes_input, weights)
# SYNTAX: columns_to_cluster = ['EBITDA ($000)','Total Assets ($000)','Total Revenue ($000)','Global Region ']  # Add more columns if needed
# cluster_dfs = clustering(encoded_dataframe_all_quarters,columns_to_cluster,[0.25,0.25,0.25,0.25])

# EXAMPLE: Get dataframe of companies in cluster of certain quarter
# SYNTAX: cluster_dfs[quarter][cluster]

def clustering_kmeans(dataframes_input, columns_to_cluster, weights, optimize, k):
    
    avg_cluster_sizes_quarters = []  
    optimals_quarters = []
    nr_of_clust_quarters = []
    avg_ss_quarters = []
    avg_time_quarters = []
    
    
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)
        
        avg_cluster_sizes_final = []  
        optimals_final = []
        nr_of_clust = []
        avg_ss = []
        
        clustering_start_timer = time.time()
        
        for i in range(1, len(dataframes_input[quarter])+1): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", i)


            dataframe = dataframes_input[quarter][i]
            avg_cluster_sizes =  []
            optimals = []
            dataframe_2 = dataframe.copy()
                            
            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 10 occurrences of the value in the last column (the last column is also the last "run of clustering")
            while (any(dataframe_2.iloc[:, -1].value_counts() > 20)):
                    
                    dataframe = dataframes_input[quarter][i]
                    dataframe_2 = dataframe.copy()
                    print("here")
                
                    values_to_iterate = dataframe_2.iloc[:,-1].dropna().unique().tolist()
                    print("VALUES TO CONSIDER:", values_to_iterate)
                    
                    #while(values_to_iterate):
                    for value in values_to_iterate:

                        #value = values_to_iterate.pop(0)

                        run = 1
                        print("Now looking at: " , value, " in row", dataframe_2.columns[-1])
                        
                        # If Subset of Cluster is only one company - no distances can be calculated!  
                        if (len(dataframe_2[dataframe_2.iloc[:,-1] == value]) <= 10):
                            print("FQ: ", quarter,"Peer group ", i,"is too small: Only" , len(dataframe_2[dataframe_2.iloc[:, -1] == value]) , "companies in this group")
                            print("not breaking")
                            continue
                        
                        else:
                            
                            if (len(dataframe_2[dataframe_2.iloc[:, -1] == value]) > 10):
                                                        
                                previous_cluster_index = dataframe_2.columns[-1][-1]
    
                                print("NOW SPLITTING UP: FQ", quarter, " cluster", i, " on value ", value, "// ", len(dataframe_2[dataframe_2.iloc[:, -1] == value]) , "companies in this group")
    
                                # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
                                cluster_input = dataframe_2[dataframe_2.iloc[:, -1] == value][columns_to_cluster]
    
                                # Indices (for later on to paste right cluster labels)
                                indices = dataframe_2[dataframe_2.iloc[:, -1] == value].index
                                if (len(indices) == 1):
                                    # This is some bug fixing i encountered
                                    print("breaking")
                                    break
    
                                wcss = [] 
                                k_list = []
                                                               
                                print("Time started")
                                start = time.time()
                                
                                try:
                                    optimal_k = optimal_k
                                except NameError:
                                    optimal_k = 2
                                    
                                if optimize == True and optimal_k == 2:
                                    
                                    for k_i in range(1,7):
                                        
                                        # change parameters if more efficient computation needed
                                        kmeans = KMeans(n_clusters=k_i,n_init=10, init = 'k-means++', 
                                                        max_iter = 100)  
                                        #weighted_cluster_input = cluster_input * weights
        
                                        # Fit the weighted K-means model
                                        kmeans.fit(cluster_input)
                                        wcss.append(kmeans.inertia_)
                                        k_list.append(k_i)
                                        
                                    plt.plot(wcss)
                                        
                                    # automatic definition of the optimal number of cluster
                                    
                                    # variant 1: by analyzing the change in the slope of the WCSS curve
                                    
                                    #differences = [wcss[w] - wcss[w - 1] for w in range(1, len(wcss))]
                                    #optimal_k = differences.index(max(differences)) + 2
                                    
                                    # variant 2: defining optimal k using kneed package
                                                        
                                    try:
                                        kneedle = KneeLocator(k_list, wcss, S=1.0, curve="convex", direction="decreasing")
                                        optimal_k = kneedle.knee
                                        optimal_k = int(optimal_k)
                                    except:
                                        try:
                                            kneedle = KneeLocator(k_list, wcss, S=2.0, curve="convex", direction="decreasing")
                                            optimal_k = kneedle.knee
                                            optimal_k = int(optimal_k)
                                        except:
                                            optimal_k = 2
                                    
                                    optimal_k = int(optimal_k)
                                    
                                                                    
                                    
                                if optimize == False:
                                   optimal_k = k
                                   
                                   
                                # perform clustering with the optimal k
                                
                                kmeans = KMeans(n_clusters=optimal_k,n_init=10, init = 'k-means++', 
                                                max_iter = 100)
                                
                                # Get the cluster assignments
                                cluster_labels = kmeans.fit_predict(cluster_input)
                                
                                end = time.time()
                                print("Time ended", end - start)
    
                                #cluster_labels = kmeans.labels_
                                
                                cluster_stack = cluster_labels.tolist()
                                s = 0
    
                                while(cluster_stack):
                                    #value = cluster_stack.pop(0)
                                    #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                                    #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                                    #k += 1
                                    values = cluster_stack.pop(0)
                                    row_index = indices[s]
    
                                    # THIS NEEDS TO BE CHANGED
                                    # dataframe_2.iloc[row_index,-1] = values
    
                                    old_subcluster = str(dataframe_2.loc[row_index,dataframe_2.columns[-1]])
                                    new_subcluster = old_subcluster + "." + str(values)
                                    dataframe.at[row_index, dataframe.columns[-1]] = new_subcluster
                                    s += 1
    
                                print("Splitting ", quarter, " ", i, "done: ", len(dataframe[dataframe.iloc[:, -1] == value]))
                                
                                run += 1
                                
                        optimals.append(optimal_k)
                        optimal_k = 2
                        
                                
                    print("here after break")
                    dataframes_input[quarter][i] = dataframe
                                
            # some stats
            all_cluster_labels = dataframe.iloc[:,-1].dropna().tolist()
            cluster_counts = pd.Series(all_cluster_labels).value_counts()
            largest_cluster_size = cluster_counts.max()
            smallest_cluster_size = cluster_counts.min()
            avg_cluster_size = cluster_counts.mean()
            
            avg_cluster_sizes.append(avg_cluster_size)
                    
             
            avg_cluster_sizes_final.append(pd.Series(avg_cluster_sizes).mean())
            optimals_final.append(pd.Series(optimals).mean())
            if len(cluster_counts) >= 2:
                sil = ss(dataframe[columns_to_cluster].to_numpy(), dataframe.iloc[:,-1])
                avg_ss.append(sil)
                print("Silhouette score of these industry's clusters: ", sil )            
            nr_of_clust.append(len(cluster_counts))
            
            print("Number of clusters:")
            print(cluster_counts)
            print("Avg cluster size:")
            print(avg_cluster_size)
            print("")
            
            
        print("")
        avg_cluster_sizes_quarter = pd.Series(avg_cluster_sizes_final).mean()
        avg_cluster_sizes_quarters.append(avg_cluster_sizes_quarter)
        print(f"Avg. cluster size for the quarter {quarter}:", avg_cluster_sizes_quarter)
        optimals_quarter = pd.Series(optimals_final).mean()
        optimals_quarters.append(optimals_quarter)
        print(f"Optimal k (avg) for quarter {quarter}:", optimals_quarter)
        avg_ss_quarter = pd.Series(avg_ss).mean()
        avg_ss_quarters.append(avg_ss_quarter)
        print(f"Avg. Silhouette score for quarter {quarter}:", avg_ss_quarter)
        nr_of_clust_quarter = pd.Series(nr_of_clust).mean()
        nr_of_clust_quarters.append(nr_of_clust_quarter)
        print(f"Avg. nr of clusters for quarter {quarter}:", nr_of_clust_quarter)
        print("")
        clustering_end_timer = time.time()
        cluster_time = (clustering_end_timer - clustering_start_timer)/60
        avg_time_quarters.append(cluster_time)
        print(f"Time needed for quarter {quarter}", cluster_time)
        
                            
    print("")           
    print("Clustering done")
    print("Avg. cluster size for all quarters:", pd.Series(avg_cluster_sizes_quarters).mean())
    print("Optimal k (avg) for all quarters:", pd.Series(optimals_quarters).mean())
    print("Avg. Silhouette score for all quarters:", pd.Series(avg_ss_quarters).mean())
    print("Avg. nr of clusters for all quarters:", pd.Series(nr_of_clust_quarters).mean())
    print("Avg. time needed for all quarters:", pd.Series(avg_time_quarters).mean())
    
    num_results = pd.DataFrame()
    num_results['Quarters'] = list(dataframes_input.keys())
    num_results['Cluster size'] = avg_cluster_sizes_quarters
    num_results['Optimal k'] = optimals_quarters
    num_results['Silhouette score'] = avg_ss_quarters
    num_results['Nr of clusters'] = nr_of_clust_quarters
    num_results['Time needed'] = avg_time_quarters
    
    return dataframes_input, num_results


#no limit per cluster variant of kmeans

def clustering_kmeans_no_limit(dataframes_input, columns_to_cluster, weights, optimize, k):
    avg_cluster_sizes_quarters = []  
    optimals_quarters = []
    nr_of_clust_quarters = []
    avg_ss_quarters = []
    avg_time_quarters = []
    
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)
        
        avg_cluster_sizes_final = []  
        optimals_final = []
        nr_of_clust = []
        avg_ss = []
        
        clustering_start_timer = time.time()
        
        for i in range(1, len(dataframes_input[quarter])+1): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", i)


            dataframe = dataframes_input[quarter][i]
            avg_cluster_sizes =  []
            optimals = []
                            
            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 10 occurrences of the value in the last column (the last column is also the last "run of clustering")
            #while (any(dataframe.iloc[:, -1].value_counts() > 10)):
                    
            dataframe = dataframes_input[quarter][i]
            dataframe_2 = dataframe.copy()
            print("here")
        
            values_to_iterate = dataframe.iloc[:,-1].dropna().unique().tolist()
            print("VALUES TO CONSIDER:", values_to_iterate)
            
            #while(values_to_iterate):
            
            run = 1
            print("Now looking at: " , i, " in column", dataframe.columns[-1])
                                                    
            previous_cluster_index = dataframe.columns[-1][-1]

            print("NOW SPLITTING UP: FQ", quarter, " cluster", i, "// ", len(dataframe[dataframe.iloc[:, -1] == i]) , "companies in this group")

            # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
            cluster_input = dataframe[dataframe.iloc[:, -1] == i][columns_to_cluster]

            # Indices (for later on to paste right cluster labels)
            indices = dataframe[dataframe.iloc[:, -1] == i].index
            if (len(indices) == 1):
                # This is some bug fixing i encountered
                print("breaking")
                break

            wcss = [] 
            k_list = []
                                           
            print("Time started")
            start = time.time()
            
            try:
                optimal_k = optimal_k
            except NameError:
                optimal_k = 2
            upper_limit = int(len(cluster_input)) if int(len(cluster_input)) <= int(7) else int(7)
            
                
            if optimize == True and optimal_k == 2:
                
                for k_i in range(1,upper_limit):
                    
                    # change parameters if more efficient computation needed
                    kmeans = KMeans(n_clusters=k_i,n_init=10, init = 'k-means++', 
                                    max_iter = 200)  
                    #weighted_cluster_input = cluster_input * weights

                    # Fit the weighted K-means model
                    try:
                        kmeans.fit(cluster_input)
                        wcss.append(kmeans.inertia_)
                        k_list.append(k_i)
                    except:
                        continue
                    
                plt.plot(wcss)
                
                # automatic definition of the optimal number of cluster
                
                # variant 1: by analyzing the change in the slope of the WCSS curve
                
                #differences = [wcss[w] - wcss[w - 1] for w in range(1, len(wcss))]
                #optimal_k = differences.index(max(differences)) + 2
                
                # variant 2: defining optimal k using kneed package
                
                try:
                    kneedle = KneeLocator(k_list, wcss, S=1.0, curve="convex", direction="decreasing")
                    optimal_k = kneedle.knee
                    optimal_k = int(optimal_k)
                except:
                    try:
                        kneedle = KneeLocator(k_list, wcss, S=2.0, curve="convex", direction="decreasing")
                        optimal_k = kneedle.knee
                        optimal_k = int(optimal_k)
                    except:
                        optimal_k = 2
                
                optimal_k = int(optimal_k)
                
                                                
                
            if optimize == False:
               optimal_k = k
               
               
            # perform clustering with the optimal k
            
            kmeans = KMeans(n_clusters=optimal_k,n_init=10, init = 'k-means++', 
                            max_iter = 100)
            
            # Get the cluster assignments
            cluster_labels = kmeans.fit_predict(cluster_input)
            
            end = time.time()
            print("Time ended", end - start)

            #cluster_labels = kmeans.labels_
            
            cluster_stack = cluster_labels.tolist()
            s = 0

            while(cluster_stack):
                #value = cluster_stack.pop(0)
                #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                #k += 1
                values = cluster_stack.pop(0)
                row_index = indices[s]

                # THIS NEEDS TO BE CHANGED
                # dataframe_2.iloc[row_index,-1] = values

                old_subcluster = str(dataframe.loc[row_index,dataframe.columns[-1]])
                new_subcluster = old_subcluster + "." + str(values)
                dataframe_2.at[row_index, dataframe_2.columns[-1]] = new_subcluster
                s += 1

            print("Splitting ", quarter, " ", i, "done: ", len(dataframe[dataframe.iloc[:, -1] == i]))
            
            run += 1
                    
            optimals.append(optimal_k)
            optimal_k = 2
                
                        
            print("here after break")
            dataframes_input[quarter][i] = dataframe_2
                                
            # some stats
            #all_cluster_labels = dataframe.iloc[:,-1].dropna().tolist()
            cluster_counts = pd.Series(cluster_labels).value_counts()
            largest_cluster_size = cluster_counts.max()
            smallest_cluster_size = cluster_counts.min()
            avg_cluster_size = cluster_counts.mean()
            
            avg_cluster_sizes.append(avg_cluster_size)
                    
            avg_cluster_sizes_final.append(pd.Series(avg_cluster_sizes).mean())
            optimals_final.append(pd.Series(optimals).mean())
            if len(cluster_counts) >= 2 and len(dataframe) > 2:
                try:
                    sil = ss(dataframe_2[columns_to_cluster].to_numpy(), dataframe_2.iloc[:,-1])
                    avg_ss.append(sil)
                    print("Silhouette score of these industry's clusters: ", sil )            
                except:
                    pass
            nr_of_clust.append(len(cluster_counts))
            
            print("Number of clusters:")
            print(cluster_counts)
            print("Avg cluster size:")
            print(avg_cluster_size)
            print("") 
                        
        print("")
        avg_cluster_sizes_quarter = pd.Series(avg_cluster_sizes_final).mean()
        avg_cluster_sizes_quarters.append(avg_cluster_sizes_quarter)
        print(f"Avg. cluster size for the quarter {quarter}:", avg_cluster_sizes_quarter)
        optimals_quarter = pd.Series(optimals_final).mean()
        optimals_quarters.append(optimals_quarter)
        print(f"Optimal k (avg) for quarter {quarter}:", optimals_quarter)
        avg_ss_quarter = pd.Series(avg_ss).mean()
        avg_ss_quarters.append(avg_ss_quarter)
        print(f"Avg. Silhouette score for quarter {quarter}:", avg_ss_quarter)
        nr_of_clust_quarter = pd.Series(nr_of_clust).mean()
        nr_of_clust_quarters.append(nr_of_clust_quarter)
        print(f"Avg. nr of clusters for quarter {quarter}:", nr_of_clust_quarter)
        print("")
        clustering_end_timer = time.time()
        cluster_time = (clustering_end_timer - clustering_start_timer)/60
        avg_time_quarters.append(cluster_time)
        print(f"Time needed for quarter {quarter}", cluster_time)
                    
    print("")           
    print("Clustering done")
    print("Avg. cluster size for all quarters:", pd.Series(avg_cluster_sizes_quarters).mean())
    print("Optimal k (avg) for all quarters:", pd.Series(optimals_quarters).mean())
    print("Avg. Silhouette score for all quarters:", pd.Series(avg_ss_quarters).mean())
    print("Avg. nr of clusters for all quarters:", pd.Series(nr_of_clust_quarters).mean())
    print("Avg. time needed for all quarters:", pd.Series(avg_time_quarters).mean())
    
    num_results = pd.DataFrame()
    num_results['Quarters'] = list(dataframes_input.keys())
    num_results['Cluster size'] = avg_cluster_sizes_quarters
    num_results['Optimal k'] = optimals_quarters
    num_results['Silhouette score'] = avg_ss_quarters
    num_results['Nr of clusters'] = nr_of_clust_quarters
    num_results['Time needed'] = avg_time_quarters
    
    return dataframes_input, num_results



# function for DBSCAN - attempt version without min. 10 companies condition loop

def clustering_dbscan_gridsearch(dataframes_input, columns_to_cluster, weights, optimize, eps, min_pts):
    
    avg_cluster_sizes_quarters = []  
    optimals_quarters = []
    nr_of_clust_quarters = []
    avg_ss_quarters = []
    avg_time_quarters = []
    optimal_minpts_quarters = []
    
    
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)
        
        avg_cluster_sizes_final = []  
        optimals_final = []
        nr_of_clust = []
        avg_ss = []
        
        
        clustering_start_timer = time.time()
        
        for ind in range(1, len(dataframes_input[quarter])+1): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", ind)


            dataframe = dataframes_input[quarter][ind]
            avg_cluster_sizes =  []
            optimals = []
            optimal_minpts = []
            
            
                            
            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 10 occurrences of the value in the last column (the last column is also the last "run of clustering")
            #while (any(dataframe.iloc[:, -1].value_counts() > 10)):
                
            #dataframe = dataframes_input[quarter][i]
            dataframe_2 = dataframe.copy()
            print("here")
        
            #values_to_iterate = dataframe.iloc[:,-1].dropna().unique().tolist()
            #print("VALUES TO CONSIDER:", values_to_iterate)
            
            run = 1
            print("Now looking at: " , ind, " in column", dataframe.columns[-1])
            
            previous_cluster_index = dataframe.columns[-1][-1]

            print("NOW SPLITTING UP: FQ", quarter, " cluster", ind, "// ", len(dataframe[dataframe.iloc[:, -1] == ind]) , "companies in this group")

            # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
            cluster_input = dataframe[dataframe.iloc[:, -1] == ind][columns_to_cluster]
            #weighted_cluster_input = cluster_input * weights

            # Indices (for later on to paste right cluster labels)
            indices = dataframe[dataframe.iloc[:, -1] == ind].index
            if (len(indices) <= 2):
                # This is some bug fixing i encountered
                print("breaking")
                continue
            
            # must be maximum 10 because there are some industries with only 10 companies = samples, so 
            # minpts > 10 would not be possible in these industries
            minpts = min_pts
                        
            print("Time started")
            start = time.time()
            
            start = 2
            upper_limit = int(len(cluster_input)) if int(len(cluster_input)) <= int(10) else int(10)
                        
            ss_scores_per_m_i = []
            m_i_indices = []
            optimal_eps_indices = []
            
            if optimize == True:
                
                # implement optimization of MinPts
                for m_i in range(start, upper_limit):
                                        
                    try:
                    
                        # perform clustering with rule of thumb for minpts and approximation for eps using KNN
                        
                        neigh = NearestNeighbors(n_neighbors=m_i)
                        nbrs = neigh.fit(cluster_input)
                        distances, d_indices = nbrs.kneighbors(cluster_input)
                        
                        distances = np.sort(distances, axis=0)
                        distances = np.mean(distances, axis=1)
                        #plt.plot(distances)
                        
                        # kneedle optimization
                                                  
                        try:
                            kneedle = KneeLocator(np.indices(distances.shape)[0].tolist(), 
                                                  distances.tolist(), S=1.0, curve="convex", direction="increasing" )
                            optimal_eps = kneedle.knee_y
                        except:
                            try:
                                kneedle = KneeLocator(np.indices(distances.shape)[0].tolist(), 
                                                      distances.tolist(), S=2.0, curve="convex", direction="increasing" )
                                optimal_eps = kneedle.knee_y
                            except:
                                optimal_eps = eps
                        
                        dbscan = DBSCAN(eps = optimal_eps, min_samples=m_i)
                        dbscan.fit(cluster_input)
                    except:
                        optimal_eps = eps
                        dbscan = DBSCAN(eps = optimal_eps, min_samples=m_i)
                        dbscan.fit(cluster_input)
                        
                    cluster_labels = dbscan.labels_
                    cluster_counts = pd.Series(cluster_labels).value_counts()
                        
                    if len(cluster_counts) >= 2 and len(cluster_input) > 2:
                        try:
                            sil = ss(cluster_input.to_numpy(), cluster_labels)
                            
                            #print("Silhouette score of these industry's clusters: ", sil ) 
                        except:
                            print('Too few clusters to compute SS!')
                    else:
                        sil = 0.5
                            
                    ss_scores_per_m_i.append(sil)
                    m_i_indices.append(m_i)
                    optimal_eps_indices.append(optimal_eps)
                    
                try:
                    max_i = ss_scores_per_m_i.index(max(ss_scores_per_m_i))
                except:
                    pass
                optimal_minpts = m_i_indices[int(max_i)]
                optimal_eps_final = optimal_eps_indices[int(max_i)]
                
                # actual dbscan application with optimized parameters
                dbscan = DBSCAN(eps = optimal_eps_final, min_samples=optimal_minpts)
                dbscan.fit(cluster_input)
                cluster_labels_final = dbscan.labels_
                    
            
            else:
                optimal_eps = eps
                dbscan = DBSCAN(eps = optimal_eps, min_samples=minpts)
                dbscan.fit(cluster_input)
                cluster_labels = dbscan.labels_
            
            
            # Get the cluster assignments
            #cluster_labels = dbscan.labels_
            
            end = time.time()
            print("Time ended", end - start)

            
            
            cluster_stack = cluster_labels_final.tolist()
            s = 0

            while(cluster_stack):
                #value = cluster_stack.pop(0)
                #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                #k += 1
                values = cluster_stack.pop(0)
                row_index = indices[s]

                old_subcluster = str(dataframe.loc[row_index,dataframe.columns[-1]])
                new_subcluster = old_subcluster + "." + str(values)
                dataframe_2.at[row_index, dataframe_2.columns[-1]] = new_subcluster
                s += 1

            run += 1
            optimals.append(optimal_eps_final)
                                    
            print("here after break")
            dataframes_input[quarter][ind] = dataframe_2
            
                                
            # some stats
            cluster_counts = pd.Series(cluster_labels_final).value_counts()
            largest_cluster_size = cluster_counts.max()
            smallest_cluster_size = cluster_counts.min()
            avg_cluster_size = cluster_counts.mean()
            
            avg_cluster_sizes.append(avg_cluster_size)
                    
             
            avg_cluster_sizes_final.append(pd.Series(avg_cluster_sizes).mean())
            optimals_final.append(pd.Series(optimals).mean())
            
            if len(cluster_counts) >= 2 and len(dataframe_2) > 2:
                try:
                    #sil = ss(cluster_input.to_numpy(), cluster_labels)
                    sil = ss(dataframe_2[columns_to_cluster].to_numpy(), dataframe_2.iloc[:,-1])
                    avg_ss.append(sil)
                    print("Silhouette score of these industry's clusters: ", sil )  
                except:
                    pass
        
            nr_of_clust.append(len(cluster_counts))
            optimal_eps_indices.append(optimal_minpts)
            
            print("Number of clusters:")
            print(cluster_counts)
            print("Avg cluster size:")
            print(avg_cluster_size)
            #print("Splitting ", quarter, " ", ind, "done: ", len(dataframe[dataframe.iloc[:, -1] == ind]))
            
        print("")
        avg_cluster_sizes_quarter = pd.Series(avg_cluster_sizes_final).mean()
        avg_cluster_sizes_quarters.append(avg_cluster_sizes_quarter)
        print(f"Avg. cluster size for the quarter {quarter}:", avg_cluster_sizes_quarter)
        optimals_quarter = pd.Series(optimals_final).mean()
        optimals_quarters.append(optimals_quarter)
        print(f"Optimal k (avg) for quarter {quarter}:", optimals_quarter)
        avg_ss_quarter = pd.Series(avg_ss).mean()
        avg_ss_quarters.append(avg_ss_quarter)
        print(f"Avg. Silhouette score for quarter {quarter}:", avg_ss_quarter)
        nr_of_clust_quarter = pd.Series(nr_of_clust).mean()
        nr_of_clust_quarters.append(nr_of_clust_quarter)
        print(f"Avg. nr of clusters for quarter {quarter}:", nr_of_clust_quarter)
        print("")
        clustering_end_timer = time.time()
        cluster_time = (clustering_end_timer - clustering_start_timer)/60
        avg_time_quarters.append(cluster_time)
        print(f"Time needed for quarter {quarter}", cluster_time)
        optimal_minpts_quarter = pd.Series(optimal_eps_indices).mean()
        optimal_minpts_quarters.append(optimal_minpts_quarter)
            
                                
    print("")           
    print("Clustering done")
    print("Avg. cluster size for all quarters:", pd.Series(avg_cluster_sizes_quarters).mean())
    print("Optimal eps (avg) for all quarters:", pd.Series(optimals_quarters).mean())
    print("Avg. Silhouette score for all quarters:", pd.Series(avg_ss_quarters).mean())
    print("Avg. nr of clusters for all quarters:", pd.Series(nr_of_clust_quarters).mean())
    print("Avg. time needed for all quarters:", pd.Series(avg_time_quarters).mean())
    
    num_results = pd.DataFrame()
    num_results['Quarters'] = list(dataframes_input.keys())
    num_results['Cluster size'] = avg_cluster_sizes_quarters
    num_results['Optimal eps'] = optimals_quarters
    num_results['Optimal minpts'] = optimal_minpts_quarters
    num_results['Silhouette score'] = avg_ss_quarters
    num_results['Nr of clusters'] = nr_of_clust_quarters
    num_results['Time needed'] = avg_time_quarters
        
    return dataframes_input, num_results
    


# function for DBSCAN - attempt version without min. 10 companies condition loop

def clustering_dbscan(dataframes_input, columns_to_cluster, weights, optimize, eps):
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)
        
        avg_cluster_sizes_final = []  
        optimals_final = []
        
        clustering_start_timer = time.time()
        
        for i in range(1, len(dataframes_input[quarter])+1): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", i)


            dataframe = dataframes_input[quarter][i]
            avg_cluster_sizes =  []
            optimals = []
                            
            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 10 occurrences of the value in the last column (the last column is also the last "run of clustering")
            #while (any(dataframe.iloc[:, -1].value_counts() > 10)):
                    
            #dataframe = dataframes_input[quarter][i]
            dataframe_2 = dataframe.copy()
            print("here")
        
            values_to_iterate = dataframe.iloc[:,-1].dropna().unique().tolist()
            print("VALUES TO CONSIDER:", values_to_iterate)
            
            
            #while(values_to_iterate):
            for value in values_to_iterate:

                #value = values_to_iterate.pop(0)

                run = 1
                print("Now looking at: " , value, " in row", dataframe.columns[-1])
                
                # without if else clause (check whether >2 and <10)
                
                previous_cluster_index = dataframe.columns[-1][-1]

                print("NOW SPLITTING UP: FQ", quarter, " cluster", i, " on value ", value, "// ", len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")

                # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
                cluster_input = dataframe[dataframe.iloc[:, -1] == value][columns_to_cluster]
                #weighted_cluster_input = cluster_input * weights

                # Indices (for later on to paste right cluster labels)
                indices = dataframe[dataframe.iloc[:, -1] == value].index
                if (len(indices) == 1):
                    # This is some bug fixing i encountered
                    print("breaking")
                    break

                # Define MinPts argument using rule of thumb
                # e.g.
                #minpts = cluster_input.shape[1] +1
                
                # must be maximum 10 because there are some industries with only 10 companies = samples, so 
                # minpts > 10 would not be possible in these industries
                minpts = 5
                
                
                print("Time started")
                start = time.time()

                if optimize == True:
                    
                    # perform clustering with rule of thumb for minpts and approximation for eps using KNN
                    
                    neigh = NearestNeighbors(n_neighbors=minpts)
                    nbrs = neigh.fit(cluster_input)
                    distances, d_indices = nbrs.kneighbors(cluster_input)
                    
                    distances = np.sort(distances, axis=0)
                    distances = np.mean(distances, axis=1)
                    plt.plot(distances)
                    
                    # kneedle optimization
                    
                    kneedle = KneeLocator(np.indices(distances.shape)[0].tolist(), 
                                          distances.tolist(), S=2.0, curve="convex", direction="increasing" )
                    
                    optimal_eps = kneedle.knee_y
                    
                    # adjust for DBSCAN case
                    #differences = [distances[d] - distances[d - 1] for d in range(1, len(distances))]
                    
                    #optimal_eps = distances[differences.index(max(differences)) + 1]
                
                else:
                    optimal_eps = eps
                
                dbscan = DBSCAN(eps = optimal_eps, min_samples=minpts)
                dbscan.fit(cluster_input)
                
                # Get the cluster assignments
                cluster_labels = dbscan.labels_
                
                end = time.time()
                print("Time ended", end - start)

                
                
                cluster_stack = cluster_labels.tolist()
                s = 0

                while(cluster_stack):
                    #value = cluster_stack.pop(0)
                    #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                    #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                    #k += 1
                    values = cluster_stack.pop(0)
                    row_index = indices[s]

                    # THIS NEEDS TO BE CHANGED
                    # dataframe_2.iloc[row_index,-1] = values

                    old_subcluster = str(dataframe.loc[row_index,dataframe.columns[-1]])
                    new_subcluster = old_subcluster + "." + str(values)
                    dataframe_2.at[row_index, dataframe_2.columns[-1]] = new_subcluster
                    s += 1

                print("Splitting ", quarter, " ", i, "done: ", len(dataframe[dataframe.iloc[:, -1] == value]))
                
                run += 1
                
                
                                
                optimals.append(optimal_eps)
                                        
                            
                print("here after break")
                dataframes_input[quarter][i] = dataframe_2
                                
            # some stats
            cluster_counts = pd.Series(cluster_labels).value_counts()
            largest_cluster_size = cluster_counts.max()
            smallest_cluster_size = cluster_counts.min()
            avg_cluster_size = cluster_counts.mean()
            
            avg_cluster_sizes.append(avg_cluster_size)
                    
             
            avg_cluster_sizes_final.append(pd.Series(avg_cluster_sizes).mean())
            optimals_final.append(pd.Series(optimals).mean())
            
        print("")
        print(f"Avg. cluster size for the quarter {quarter}:", pd.Series(avg_cluster_sizes_final).mean())
        print(f"Optimal epsilon (avg) for quarter {quarter}:", pd.Series(optimals_final).mean())
        print("")
        clustering_end_timer = time.time()
        print(f"Time needed for quarter {quarter}", clustering_end_timer - clustering_start_timer)
                    
    print("")           
    print("Clustering done")
    
    return dataframes_input


# hierarchical clustering

def clustering_hclust(dataframes_input, columns_to_cluster, weights, optimize, k):
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)
        
        avg_cluster_sizes_final = []  
        #optimals_final = []
        
        clustering_start_timer = time.time()
        
        for i in range(1, len(dataframes_input[quarter])+1): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", i)


            dataframe = dataframes_input[quarter][i]
            avg_cluster_sizes =  []
            optimals = []
                            
            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 10 occurrences of the value in the last column (the last column is also the last "run of clustering")
            while (any(dataframe.iloc[:, -1].value_counts() > 10)):
                    
                    dataframe = dataframes_input[quarter][i]
                    dataframe_2 = dataframe.copy()
                    print("here")
                
                    values_to_iterate = dataframe.iloc[:,-1].dropna().unique().tolist()
                    print("VALUES TO CONSIDER:", values_to_iterate)
                    
                    #while(values_to_iterate):
                    for value in values_to_iterate:

                        #value = values_to_iterate.pop(0)

                        run = 1
                        print("Now looking at: " , value, " in row", dataframe.columns[-1])
                        
                        # If Subset of Cluster is only one company - no distances can be calculated!  
                        if (len(dataframe[dataframe.iloc[:,-1] == value]) < 2):
                            print("FQ: ", quarter,"Peer group ", i,"is too small: Only" , len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")
                            print("not breaking")
                        
                        else:
                            
                            if (len(dataframe[dataframe.iloc[:, -1] == value]) > 10):
                                                        
                                previous_cluster_index = dataframe.columns[-1][-1]
    
                                print("NOW SPLITTING UP: FQ", quarter, " cluster", i, " on value ", value, "// ", len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")
    
                                # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
                                cluster_input = dataframe[dataframe.iloc[:, -1] == value][columns_to_cluster]
    
                                # Indices (for later on to paste right cluster labels)
                                indices = dataframe[dataframe.iloc[:, -1] == value].index
                                if (len(indices) == 1):
                                    # This is some bug fixing i encountered
                                    print("breaking")
                                    break
    
                                # Weights on the input parameters
                                
                                print("Time started")
                                start = time.time()
                                
                                ####
                                
                                # Calculate the pairwise distances between data points
                                
                                distances = pdist(cluster_input, metric='euclidean') # weighted

                                # Perform hierarchical clustering using Ward linkage method
                                linkage_matrix = linkage(distances, method='ward')

                                max_cluster_size = 10
                                cluster_labels = fcluster(linkage_matrix, max_cluster_size, criterion='maxclust')
                                
                                end = time.time()
                                print("Time ended", end - start)
    
                                #cluster_labels = kmeans.labels_
                                
                                cluster_stack = cluster_labels.tolist()
                                s = 0
    
                                while(cluster_stack):
                                    #value = cluster_stack.pop(0)
                                    #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                                    #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                                    #k += 1
                                    values = cluster_stack.pop(0)
                                    row_index = indices[s]
    
                                    # THIS NEEDS TO BE CHANGED
                                    # dataframe_2.iloc[row_index,-1] = values
    
                                    old_subcluster = str(dataframe.loc[row_index,dataframe.columns[-1]])
                                    new_subcluster = old_subcluster + "." + str(values)
                                    dataframe_2.at[row_index, dataframe_2.columns[-1]] = new_subcluster
                                    s += 1
    
                                print("Splitting ", quarter, " ", i, "done: ", len(dataframe[dataframe.iloc[:, -1] == value]))
                                
                                run += 1
                                
                    print("here after break")
                    dataframes_input[quarter][i] = dataframe_2
                                
            # some stats
            cluster_counts = pd.Series(cluster_labels).value_counts()
            largest_cluster_size = cluster_counts.max()
            smallest_cluster_size = cluster_counts.min()
            avg_cluster_size = cluster_counts.mean()
            
            avg_cluster_sizes.append(avg_cluster_size)
                    
             
            avg_cluster_sizes_final.append(pd.Series(avg_cluster_sizes).mean())
            
        print("")
        print(f"Avg. cluster size for the quarter {quarter}:", pd.Series(avg_cluster_sizes_final).mean())
        print("")
        clustering_end_timer = time.time()
        print(f"Time needed for quarter {quarter}", clustering_end_timer - clustering_start_timer)
                    
    print("")           
    print("Clustering done")
    
    return dataframes_input

# function for DBSCAN

def clustering_dbscan_old(dataframes_input, columns_to_cluster, weights, optimize, eps):
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)
        
        avg_cluster_sizes_final = []  
        optimals_final = []
        
        clustering_start_timer = time.time()
        
        for i in range(1, len(dataframes_input[quarter])+1): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", i)


            dataframe = dataframes_input[quarter][i]
            avg_cluster_sizes =  []
            optimals = []
                            
            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 10 occurrences of the value in the last column (the last column is also the last "run of clustering")
            #while (any(dataframe.iloc[:, -1].value_counts() > 10)):
                    
            #dataframe = dataframes_input[quarter][i]
            dataframe_2 = dataframe.copy()
            print("here")
        
            values_to_iterate = dataframe.iloc[:,-1].dropna().unique().tolist()
            print("VALUES TO CONSIDER:", values_to_iterate)
            
            #while(values_to_iterate):
            for value in values_to_iterate:

                #value = values_to_iterate.pop(0)

                run = 1
                print("Now looking at: " , value, " in row", dataframe.columns[-1])
                
                # If Subset of Cluster is only one company - no distances can be calculated!  
                if (len(dataframe[dataframe.iloc[:,-1] == value]) < 2):
                    print("FQ: ", quarter,"Peer group ", i,"is too small: Only" , len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")
                    print("not breaking")
                
                else:
                    
                    if (len(dataframe[dataframe.iloc[:, -1] == value]) > 10):
                                                
                        previous_cluster_index = dataframe.columns[-1][-1]

                        print("NOW SPLITTING UP: FQ", quarter, " cluster", i, " on value ", value, "// ", len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")

                        # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
                        cluster_input = dataframe[dataframe.iloc[:, -1] == value][columns_to_cluster]
                        #weighted_cluster_input = cluster_input * weights

                        # Indices (for later on to paste right cluster labels)
                        indices = dataframe[dataframe.iloc[:, -1] == value].index
                        if (len(indices) == 1):
                            # This is some bug fixing i encountered
                            print("breaking")
                            break

                        # Define MinPts argument using rule of thumb
                        # e.g.
                        #minpts = cluster_input.shape[1] +1
                        minpts = 11
                        
                        print("Time started")
                        start = time.time()

                        if optimize == True:
                            
                            # perform clustering with rule of thumb for minpts and approximation for eps using KNN
                            
                            neigh = NearestNeighbors(n_neighbors=minpts)
                            nbrs = neigh.fit(cluster_input)
                            distances, indices = nbrs.kneighbors(cluster_input)
                            
                            distances = np.sort(distances, axis=0)
                            distances = distances[:,1]
                            plt.plot(distances)
                            
                            # adjust for DBSCAN case
                            differences = [distances[d] - distances[d - 1] for d in range(1, len(distances))]
                            
                            optimal_eps = distances[differences.index(max(differences)) + 1]
                        
                        else:
                            optimal_eps = eps
                        
                        dbscan = DBSCAN(eps = optimal_eps, min_samples=minpts)
                        dbscan.fit(cluster_input)
                        
                        # Get the cluster assignments
                        cluster_labels = dbscan.labels_
                        
                        end = time.time()
                        print("Time ended", end - start)

                        
                        
                        cluster_stack = cluster_labels.tolist()
                        s = 0

                        while(cluster_stack):
                            #value = cluster_stack.pop(0)
                            #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                            #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                            #k += 1
                            values = cluster_stack.pop(0)
                            row_index = indices[s]

                            # THIS NEEDS TO BE CHANGED
                            # dataframe_2.iloc[row_index,-1] = values

                            old_subcluster = str(dataframe.loc[row_index,dataframe.columns[-1]])
                            new_subcluster = old_subcluster + "." + str(values)
                            dataframe_2.at[row_index, dataframe_2.columns[-1]] = new_subcluster
                            s += 1

                        print("Splitting ", quarter, " ", i, "done: ", len(dataframe[dataframe.iloc[:, -1] == value]))
                        
                        run += 1
                                
                    optimals.append(optimal_eps)
                                            
                                
                    print("here after break")
                    dataframes_input[quarter][i] = dataframe_2
                                
            # some stats
            cluster_counts = pd.Series(cluster_labels).value_counts()
            largest_cluster_size = cluster_counts.max()
            smallest_cluster_size = cluster_counts.min()
            avg_cluster_size = cluster_counts.mean()
            
            avg_cluster_sizes.append(avg_cluster_size)
                    
             
            avg_cluster_sizes_final.append(pd.Series(avg_cluster_sizes).mean())
            optimals_final.append(pd.Series(optimals).mean())
            
        print("")
        print(f"Avg. cluster size for the quarter {quarter}:", pd.Series(avg_cluster_sizes_final).mean())
        print(f"Optimal number of clusters (avg) for quarter {quarter}:", pd.Series(optimals_final).mean())
        print(f"Optimal number of clusters (avg) for quarter {quarter}:", pd.Series(optimals_final).mean())
        print("")
        clustering_end_timer = time.time()
        print(f"Time needed for quarter {quarter}", clustering_end_timer - clustering_start_timer)
                    
    print("")           
    print("Clustering done")
    
    return dataframes_input



# METHOD TO GROUP DATA IN ORDER TO OUTPUT IT LATER ON
# INPUT: Clustered dictionary of dataframes from previous method
# OUTPUT: Double linked dictionary of dataframes for every quarter and subcluster (OUTPUT 0)
# OUTPUT: Dictionary with quarter as key and dataframe of cluster names in one column with corresponding Entity IDs in other column (OUTPUT 2)


# EXAMPLE: Get all the companies in a cluster: (OUTPUT 1)
# SYNTAX: grouped_dfs[quarter][cluster_name]
# grouped_dfs = group_output(cluster_dfs)[0]
# grouped_dfs["FQ12017"]["1.0.0.0.0"]

# EXAMPLE: Get all clusters and corresponding Entity IDs for a certain quarter:
# SYNTAX: peer_groups_every_quarter[quarter]
# peer_groups_every_quarter = group_output(cluster_dfs)[1]
# peer_groups_every_quarter["FQ12017"]

def group_output(primary_industries_clustered):
    grouped_dfs = {}

    for quarter in primary_industries_clustered:
        dataframes = []

        for i in primary_industries_clustered[quarter].keys():

            peer_group_df = primary_industries_clustered[quarter][i]
            
            #drop outlier indices
            outlier_indices = peer_group_df[peer_group_df.loc[:,"Industry Cluster"].astype(str).str.endswith('-1')].index
            print(f'{len(outlier_indices)} outliers removed')
            peer_group_df.drop(outlier_indices, axis = 0, inplace = True)
                        
            dataframes.append(peer_group_df)
            
        concatenated_df = pd.concat(dataframes)
        
        try:
            nr_outlrs = concatenated_df[concatenated_df.loc[:,"Industry Cluster"].str.endswith('-1')].shape
            print(f"Number of outliers in quarter {quarter}: ", nr_outlrs)
        except:
            pass
        grouped_df = concatenated_df.groupby("Industry Cluster", dropna=False)
        
        
        unique_groups = grouped_df.groups.keys()
        grouped_dfs[quarter] = {}
        
        for group_name in unique_groups:
            grouped_dfs[quarter][group_name] = grouped_df.get_group(group_name)

    peer_groups_every_quarter = {}

    # Create a new DataFrame that has the entity name in one column and all companies in the same group in the other column
    for quarter in grouped_dfs:
        df_list = []
        cluster_sizes_list = []

        # Group all companies from one cluster
        for group in grouped_dfs[quarter]:
            new_rows = pd.DataFrame({"Company Name": grouped_dfs[quarter][group]["Company Name"].tolist(), "Group": group})
            df_list.append(new_rows)
            cluster_sizes_list.append(len(new_rows))

        # Concatenate and add as new column
        temporary_dataframe = pd.concat(df_list, ignore_index=True)
        peer_groups_every_quarter[quarter] = temporary_dataframe.groupby("Group")["Company Name"].apply(list).reset_index()
        #peer_groups_every_quarter[quarter] = temporary_dataframe.groupby("Group")["GV_Key"].apply(list)
    print(np.nanmean(np.array(cluster_sizes_list)))

    return grouped_dfs,peer_groups_every_quarter





    




