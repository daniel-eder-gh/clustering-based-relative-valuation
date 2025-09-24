# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:27:33 2023

@author: Dani
"""

#Trading strategy¶
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

# METHOD TO IMPORT DATA AND SPLIT IT
def input(path):
    #Importing data
    dataset = pd.read_csv(path)
    
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
    for quarter in quarters:
        dataframe_one_quarter = dataset[dataset['Quarter'] == quarter]
        dataframe_all_quarters[quarter] = dataframe_one_quarter
        

    # Encoding the data
    label_encoder = LabelEncoder()

    # Columns to encode
    columns_to_encode = ['Primary Industry', 'Industry Sector', 'Industry Group', 'Industry']

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
    #   'Unnamed: 0.1', 'Unnamed: 0', 'Entity Name ', 'Entity ID ', 'Quarter',
    #   'Total Assets ($000)', 'Total Revenue ($000)', 'Primary Industry ',
    #   '1st Level Primary Industry ', '2nd Level Primary Industry ',
    #   'Global Region ', 'Market Cap', 'EBIT margin', 'EBITDA margin',
    #   'Profit margin', 'Debt-to-assets', 'Leverage', 'Cash-to-assets',
    #   'Enterprise value', 'EV/Sales', 'EV/EBIT', 'EV/EBITDA', 'P/E'

    columns_to_cluster = ['Beta', 'EBITDA margin','Total Assets ($000)','Total Revenue ($000)','Primary Industry ','Global Region ', 'Debt-to-assets', 'Cash-to-assets',
       'Enterprise value', 'EV/Sales', 'EV/EBITDA', 'P/E']  # Add more columns if needed

    cluster_dataframes = {}

    # For every FQ
    for quarter in encoded_dataframe_all_quarters:
        
        # First clustering based on Industry
        # Subset the dataframe with the selected columns
        clustering_input = encoded_dataframe_all_quarters[quarter][columns_to_cluster]

        # Apply weights to the data, This chan be changed
        weights = [0.00, 0.00, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Calculate the pairwise distances between data points
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
            companies = group['Entity Name '].tolist()
            print(f"Companies in Cluster '{cluster_id}': {', '.join(companies)}")

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

def clustering(dataframes_input, columns_to_cluster, weights):
    for quarter in dataframes_input:
        print("CURRENTLY CLUSTERING QUARTER: ", quarter)

        for i in range(1, len(dataframes_input[quarter])): # i for Industry Based Cluster
            print("CURRENTLY CLUSTERING INDUSTRY: ", i)

            # The Peer Group is too big: Value in second last and last column is the same, and there are more than 15 occurrences of the value in the last column (the last column is also the last "run of clustering")
            while (any(dataframes_input[quarter][i].iloc[:, -1].value_counts() > 15)):
                    
                    dataframe = dataframes_input[quarter][i]
                    dataframe_2 = dataframe.copy()
                    print("here")
                
                    values_to_iterate = dataframe.iloc[:,-1].dropna().unique().tolist()
                    print("VALUES TO CONSIDER:", values_to_iterate)
                    
                    #while(values_to_iterate):
                    for value in dataframe.iloc[:,-1].dropna().unique().tolist():

                        #value = values_to_iterate.pop(0)

                        run = 1;
                        print("Now looking at: " , value, " in row", dataframe.columns[-1])
                        
                        # If Subset of Cluster is only one company - no distances can be calculated!  
                        if (len(dataframe[dataframe.iloc[:,-1] == value]) < 2):
                            print("FQ: ", quarter,"Peer group ", i,"is too small: Only" , len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")
                            print("not breaking")
                        
                        else:

                            if (len(dataframe[dataframe.iloc[:, -1] == value]) > 15):
                            
                                previous_cluster_index = dataframe.columns[-1][-1]

                                print("NOW SPLITTING UP: FQ", quarter, " cluster", i, " on value ", value, "// ", len(dataframe[dataframe.iloc[:, -1] == value]) , "companies in this group")

                                # Subset the dataframe with the selected columns: Only the selected rows which are from same cluster (saved in last column)
                                cluster_input = dataframe[dataframe.iloc[:, -1] == value][columns_to_cluster]

                                # Indices (for later on to paste right cluster labels)
                                indices = dataframe[dataframe.iloc[:, -1] == value].index
                                if (len(indices) == 1):
                                    # This is some bug fixing i encountered
                                    print("breaking")
                                    break;

                                # Weights on the input parameters
                                # weights = [0.25, 0.25, 0.25, 0.25]
                                
                                k = 2  # Number of clusters
                                kmeans = KMeans(n_clusters=k,n_init=10)

                                weighted_cluster_input = cluster_input * weights

                                # Fit the weighted K-means model
                                kmeans.fit(weighted_cluster_input)

                                # Get the cluster assignments
                                cluster_labels = kmeans.labels_
                                
                                cluster_stack = cluster_labels.tolist()
                                k = 0;

                                while(cluster_stack):
                                    #value = cluster_stack.pop(0)
                                    #row_index = dataframe.loc[dataframe.iloc[:, -1] == value].index[k]
                                    #dataframe.loc[dataframe.iloc[:, -1] == value, sub_cluster_name].at[row_index, sub_cluster_name] = value
                                    #k += 1
                                    values = cluster_stack.pop(0)
                                    row_index = indices[k]

                                    # THIS NEEDS TO BE CHANGED
                                    # dataframe_2.iloc[row_index,-1] = values

                                    old_subcluster = str(dataframe.loc[row_index,dataframe.columns[-1]])
                                    new_subcluster = old_subcluster + "." + str(values)
                                    dataframe_2.at[row_index, dataframe_2.columns[-1]] = new_subcluster
                                    k += 1

                                print("Splitting ", quarter, " ", i, "done: ", len(dataframe[dataframe.iloc[:, -1] == value]))
                                
                                run += 1;
                    
                    print("here after break")
                    dataframes_input[quarter][i] = dataframe_2;
                    
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

            dataframes.append(primary_industries_clustered[quarter][i])
        concatenated_df = pd.concat(dataframes)
        grouped_df = concatenated_df.groupby("Industry Cluster", dropna=False)
        
        unique_groups = grouped_df.groups.keys()
        grouped_dfs[quarter] = {}
        
        for group_name in unique_groups:
            grouped_dfs[quarter][group_name] = grouped_df.get_group(group_name)

    peer_groups_every_quarter = {}

    # Create a new DataFrame that has the entity name in one column and all companies in the same group in the other column
    for quarter in grouped_dfs:
        df_list = []

        # Group all companies from one cluster
        for group in grouped_dfs[quarter]:
            new_rows = pd.DataFrame({"Entity ID ": grouped_dfs[quarter][group]["Entity ID "].tolist(), "Group": group})
            df_list.append(new_rows)

        # Concatenate and add as new column
        temporary_dataframe = pd.concat(df_list, ignore_index=True)
        peer_groups_every_quarter[quarter] = temporary_dataframe.groupby("Group")["Entity ID "].apply(list).reset_index()
        #peer_groups_every_quarter[quarter] = temporary_dataframe.groupby("Group")["Entity ID "].apply(list)

    return grouped_dfs,peer_groups_every_quarter

# METHOD TO GET PEER GROUP FOR A CERTAIN COMPANY IN A CERTAIN QUARTER
# INPUT: Entity ID of company to find and quarter in which to look as well as the peer_groups_every_quarter from previous method (Dataframe of all peer groups in a quarter)
# OUTPUT: List of Entity IDs of peers

# EXAMPLE: (entity_id of company to look for)
# SYNTAX: find_peer_group(entity_id, quarter)
# find_peer_group(4161767,"FQ12019")

def find_peer_group(entity_id, quarter, peer_groups_every_quarter):
        df = peer_groups_every_quarter[quarter]
        for row in range(0, len(df)):
            if entity_id in df.iloc[row, 1]:
                #print("Quarter: ", quarter, "      Peer Group:", df.iloc[row,1])
                return df.iloc[row,1]                
            
# METHOD TO GET THE MEDIAN MULTIPLE FOR EVERY QUARTER OF CERTAIN COMPANY
# INPUT: Company Entity ID as well as grouped_dfs from previous method (Dictionary with quarters and clusters)
# OUTPUT: Dataframe with quarters in one column and median and mean multiple for the corresponding quarter in columns

# EXAMPLE: Get the multiples for company x
# SYNTAX: multiples_company_x = get_multiples(entity_id, grouped_dfs)
# get_multiples(4161767, grouped_dfs)


def get_multiples(entity_id, quarter, peer_groups_every_quarter,raw_data,multiple_type):
    #dataframe_multiples = pd.DataFrame(columns=['Quarter', 'Median Multiple', 'Mean Multiple','Std. Dev.'])
    
    # print(find_peer_group(entity_id,quarter))
    peer_group = find_peer_group(entity_id,quarter,peer_groups_every_quarter)

    # List of all multiples
    list_of_multiples = []

    # Get multiple for each company:
    for company in peer_group:
        multiple = raw_data[quarter][raw_data[quarter]["Entity ID "] == company][multiple_type].values[0]
        list_of_multiples.append(multiple) 
        
    new_multiple = round(np.median(list_of_multiples),2)

    return new_multiple           

def get_valuation_df (raw_data,peer_groups_every_quarter):
#get peer group companies' mean multiples as a separate dataframe to then be joined with valuation df
    
    valuation_df = {}

    for quarter in raw_data:
        peers = {"EV/EBIT AVG": [],"EV/EBITDA AVG":[],"EV/Sales AVG":[],"P/E AVG":[], "Index": []}
        
        print(quarter)
        for i in raw_data[quarter].index:
            for m in ["EV/EBIT","EV/EBITDA","EV/Sales","P/E"]:

                peers[m + " AVG"].append(get_multiples(raw_data[quarter].loc[i, "Entity ID "], raw_data[quarter].loc[i,"Quarter"],peer_groups_every_quarter,raw_data,m))
            
            peers["Index"].append(i)

        peers_df = pd.DataFrame(peers)
        peers_df = peers_df.set_index('Index')

        valuation_df[quarter] = pd.merge(raw_data[quarter],peers_df, left_index = True, right_index = True)
     
    return valuation_df

def old_trading_strategy (valuation_df):
    #define implied shareprices
    for quarter in valuation_df:
        valuation_df[quarter]["implied_shareprice_EV/EBIT"] = (valuation_df[quarter]["EV/EBIT AVG"] * valuation_df[quarter]["EBIT ($000)"] - valuation_df[quarter]["Net Debt ($000)"])/(valuation_df[quarter]["Market Cap"]/valuation_df[quarter]["Prices"])
        valuation_df[quarter]["implied_shareprice_EV/EBITDA"] = (valuation_df[quarter]["EV/EBITDA AVG"] * valuation_df[quarter]["EBITDA ($000)"] - valuation_df[quarter]["Net Debt ($000)"])/(valuation_df[quarter]["Market Cap"]/valuation_df[quarter]["Prices"])
        valuation_df[quarter]["implied_shareprice_EV/Sales"] = (valuation_df[quarter]["EV/Sales AVG"] * valuation_df[quarter]["Total Revenue ($000)"] - valuation_df[quarter]["Net Debt ($000)"])/(valuation_df[quarter]["Market Cap"]/valuation_df[quarter]["Prices"])
        valuation_df[quarter]["implied_shareprice_P/E"] = valuation_df[quarter]["P/E AVG"] * valuation_df[quarter]["Net Income ($000)"]/(valuation_df[quarter]["Market Cap"]/valuation_df[quarter]["Prices"])


        #calculations of final implied shareprices
        list_prices_income = ["implied_shareprice_EV/EBIT", "implied_shareprice_EV/EBITDA", "implied_shareprice_P/E"]
        list_prices_total = ["implied_shareprice_EV/EBIT", "implied_shareprice_EV/EBITDA", "implied_shareprice_EV/Sales", "implied_shareprice_P/E"]

        #data["income_sensitive_price"] = data.loc[:,list_prices_income].median(axis = 1)
        #data["bool_income_sensitive_price"] = np.where(data['income_sensitive_price'] < 0, 1, 0)

        valuation_df[quarter]["median_implied_shareprice"] = np.where(valuation_df[quarter].loc[:,list_prices_income].median(axis = 1) < 0, valuation_df[quarter]["implied_shareprice_EV/Sales"], valuation_df.loc[:,list_prices_total].median(axis = 1))
        valuation_df[quarter]["mean_implied_shareprice"] = np.where(valuation_df[quarter].loc[:,list_prices_income].median(axis = 1) < 0, valuation_df[quarter]["implied_shareprice_EV/Sales"], valuation_df.loc[:,list_prices_total].mean(axis = 1))

        #data["final_implied_shareprice"] = data["bool_income_sensitive_price"].apply(lambda x: data["implied_shareprice_EV/Sales"] if x == 1 else data.loc[:,list_prices_total].median(axis = 1))

        #signal generation
        valuation_df[quarter]["median_signals"] = np.where(valuation_df[quarter]["Prices"] < valuation_df[quarter]["median_implied_shareprice"], 1, 0)
        valuation_df[quarter]["mean_signals"] = np.where(valuation_df[quarter]["Prices"] < valuation_df[quarter]["mean_implied_shareprice"], 1, 0)
        
        #company-specific returns
        companies = valuation_df[quarter]["Entity ID "].unique()
        data_companies = {}
        returns = []
        
        for company in companies:
            status = np.where(companies==company)[0]/len(companies)
            #print('Trading strategy for company', company, status)
            #print(np.where(status == round(status, 0)*100 % 2 == 0))
            data_one_company = valuation_df[valuation_df['Entity ID '] == company].copy()
            
            #sort dataframe by quarter
            data_one_company.sort_values(by = ['Quarter'])
            
            #add company-specific return columns
            rets = data_one_company.loc[:,"Prices"].pct_change()
            data_one_company.loc[:,"median_returns"] = rets * data_one_company.loc[:,"median_signals"]
            data_one_company.loc[:,"mean_returns"] = rets * data_one_company.loc[:,"mean_signals"]
            
            #calculate cumulated returns (average of median and mean for simplicity)
            company_return = 0.5 * (float(data_one_company["median_returns"].cumsum().iloc[-1:].item()) + float(data_one_company["mean_returns"].cumsum().iloc[-1:].item()))
            returns.append(company_return)
            data_companies[company] = data_one_company
        
        #print(data_companies)
        
    return [data_companies, np.mean(returns)]

def trading_strategy (valuation_df):
    #define implied shareprices

    dataframes_to_concat = []
    for quarter in valuation_df:
        #concat all the dataframes in the dictionary
        dataframes_to_concat.append(valuation_df[quarter])
    data = pd.concat(dataframes_to_concat)


    data["implied_shareprice_EV/EBIT"] = (data["EV/EBIT AVG"] * data["EBIT ($000)"] - data["Net Debt ($000)"])/(data["Market Cap"]/data["Prices"])
    data["implied_shareprice_EV/EBITDA"] = (data["EV/EBITDA AVG"] * data["EBITDA ($000)"] - data["Net Debt ($000)"])/(data["Market Cap"]/data["Prices"])
    data["implied_shareprice_EV/Sales"] = (data["EV/Sales AVG"] * data["Total Revenue ($000)"] - data["Net Debt ($000)"])/(data["Market Cap"]/data["Prices"])
    data["implied_shareprice_P/E"] = data["P/E AVG"] * data["Net Income ($000)"]/(data["Market Cap"]/data["Prices"])


    #calculations of final implied shareprices
    list_prices_income = ["implied_shareprice_EV/EBIT", "implied_shareprice_EV/EBITDA", "implied_shareprice_P/E"]
    list_prices_total = ["implied_shareprice_EV/EBIT", "implied_shareprice_EV/EBITDA", "implied_shareprice_EV/Sales", "implied_shareprice_P/E"]

    #data["income_sensitive_price"] = data.loc[:,list_prices_income].median(axis = 1)
    #data["bool_income_sensitive_price"] = np.where(data['income_sensitive_price'] < 0, 1, 0)

    data["median_implied_shareprice"] = np.where(data.loc[:,list_prices_income].median(axis = 1) < 0, data["implied_shareprice_EV/Sales"], data.loc[:,list_prices_total].median(axis = 1))
    data["mean_implied_shareprice"] = np.where(data.loc[:,list_prices_income].median(axis = 1) < 0, data["implied_shareprice_EV/Sales"], data.loc[:,list_prices_total].mean(axis = 1))

        #data["final_implied_shareprice"] = data["bool_income_sensitive_price"].apply(lambda x: data["implied_shareprice_EV/Sales"] if x == 1 else data.loc[:,list_prices_total].median(axis = 1))

        #signal generation
    data["median_signals"] = np.where(data["Prices"] < data["median_implied_shareprice"], 1, 0)
    data["mean_signals"] = np.where(data["Prices"] < data["mean_implied_shareprice"], 1, 0)
        
        #company-specific returns
    companies = data["Entity ID "].unique()
    data_companies = {}
    returns = []
        
    for company in companies:
        status = np.where(companies==company)[0]/len(companies)
        print('Trading strategy for company', company, status)

            
            #print(np.where(status == round(status, 0)*100 % 2 == 0))
        data_one_company = data[data['Entity ID '] == company].copy()
        #print(data_one_company)

            #sort dataframe by quarter
        data_one_company = data_one_company.sort_values(by = ['Quarter'])
            
        #print(data_one_company)

            #add company-specific return columns
        rets = data_one_company["Prices"].pct_change()

        data_one_company["median_returns"] = rets * data_one_company.loc[:,"median_signals"]


        #print( "ONE: ", rets * data_one_company.loc[:,"median_signals"])
        #print(rets)


        data_one_company["mean_returns"] = rets * data_one_company.loc[:,"mean_signals"]


        #print( "TWO: ",rets * data_one_company.loc[:,"mean_signals"])


            
            #calculate cumulated returns (average of median and mean for simplicity)
        company_return = 0.5 * (float(data_one_company["median_returns"].cumsum().iloc[-1:].item()) + float(data_one_company["mean_returns"].cumsum().iloc[-1:].item()))
        returns.append(company_return)
        data_companies[company] = data_one_company
        
    '''
    dataframes_to_concat_companies = []
    for company in valuation_df:
        #concat all the dataframes in the dictionary
        dataframes_to_concat_companies.append(valuation_df[quarter])
    df_companies = pd.concat(dataframes_to_concat_companies)
    '''

        
        #print(data_companies)
        
    return [data_companies, np.mean(returns)]
        
def optimization (data, columns_to_cluster, weights):
    
    return_results = []

    #loop through possible weight combinations
    for w in weights:
        
        #optimization funciton within this foor loop
        
        primary_industry_clusters = first_clustering(data)
        primary_industries_clustered = clustering(primary_industry_clusters, columns_to_cluster, w)
        
        print('clustering done')

        # Group the Clusters to prep for output:
        grouped_dfs,peer_groups_every_quarter = group_output(primary_industries_clustered)
        
        print('grouping done')
        
        
        print('optimization phase: weights', w)
        #form valuation_df
        valuation_df = get_valuation_df(data, peer_groups_every_quarter)
        
        print('generating valuation df done')
        
        '''
        rets_w = trading_strategy(valuation_df)[1]
        return_results.append(rets_w)
        print(rets_w)

    index_max = np.argmax(return_results)
    
    return ('Weight combination:', weights[index_max], 'Total average return:', return_results[index_max])
    '''
    return valuation_df


#function application
columns_to_cluster = ['Total Assets ($000)', 'Debt-to-assets', 'Cash-to-assets', 'EBITDA margin','Total Revenue ($000)', 'Beta',  'Enterprise value', 'EV/Sales', 'EV/EBITDA', 'P/E', 'Global Region ']

combinations = [[0.1333333333, 0.1333333333, 0.1333333333, 0.0, 0.0, 0.12, 0.12, 0.12, 0.12, 0.12, 0.0]]



raw_data = input(r"C:\Users\Dani\OneDrive - Wirtschaftsuniversität Wien - IT-SERVICES\Dokumente\WU\Bachelorarbeit\02 Code\02 Data Processing\clustering_data.csv")

#only first 1000 rows for testing purposes
#test_nr_industries = int(len(pd.read_csv("[testing]_complete_input.csv").iloc[:1000,:]["Primary Industry "].unique()))

#since the # of industries was hardcoded into the functions before, I paramaterized it to make it more dynamic
#raw_raw = pd.read_csv("[testing]_complete_input.csv")
#raw_raw = raw_raw[raw_raw["exticker"].isin(list(raw_raw["exticker"].unique())[:500])]
#nr_industries = int(len(raw_raw["Primary Industry "].unique()))

#perform initial clustering
#primary_industry_clusters = first_clustering(raw_data)
#display(primary_industry_clusters)

#run second clustering, grouping of dataframes, multiple calculation, forming of valuation dataframe, 
#generating tradings trategy signals and optimize over different weight combinations based on returns
valuation_df = optimization(raw_data, columns_to_cluster, combinations)
#valuation_df.to_csv('[testing]_valuation_df.csv')
dict_companies = trading_strategy(valuation_df)[0]

dict_rets = {}
quarters = []
for company in dict_companies:
    temp_rets = []
    df_one_company = dict_companies[company]
    df_one_company['median_cumret'] = (1+df_one_company["median_returns"]).cumprod() - 1
    df_one_company['mean_cumret'] = (1+df_one_company["mean_returns"]).cumprod() - 1
    #print(df_one_company[["Quarter", "mean_cumret"]])
    
    #dict_rets[company] = (list(0.5 * (df_one_company['median_cumret'] + df_one_company['mean_cumret'])))
    dict_rets[company] = list((df_one_company['mean_cumret']))
    quarters = list(df_one_company["Quarter"].unique())
    
def average(l):
    llen = len(l)
    def divide(x): return x / llen
    return list(map(divide, map(sum, zip(*l))))

all_rets = list(dict_rets.values())
#all_rets_arr = np.array(all_rets)

for i in all_rets:
    if len(i) != 20:
        print(i, 'index:', all_rets.index(i))
        
#all_rets = all_rets[:1376] + all_rets[1377:]

avg_list = average(all_rets)
print(len(avg_list))

clustering_ret_df = pd.DataFrame()
clustering_ret_df.to_csv("clustering_returns.csv")
clustering_ret_df["quarter"] = quarters
clustering_ret_df["returns"] = avg_list
clustering_ret_df.plot()

#!pip install pandas-datareader
#from pandas_datareader.data as web
#from datetime import datetime

#start = datetime(2017, 1, 1)
#end = datetime(2022, 1, 1)
#sp = web.DataReader("INPX", "morningstar", start = start, end = end)
    




