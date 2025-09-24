# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:41:20 2024

@author: Dani
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import random
import warnings
import time
import concurrent.futures
warnings.filterwarnings('ignore')


# METHOD TO GET PEER GROUP FOR A CERTAIN COMPANY IN A CERTAIN QUARTER
# INPUT: GV_Key of company to find and quarter in which to look as well as the peer_groups_every_quarter from previous method (Dataframe of all peer groups in a quarter)
# OUTPUT: List of Entity IDs of peers

# EXAMPLE: (entity_id of company to look for)
# SYNTAX: find_peer_group(entity_id, quarter)
# find_peer_group(4161767,"FQ12019")

def find_peer_group(company, quarter, peer_groups_every_quarter):
        df = peer_groups_every_quarter[quarter]
        for row in range(0, len(df)):
            if company in df.iloc[row, 1]:
                #print("Quarter: ", quarter, "      Peer Group:", df.iloc[row,1])
                return df.iloc[row,1]                
            
# METHOD TO GET THE MEDIAN MULTIPLE FOR EVERY QUARTER OF CERTAIN COMPANY
# INPUT: Company GV_Key as well as grouped_dfs from previous method (Dictionary with quarters and clusters)
# OUTPUT: Dataframe with quarters in one column and median and mean multiple for the corresponding quarter in columns

# EXAMPLE: Get the multiples for company x
# SYNTAX: multiples_company_x = get_multiples(GV_Key, grouped_dfs)
# get_multiples(4161767, grouped_dfs)


def get_multiples(company, quarter, peer_groups_every_quarter,raw_data,multiple_type):
    #dataframe_multiples = pd.DataFrame(columns=['Quarter', 'Median Multiple', 'Mean Multiple','Std. Dev.'])
    
    # print(find_peer_group(entity_id,quarter))
    peer_group = find_peer_group(company,quarter,peer_groups_every_quarter)

    # List of all multiples
    list_of_multiples = []

    # Get multiple for each company:
    for comp in peer_group:
        multiple = raw_data[quarter][raw_data[quarter]["Company Name"] == comp][multiple_type].values[0]
        list_of_multiples.append(multiple) 
        
    new_multiple = round(np.median(list_of_multiples),2)

    return new_multiple           

def get_valuation_df(raw_data, peer_groups_every_quarter):
    valuation_df = {}
    
    for quarter, data in raw_data.items():
        peers = {"P/EBIT AVG": [], "P/EBITDA AVG": [], "P/Sales AVG": [], "P/E AVG": [], "P/B AVG": [], "Index": []}
        
        print(quarter)
        counter_outliers = 0
        counter_companies = 0
        start_timer = time.time()
        
        for i, row in data.iterrows():
            company = row["Company Name"]
            quarter = row["Quarter"]
            
            for m in ["P/EBIT", "P/EBITDA", "P/Sales", "P/E", "P/B"]:
                try:
                    mult_value = get_multiples(company, quarter, peer_groups_every_quarter, raw_data, m)
                    peers[m + " AVG"].append(mult_value)
                    counter_companies += 1
                except TypeError:
                    peers[m + " AVG"].append(np.nan)
                    counter_outliers += 1
                    continue
            
            peers["Index"].append(i)
        
        peers_df = pd.DataFrame(peers)
        peers_df = peers_df.set_index('Index')

        valuation_df[quarter] = pd.merge(data, peers_df, left_index=True, right_index=True)
        
        end_timer = time.time()
        print(f"{round(counter_outliers / counter_companies * 100, 1)}% outliers in quarter {quarter}, {counter_companies / 5} companies in total!")
        print(f"Quarter {quarter} took {round((end_timer - start_timer) / 60, 1)} minutes!")
        

    return valuation_df



def trading_strategy(valuation_df):
    #define implied shareprices

    dataframes_to_concat = []
    for quarter in valuation_df:
        #concat all the dataframes in the dictionary
        dataframes_to_concat.append(valuation_df[quarter])
    data = pd.concat(dataframes_to_concat)
    data.dropna(inplace = True)


    data["implied_shareprice_P/EBIT"] = data["P/EBIT AVG"] * data["EBIT"] / data["Shares_out"]
    data["implied_shareprice_P/EBITDA"] = data["P/EBITDA AVG"] * data["EBITDA"] / data["Shares_out"]
    data["implied_shareprice_P/Sales"] = data["P/Sales AVG"] * data["Revenue"] / data["Shares_out"]
    data["implied_shareprice_P/E"] = data["P/E AVG"] * data["Net Income"] / data["Shares_out"]
    data["implied_shareprice_P/B"] = data["P/B AVG"] * data["Stockholders Equity"] / data["Shares_out"]


    #calculations of final implied shareprices
    list_prices_income = ["implied_shareprice_P/EBIT", "implied_shareprice_P/EBITDA", "implied_shareprice_P/E"]
    list_prices_realistic = ["implied_shareprice_P/Sales", "implied_shareprice_P/B"]
    list_prices_total = ["implied_shareprice_P/EBIT", "implied_shareprice_P/EBITDA", "implied_shareprice_P/Sales", "implied_shareprice_P/E", "implied_shareprice_P/B"]

    #data["median_implied_shareprice"] = np.where(data.loc[:,list_prices_income].median(axis = 1) < 0, data["implied_shareprice_P/Sales"], data.loc[:,list_prices_total].median(axis = 1))
    #data["mean_implied_shareprice"] = np.where(data.loc[:,list_prices_income].mean(axis = 1) < 0, data["implied_shareprice_P/Sales"], data.loc[:,list_prices_total].mean(axis = 1))

    #more aggressive alternative
    
    data["realistic_implied_shareprice"] = np.where(data.loc[:,list_prices_income].median(axis = 1) < 0, data.loc[:,list_prices_realistic].median(axis = 1), data.loc[:,list_prices_total].median(axis = 1))
    data["bull_implied_shareprice"] = np.where(data.loc[:,list_prices_income].mean(axis = 1) < 0, data.loc[:,list_prices_total].mean(axis = 1), data.loc[:,list_prices_total].max(axis = 1))

    
        #signal generation
    data["realistic_signals"] = np.where(data["Share_Price"] < (data["realistic_implied_shareprice"]), 1, 0)
    data["bull_signals"] = np.where(data["Share_Price"] < (1.1 * data["bull_implied_shareprice"]), 1, 0)
        
        #company-specific returns
    companies = data["Company Name"].unique()
    data_companies = {}
    data_quarters_df_list = []
    returns = []
        
    for company in companies:
        status = np.where(companies==company)[0]/len(companies)
        print('Trading strategy for company', company, status)

            
            #print(np.where(status == round(status, 0)*100 % 2 == 0))
        data_one_company = data[data['Company Name'] == company].copy()
        #print(data_one_company)

            #sort dataframe by quarter
        data_one_company = data_one_company.sort_values(by = ['Quarter'])
            
        #print(data_one_company)

            #add company-specific return columns
        rets = data_one_company["Share_Price"].pct_change()

        data_one_company["realistic_returns"] = rets * data_one_company.loc[:,"realistic_signals"]


        #print( "ONE: ", rets * data_one_company.loc[:,"median_signals"])
        #print(rets)


        data_one_company["bull_returns"] = rets * data_one_company.loc[:,"bull_signals"]
        data_one_company["benchmark_returns"] = rets * 0.9


        #print( "TWO: ",rets * data_one_company.loc[:,"mean_signals"])


            
        #calculate cumulated returns (average of median and mean for simplicity)
        company_return = 0.5 * (float(((1 + data_one_company["realistic_returns"]).cumprod()-1).iloc[-1].item()) + float(((1 + data_one_company["bull_returns"]).cumprod()-1).iloc[-1].item()))
        returns.append(company_return)
        data_companies[company] = data_one_company
        
        #concatenate to one single dataframe per quarter, containing the company names and the returns
        data_quarters_df_list.append(data_one_company)
    
    data_quarters_df = pd.concat(data_quarters_df_list)
        
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
        
    return data_quarters_df, data_companies, np.mean(returns)

def return_calc(data_quarters_df):
    
    returns_per_quarter = {}
    data_quarters_df.sort_values(by = "Quarter", axis = 0, ascending = True, inplace = True)
    
    for quarter in data_quarters_df["Quarter"].unique().tolist()[1:]:
        
        each_quarter = data_quarters_df[data_quarters_df["Quarter"] == quarter]
        return_data = each_quarter.iloc[:,-3:]
        
        #NA removal
        na_rows = return_data[return_data.isna().any(axis=1)].index
        print("Nr of NAs:", len(na_rows))
        return_data.dropna(axis = 0, inplace = True)
        
        mean_ret = np.nanmean(np.array(return_data["bull_returns"].tolist()))
        median_ret = np.nanmean(np.array(return_data["realistic_returns"].tolist()))
        bench_ret = np.nanmean(np.array(return_data["benchmark_returns"].tolist()))
        returns_per_quarter[quarter] = [quarter, mean_ret, median_ret, bench_ret]
        
        
    returns_per_quarter_df = pd.DataFrame.from_dict(returns_per_quarter, orient = 'index')
    returns_per_quarter_df.columns = ["Quarter", "bull_returns", "realistic_returns", "benchmark_returns"]
    
    returns_per_quarter_df.sort_values(by = "Quarter", axis = 0, ascending = True, inplace = True)
    returns_per_quarter_df["bull_returns_cum"] = (1 + returns_per_quarter_df["bull_returns"]).cumprod() - 1
    returns_per_quarter_df["realistic_returns_cum"] = (1 + returns_per_quarter_df["realistic_returns"]).cumprod() - 1
    returns_per_quarter_df["benchmark_returns_cum"] = (1 + returns_per_quarter_df["benchmark_returns"]).cumprod() - 1
    
    return returns_per_quarter_df
    
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns given the periods per year
    """
    compounded_growth = (1+r).prod() 
    n_periods = r.shape[0] 
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns given the periods per year
    """
    return r.std()*(periods_per_year**0.5)

#sharpe ratio = annual excess returns / annual excess returns volatility
def sr_calc(returns):
    ex_ret = returns - 0.02 #assumptions for risk free rate; optional: return_df["BIL"]
    # calculate total excess return
    totexret = (ex_ret + 1).prod() - 1
    # annualize it
    annexret = (1 + totexret)**(1/4) - 1
    # calculate annualized volatility
    annexvol = (ex_ret.std()) * np.sqrt(4)
    # calculate the Sharpe Ratio
    SR = annexret / annexvol
    return SR
    
        
'''
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
        
        rets_w = trading_strategy(valuation_df)[1]
        return_results.append(rets_w)
        print(rets_w)

    index_max = np.argmax(return_results)
    
    return ('Weight combination:', weights[index_max], 'Total average return:', return_results[index_max])
   
    return valuation_df

'''