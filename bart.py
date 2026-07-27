#%% imports
"""
Created on Fri May 16 12:32:18 2025
similarity tests for nba draft prospects based on NCAA stats
@author: Subramanya.Ganti
"""

import numpy as np
import pandas as pd

import itertools
import scipy.stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.optimize import fsolve
from scipy.stats import zscore

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import requests
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('mode.chained_assignment', None)

path = "C:/Users/uttam/Desktop/Sports/basketball/bart"
#path = "C:/Users/Subramanya.Ganti/Downloads/Sports/basketball/bart"

#%% choose season to run
latest_season = 2026

#%% team ratings
def team_ratings():
    i = 2008; team_ranking = []
    while(i<latest_season+1):
        team = pd.read_csv(f'{path}/team/{i}_team_results.csv')
        if(i<2023):
            old_column_names = team.columns.to_list()
            old_column_names[-1:] = old_column_names[-1].split(', ')
            team = team.reset_index()
            team.columns = old_column_names
            #team = team[['rank','season','de Rank','Con Rec.']]
            #team.rename(columns = {'rank':'team','de Rank':'barthag','Con Rec.':'sos'}, inplace = True)
        
        team['season'] = i
        team = team[['team','season','barthag','sos','adjoe','adjde','Opp OE','Opp DE']]
        team_ranking.append(team)
        i+=1
    
    team_ranking = pd.concat(team_ranking)
    #team_ranking.rename(columns = {'rank':'team','de Rank':'rating'}, inplace = True)
    team_ranking.rename(columns = {'barthag':'rating'}, inplace = True)
    #team_ranking = pd.pivot_table(team_ranking,values=['rating'],index=['team'],columns=['season'],aggfunc=np.sum)
    #team_ranking.columns = team_ranking.columns.droplevel(level=0)
    return team_ranking
    
team_ranking = team_ratings()
#team_ranking['adj_rating'] = team_ranking['adjoe'] - team_ranking['adjde']
#team_ranking['adj_sos'] = team_ranking['Opp OE'] - team_ranking['Opp DE']
team_ranking['adj_rating'] = norm.ppf(team_ranking['rating'], loc=0.494, scale=0.255)
team_ranking['adj_sos'] = norm.ppf(team_ranking['sos'], loc=0.494, scale=0.13)
team_ranking = team_ranking[['team','season','adj_rating']]

#%% helper functions
def mins_per_season(season):
    url = "https://api.pbpstats.com/get-totals/nba"
    params = {
        "Season": f"{season}-" + str(season+1)[-2:],
        "SeasonType": "Regular Season",
        "Type": "Player"
    }
    response = requests.get(url, params=params, verify=False)
    response_json = response.json()
    player_stats = response_json["multi_row_table_data"]
    player_stats = pd.DataFrame.from_dict(player_stats)
    player_stats = player_stats[['Name','RowId','Minutes']]
    player_stats['season'] = season
    return player_stats

def hgt_age_adj(df,key):
    columns = ['TS%','usg', 'ORB%', 'DRB%', 'AST%', 'TO%', 'ast/tov', 'ftr','FT%', 'dunkar','rimar', 'rim%', 'midar', 'mid%', 
               'bpm','bpm_adj', 'ORtg','drtg','mp', '3par','3P%_regressed','blk_share','stl_share','midasst']
    
    for f in columns:
        if(key=='hgt'): 
            df[key] = df[key].fillna(60)
            df[f'{key}_{f}'] = df[key] / df[f]
        if(key=='age'): 
            df[key] = df[key].fillna(25)
            df[f'{key}_{f}'] = df[key] / df[f]
        
        df = norm_inv_column(df,f'{key}_{f}')
    
    return df

def data_imputation_08_09(df):
    imputer = IterativeImputer()
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed, columns=df.columns)
    return df_imputed

def df_class(df):
    mapping = {'Pro': 5, 'Sr': 4, 'Jr': 3, 'So': 2, 'Fr': 1, 'HS': 0, '--': np.nan}
    df['class'] = df['class'].map(mapping)
    df = df[df['class'].notna()]
    #df['class'] = stats.zscore(df['class'])
    return df

def df_role(df):
    mapping = {'Pure PG': 1, 'Combo G': 1.5, 'Wing F': 2, 'Wing G': 1.75, 'Stretch 4': 2.5, 'Scoring PG': 1.25, 'PF/C': 2.75, 'C': 3}
    df['role'] = df['role'].map(mapping)
    #df = df[df['role'].notna()]
    return df

def height_adj(df):
    df[['Feet', 'Inches']] = df['height'].str.split('-', expand=True)
    #df = df[df['Feet'].notna()]
    #df = df[df['Inches'].notna()]
    df['Feet'] = pd.to_numeric(df['Feet'], errors='coerce', downcast='integer')
    df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce', downcast='integer')
    df['hgt'] = 12*df['Feet'] + df['Inches'] #.astype(int)
    df = df[df['hgt']>=60]
    return df

def height_based_roles(df):
    #gaussian roles
    import scipy.stats as stats
    df['hgt'] = df['hgt'].fillna(60) #verify this
    df['short'] = stats.norm.cdf(72, loc=df["hgt"], scale=1) #np.where(df['hgt']<72,1,0)
    df['guard'] = stats.norm.cdf(78, loc=df["hgt"], scale=1) - stats.norm.cdf(72, loc=df["hgt"], scale=1)
    df['wing'] = stats.norm.cdf(82, loc=df["hgt"], scale=1) - stats.norm.cdf(76, loc=df["hgt"], scale=1)
    df['big'] = 1 - stats.norm.cdf(82, loc=df["hgt"], scale=1)
    return df

def mins_based_roles(df):
    #gaussian roles
    import scipy.stats as stats
    df['mp'] = df['mp'].fillna(0) #verify this
    df['reserve'] = stats.norm.cdf(10, loc=df["mp"], scale=5)
    df['bench'] = stats.norm.cdf(18, loc=df["mp"], scale=5) - stats.norm.cdf(10, loc=df["mp"], scale=5)
    df['sub'] = stats.norm.cdf(28, loc=df["mp"], scale=3) - stats.norm.cdf(18, loc=df["mp"], scale=3)
    df['starter'] = 1 - stats.norm.cdf(28, loc=df["mp"], scale=2)
    return df
    
def log_adjust(df,category):
    df[category] = np.log(df[category])
    df[category] = df[category].replace(-np.inf, np.nan)
    min_value = df[category].min(skipna=True)
    df[category] = df[category].replace(np.nan, min_value)
    return df

def iqr_column(df,category):
    df[category] = (df[category]-df[category].quantile(0.5))/(df[category].quantile(0.75) - df[category].quantile(0.25))
    return df

def norm_inv_column(df,category):
    df[category] = df[category].rank(pct=True)
    df[category] = df[category].clip(upper=.999,lower=0.001)
    df[category] = norm.ppf(df[category])
    return df

def international_stats_adjustments():
    df = pd.read_excel(f'{path}/player/foreign_players.xlsx','final')
    df['TS%'] = df['TS%']*100
    df['role'] = np.nan
    df['3par'] = df['3par'].clip(upper=1)
    df['intl'] = 1    
    return df

def pre_08_ncaa():
    df = pd.read_excel(f'{path}/player/foreign_players.xlsx','ncaa')
    #darren yates 2004 season has a ast/tov of infinity
    df['ast/tov'] = df['ast/tov'].replace(np.inf, 11)
    df['TS%'] = df['TS%']*100
    return df

def bpm_estimate(df,intl):
    df['2par'] = 1 - df['3par']
    df['2P%'] = (((df['TS%']/100) * (1+0.44*df['ftr']) * 2)-(df['ftr']*df['FT%'] + 3*df['3par']*df['3P%']))/(2*df['2par'])

    df['bpm'] = -1.618 + \
                +0.098 * df['mp'] + \
                -0.023 * df['usg'] + \
                -0.088 * df['TS%'] + \
                +0.090 * df['ORB%'] + \
                -0.017 * df['DRB%'] + \
                +0.044 * df['AST%'] + \
                +0.064 * df['TO%'] + \
                +0.220 * df['ast/tov'] + \
                +0.389 * df['BLK%'] + \
                +0.624 * df['STL%'] + \
                -0.146 * df['ftr'] + \
                -1.238 * df['FT%'] + \
                +2.365 * df['3par'] + \
                +0.577 * df['3P%'] + \
                +0.246 * df['ORtg'] + \
                -0.241 * df['drtg']
    #df['bpm'] = np.nan
    #df['bpm_adj'] = 0.41*df['bpm'] + 0.019*df['ORtg'] - 0.019*df['drtg']
    df['bpm_adj'] = 0.472*df['bpm'] + 0.028*df['ORtg'] + 0.073*df['drtg'] - 10.547
    #if(intl == 1): df['bpm_adj'] += 0.25
    df[['dunkasst','rimasst','midasst','3Passt','Rec Rank']] = np.nan
    return df

def height_estimate(df):
    df['hgt'] = +74.0115813626499 + \
                +0.02421394948148  * df['TS%']  + \
                +0.15449112263466  * df['ORB%'] + \
                +0.17303148969203  * df['DRB%'] + \
                -0.15116061217808  * df['AST%'] + \
                +0.34352812510471  * df['BLK%'] + \
                -0.95291522500520  * df['3par']
    return df

def plot_histograms(df):
    df.hist(figsize=(10, 8), bins=50)  # Adjust figsize as needed
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()

#%% player classification
def extract_player_stats():
    headers = pd.read_csv(f'{path}/player/header.csv')
    internationals = international_stats_adjustments()
    pre_2008 = pre_08_ncaa()
    
    #estimate the height and bpm for player stats pulled from RealGM
    internationals = bpm_estimate(internationals,1)
    pre_2008 = bpm_estimate(pre_2008,0)
    #pre_2008 = height_estimate(pre_2008)
    
    i = max(latest_season-15,2003) #verify how many seasons needed for consistency
    p_stats = []; unadj_p_stats = []
    while(i<latest_season+1):
        print(i,"player stats extracted")
        if(i < 2008):
            data_adj = pre_2008[pre_2008['season']==i]
        else:
            data = pd.read_csv(f'{path}/player/{i}.csv', names=headers.columns)
            
            if(i>2009):
                data_shot_type = pd.read_csv(f'{path}/player/{i}_shot_types.csv')
                data = data.merge(data_shot_type, on=['pid','player','team'], how='left')
                data[['dunkasst','rimasst','midasst','3Passt']] = data[['dunkasst','rimasst','midasst','3Passt']].replace(0, 1) #np.nan
            else:
                data[['dunkasst','rimasst','midasst','3Passt']] = np.nan
            
            data['pick'] = data['pick'].fillna(61)
            #data['pick'] = np.log(data['pick'])
            data['blocks']  = data['blk'] * data['GP']
            data['steals']  = data['stl'] * data['GP']
            data['minutes']  = data['mp'] * data['GP']
            team_bs = pd.pivot_table(data,values=['blocks','steals','minutes'],index=['team'],aggfunc=np.sum)
            team_bs['minutes'] = team_bs['minutes']/200
            team_bs['blocks'] = team_bs['blocks']/team_bs['minutes']
            team_bs['steals'] = team_bs['steals']/team_bs['minutes']
            team_bs = team_bs.reset_index()
            data = data.merge(team_bs, left_on='team', right_on='team')
            
            data = data.loc[(data['mp']>=5) & (data['GP']>=5)] #2, 2 is the default filter
            data['blk_share'] = (data['blk']*40/data['mp'])/data['blocks_y']
            data['stl_share'] = (data['stl']*40/data['mp'])/data['steals_y']
            #data = df_class(data)
            data = height_adj(data)
            data['season'] = i
            
            data['FG/mp'] = (data['2PA'] + data['3PA']) / (data['mp'] * data['GP'])
            data['dunkar'] = data['dunkFGA']/(data['2PA']+data['3PA'])
            data['rimFGA'] = data['rimFGA'] - data['dunkFGA']
            data['rimFG'] = data['rimFG'] - data['dunkFG']
            data['rim%'] = data['rimFG']/(data['rimFGA'])       
            data['rimar'] = data['rimFGA']/(data['2PA']+data['3PA'])
            data['midar'] = data['midFGA']/(data['2PA']+data['3PA'])
            data['3par'] = data['3PA']/(data['2PA']+data['3PA'])
            data['ftr'] = data['FTA']/(data['2PA']+data['3PA'])

            data['3par'] = data['3par'].clip(upper=1)
            data['2par'] = 1 - data['3par']
            data['2P%'] = (((data['TS%']/100) * (1+0.44*data['ftr']) * 2)-(data['ftr']*data['FT%'] + 3*data['3par']*data['3P%']))/(2*data['2par'])
            data['age'] = ((pd.to_datetime(f'{i}-11-01', format="%Y-%m-%d") - pd.to_datetime(pd.Series(data['dob']), format="%Y-%m-%d"))/ np.timedelta64(1, 'D'))/365
            
            #age corrections
            data.loc[data['player']=='Allen Graves','age'] = (i-2026) + 20.276
            
            if(i>2009):
                data['dunkar'] = data['dunkar'].fillna(0)
                data['mid%'] = data['mid%'].fillna(0)
                data['rim%'] = data['rim%'].fillna(0)
                #data['dunk%'] = data['dunk%'].fillna(0)
                
            data['3P%'] = data['3P%'].fillna(0)
            data['2par'] = data['2par'].fillna(0)
            data['2P%'] = data['2P%'].fillna(0)
            data['Rec Rank'] = data['Rec Rank'].fillna(0)
            
            data_adj = data[['player','pid','team','season','class','hgt','GP','mp','usg','TS%','ORB%','DRB%','AST%','TO%','ast/tov',
                             'BLK%','blk_share','STL%','stl_share','pfr','ftr','FT%','dunkar','rimar','rim%','midar','mid%',
                             '3par','3P%','ORtg','drtg','bpm','age','2par','2P%','role','dunkasst','rimasst','midasst','3Passt','Rec Rank']]
        
        data_adj['intl'] = 0
        #team relative bpm
        data_adj['MIN'] = data_adj['mp'] * data_adj['GP']
        data_adj['bpm_MIN'] = data_adj['bpm'] * data_adj['mp'] * data_adj['GP']
        bpm_team = data_adj.pivot_table(index=['team','season'],values=['bpm_MIN','MIN'], aggfunc='sum')
        bpm_team['team_bpm'] = bpm_team['bpm_MIN']/bpm_team['MIN']
        bpm_team = bpm_team.reset_index()
        data_adj = data_adj.merge(bpm_team[['team','season','team_bpm']], on=['team','season'], how='left')
        data_adj['bpm_adj'] = data_adj['bpm'] - data_adj['team_bpm']
        
        #add internationals data
        data_adj = pd.concat([data_adj, internationals[internationals['season']==i]])
        #upper and lower limits for shooting percentages
        data_adj.loc[data_adj['2P%']>1,'2P%'] = 1
        data_adj.loc[data_adj['2P%']<0,'2P%'] = 0
        data_adj.loc[data_adj['3P%']>1,'3P%'] = 1
        data_adj.loc[data_adj['3P%']<0,'3P%'] = 0
        data_adj.loc[data_adj['mid%']>1,'mid%'] = 1
        data_adj.loc[data_adj['mid%']<0,'mid%'] = 0
        data_adj.loc[data_adj['rim%']>1,'rim%'] = 1
        data_adj.loc[data_adj['rim%']<0,'rim%'] = 0
        #data_adj.loc[data_adj['dunk%']>1,'dunk%'] = 1
        #data_adj.loc[data_adj['dunk%']<0,'dunk%'] = 0

        data_adj['3par'] = data_adj['3par'].fillna(0)
        data_adj['ftr'] = data_adj['ftr'].clip(upper=3)
        data_adj['dunkar'] = data_adj['dunkar'].clip(upper=1)
        data_adj['rimar'] = data_adj['rimar'].clip(upper=1)
        data_adj['midar'] = data_adj['midar'].clip(upper=1)
        
        data_adj['3prof'] = data_adj['3P%']*(2/(1+np.exp(-4*data_adj['3par']))-1)
        data_adj['rimprof'] = data_adj['rim%']*(2/(1+np.exp(-4*data_adj['rimar']))-1)
        data_adj['midprof'] = data_adj['mid%']*(2/(1+np.exp(-4*data_adj['midar']))-1)
        
        #height based roles
        data_adj = height_based_roles(data_adj)
        data_adj = mins_based_roles(data_adj)
        
        #regress 3P% to the mean
        #data_adj['ftr'] = data_adj['ftr'].fillna(0)
        regression_subset = data_adj[(data_adj['mp']>5)&(data_adj['GP']>5)]
        X_train, X_test, y_train, y_test = train_test_split(regression_subset[['hgt','ast/tov','FT%','3par']], regression_subset['3P%'], test_size=0.2, random_state=42)
        model = Ridge(alpha=1)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        model.fit(X_train_scaled, y_train)
        #print("Random forest for 3P% R-squared",r2_score(y_test, model.predict(X_test_scaled)))
        data_adj['3p_regress_val'] = model.predict(scaler.fit_transform(data_adj[['hgt','ast/tov','FT%','3par']]))
        data_adj['3P%_regressed'] = (data_adj['3P%']*data_adj['3par']*data_adj['mp']*data_adj['GP'] + data_adj['3p_regress_val']*500)/ (data_adj['3par']*data_adj['mp']*data_adj['GP'] + 500)
        #data_adj['3P%_regressed'] = np.where(data_adj['3par']<0.01,0,data_adj['3P%_regressed'])
        
        #interaction effects
        data_adj['feel'] = data_adj['FT%'].rank(pct=True)*data_adj['ast/tov'].rank(pct=True) #*data_adj['hgt']
        data_adj['bds'] = data_adj['usg'].rank(pct=True)-data_adj['TO%'].rank(pct=True)
        data_adj['jimmy'] = (data_adj['ORB%'].rank(pct=True) + data_adj['STL%'].rank(pct=True) - data_adj['TO%'].rank(pct=True))
        data_adj['stocks'] = (data_adj['BLK%']+data_adj['STL%'])/data_adj['hgt']
        data_adj['oreb_share'] = data_adj['ORB%'].rank(pct=True)/(data_adj['DRB%'].rank(pct=True)+data_adj['ORB%'].rank(pct=True))
        data_adj['TS_ast'] = data_adj['TS%'].rank(pct=True)-data_adj['AST%'].rank(pct=True)
        
        data_adj = hgt_age_adj(data_adj,'hgt')
        data_adj = hgt_age_adj(data_adj,'age')
        
        unadj_p_stats.append(data_adj.copy())
        
        for x in ['ORB%','DRB%','BLK%','blk_share','STL%','stl_share','AST%','TO%','ast/tov',
                  'bds','TS_ast','stocks']: #'oreb_share','hgt_3par','feel','jimmy'
            data_adj = log_adjust(data_adj,x)
        for x in ['usg','ftr','rimar','midar','3par','rim%','mid%','pfr','3P%','3P%_regressed','2P%',
                  '3prof','2par','rimprof','midprof','ORtg','drtg','bpm','bpm_adj']:
            data_adj = iqr_column(data_adj,x)
        for x in ['mp','GP','dunkar','dunkasst','rimasst','midasst','3Passt','FT%',
                  'feel','guard','wing','big','oreb_share']:
            data_adj = norm_inv_column(data_adj,x)
        p_stats.append(data_adj)
        i+=1
        
    p_stats = pd.concat(p_stats)
    unadj_p_stats = pd.concat(unadj_p_stats)
    p_stats = df_class(p_stats.copy())
    unadj_p_stats = df_class(unadj_p_stats.copy())
    p_stats = df_role(p_stats)
    unadj_p_stats = df_role(unadj_p_stats)
    
    #p_stats['hgt'] = p_stats['hgt'] - 60
    p_stats = iqr_column(p_stats,'hgt')
    p_stats = log_adjust(p_stats,'class')
    p_stats = iqr_column(p_stats,'age')
    p_stats = log_adjust(p_stats,'role')    
    return p_stats,unadj_p_stats

data,player_stats = extract_player_stats()
player_stats['mp_p'] = player_stats['mp'].rank(pct=True)
data.reset_index(drop=True,inplace=True)
player_stats.reset_index(drop=True,inplace=True)

data = data.merge(team_ranking, left_on=['team','season'], right_on=['team','season'], how='left')
data['adj_rating_50'] = np.where((data['adj_rating']<=0.5)&(data['intl']==0),1,0)
#data.drop(['rating','sos'], axis=1, inplace=True)

#%% fix pids of players in both real GM and bart torvik
#player_stats.loc[(player_stats['pid']<0),'intl'] = 1
player_stats['intl'] = player_stats['intl'].fillna(0)

mapping = pd.read_excel(f'{path}/nba_stats.xlsx','mapping DARKO')
mapping = mapping[['pid','pid2']]
mapping = mapping.rename(columns={'pid': 'pid2', 'pid2': 'pid'})
mapping = mapping.dropna()
player_stats = player_stats.merge(mapping, left_on='pid', right_on='pid', how='left')

player_stats['pid'] = player_stats['pid2'].fillna(player_stats['pid'])
del player_stats['pid2']; del mapping

#international players
player_stats.loc[((player_stats['player']=='Dame Sarr')&(player_stats['pid']==-190807)),'pid'] = 133946
player_stats.loc[((player_stats['player']=='Hannes Steinbach')&(player_stats['pid']==-218597)),'pid'] = 135114
player_stats.loc[((player_stats['player']=='Mario Saint-Supery')&(player_stats['pid']==-195656)),'pid'] = 134317
player_stats.loc[((player_stats['player']=='Neoklis Avdalas')&(player_stats['pid']==-183446)),'pid'] = 134716
player_stats.loc[((player_stats['player']=='Tomislav Ivisic')&(player_stats['pid']==-139509)),'pid'] = 127979
player_stats.loc[((player_stats['player']=='Zvonimir Ivisic')&(player_stats['pid']==-139511)),'pid'] = 78304
player_stats.loc[((player_stats['player']=='David Mirkovic')&(player_stats['pid']==-188962)),'pid'] = 135431
player_stats.loc[((player_stats['player']=='Johann Gruenloh')&(player_stats['pid']==-183444)),'pid'] = 135586
player_stats.loc[((player_stats['player']=='This De Ridder')&(player_stats['pid']==-139197)),'pid'] = 134573
player_stats.loc[((player_stats['player']=='Ivan Kharchenkov')&(player_stats['pid']==-165587)),'pid'] = 133778
player_stats.loc[((player_stats['player']=='Aday Mara')&(player_stats['pid']==-177830)),'pid'] = 78229

#%% aggregate weighted career level data instead of single season (not implemented yet)
def pivot_data(df,df2):
    mapping = {0:5, 1:5, 2:3, 3:1.5, 4:0.5}
    df['weight'] = df['class'].map(mapping)
    df['weight'] *= df['mp'] * df['GP']
    pivot = df.pivot_table(values=['age', 'hgt', 'mp', 'usg', 'TS%', 'ORB%', 'DRB%', 'AST%', 'TO%', 'ast/tov', 
                                   'BLK%', 'blk_share','STL%', 'stl_share', 'pfr', 'ftr', 'FT%',
                                   'dunkar', 'rimar', 'rim%','midar', 'mid%', '3par', '3P%','ORtg', 'drtg', 'bpm'],
                              index=['player','pid'], 
                              aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'weight']))

    gp = df.pivot_table(values=['GP'], index=['player','pid'], aggfunc="sum")
    team = df.pivot_table(values=['team'], index=['player','pid'], aggfunc=lambda x: ' '.join(x.unique()))

    team = team.reset_index()
    gp = gp.reset_index()
    pivot = pivot.reset_index()

    df0 = team.merge(gp, left_on=['player','pid'], right_on=['player','pid'])
    df0 = df0.merge(pivot, left_on=['player','pid'], right_on=['player','pid'])
    return df0

#career_stats = pivot_data(player_stats.copy(),data.copy())

#%% correlation matrix for all the stats under consideration
correl_columns = ['age', 'class', 'hgt', 'usg', 'ORB%', 'DRB%', 'AST%', 'TO%', 'ast/tov', 'blk_share','stl_share', 'ftr','FT%', 
                  'dunkar','rimar', 'rim%', 'midar', 'mid%', '3par', '3P%_regressed', 'bpm','ORtg','drtg','mp',
                  'role','rimasst','midasst','bpm_adj',
                  'feel','stocks','oreb_share','adj_rating_50']
                  
interaction_columns = ['age_usg', 'age_ORB%', 'age_DRB%', 'age_AST%', 'age_TO%', 'age_ast/tov', 'age_ftr', 'age_FT%', 'age_bpm', 
                       'age_bpm_adj', 'age_mp', 'age_3par', 'age_3P%_regressed', 'age_blk_share', 'age_stl_share', 'age_TS%',
                       'hgt_usg', 'hgt_ORB%', 'hgt_DRB%', 'hgt_AST%', 'hgt_TO%', 'hgt_ast/tov', 'hgt_ftr', 'hgt_FT%', 'hgt_bpm', 
                       'hgt_bpm_adj', 'hgt_mp', 'hgt_3par', 'hgt_3P%_regressed', 'hgt_blk_share', 'hgt_stl_share', 'hgt_TS%']

data[correl_columns] = data_imputation_08_09(data[correl_columns].copy())
data_full = data[list(set(correl_columns+interaction_columns+['adj_rating','short']))]

#correl_columns = correl_columns + interaction_columns
data = data[correl_columns] 

scaler = RobustScaler()
scaler.fit(data)
data = scaler.transform(data)
data = pd.DataFrame(data, columns=correl_columns)

#correlation_matrix = data.corr()
#plot_histograms(data)

print() #linebreak post imputation

#%% correlation matrix weights
def get_analytical_weights(X):
    #v1, assume all are equally weighted
    optimal_weights = len(X.columns) * [1]
    
    #v2, intuition
    optimal_weights[3] = 0.9
    optimal_weights[8] = 1.25
    optimal_weights[11] = 1.1
    optimal_weights[12] = 1.1
    optimal_weights[17] = 0.75
    optimal_weights[19] = 0.75
    optimal_weights[20] = 1.25
    
    #v3, after grid search on a limited set
    optimal_weights = [1,1.15,0.9,1,0.95,1,1,1.05,0.95,1.05,1,1,1,0.9,1,1,1,1,0.9,1,1.25,1,0.85,1.05,1.1]
    
    #v4, positions added, verify weights
    optimal_weights = [1,1.15,0.9,1,0.95,1,1,1.05,0.95,1.05,1,1,1,0.9,1,1,1,1,0.9,1,1.25,1,0.85,1.05,1.1, 0.75]
    
    #v5, rimasst, midasst and interaction terms
    optimal_weights = [1,1.15,0.9,1,0.95,1,1,1.05,0.95,1.05,1,1,1,1,1,1,1,1,1,0.9,1.25,1.05,0.9,1.1, 1,1,1,1, 
                       1.15,0.9,0.9,0.75]
    
    #v6, post grid search after adding interaction terms  
    return np.array(optimal_weights)

def get_analytical_weights_randomized(X,i):
    import random
    optimal_weights = len(X.columns) * [1]
    target_set = [.5,.75,.9,1,1.1,1.25,1.5]
    #selected_indices = [0]
    j=0
    while(j<len(optimal_weights)):
        optimal_weights[j] = random.choice(target_set)
        j+=1

    return np.array(optimal_weights)

#%% nba mins per season (for padding the DARKO values for backup centers)
#season_nba_mins = mins_per_season(latest_season-1)

#%% function to map nba stats
def extract_nba_stats(year):
    nba_stats_y = pd.read_excel(f'{path}/nba_stats.xlsx','DARKO')
    mapping = pd.read_excel(f'{path}/nba_stats.xlsx','mapping DARKO')
    nba_stats_y = pd.merge(nba_stats_y, mapping, left_on='player_name', right_on='player_name', how='left')
    
    #padding DARKO with mins
    mins = pd.read_excel(f'{path}/nba_stats.xlsx','mins')
    mins['season'] = mins['season'] + 1
    nba_stats_y = pd.merge(nba_stats_y, mins[['RowId','season','Minutes']], left_on=['nba_id','season_x'], right_on=['RowId','season'], how='left')
    
    #impute values for missing seasons and put lower threshold to 1
    #nba_stats_y['Minutes'] = nba_stats_y['Minutes'].fillna(92*nba_stats_y['age']-1.4*nba_stats_y['age']*nba_stats_y['age']+327*nba_stats_y['o_dpm']+89*nba_stats_y['d_dpm'])
    #nba_stats_y['Minutes'] = nba_stats_y['Minutes'].clip(lower=1)
    nba_stats_y['Minutes'] = nba_stats_y['Minutes'].fillna(0)
    
    #Padding for the DARKO values
    mins = pd.pivot_table(mins,index='season',values='Minutes',aggfunc='mean')
    mins['Minutes'] /= 4
    mins['Minutes'] = np.minimum(250,mins['Minutes'])
    mins = mins.reset_index()
    mins.columns = ['season_avg','th']
    nba_stats_y = pd.merge(nba_stats_y, mins, left_on=['season_x'], right_on=['season_avg'], how='left')
    
    nba_stats_y['o_dpm'] = (nba_stats_y['o_dpm'] * nba_stats_y['Minutes'] + -3 * nba_stats_y['th'])/(nba_stats_y['Minutes'] + nba_stats_y['th']) #-2.5
    nba_stats_y['d_dpm'] = (nba_stats_y['d_dpm'] * nba_stats_y['Minutes'] + -2 * nba_stats_y['th'])/(nba_stats_y['Minutes'] + nba_stats_y['th']) #-1.5
    nba_stats_y['age_adj'] = nba_stats_y['age'].round()
    
    sample_df = nba_stats_y[['player_name','season_x','Minutes','age_adj','o_dpm','d_dpm','pid']]
    sample_df = sample_df[sample_df['season_x']<=year]
    nba_stats_y = dpm_forecaster_new(sample_df)
    nba_stats_y['pid'] = pd.to_numeric(nba_stats_y['pid'], errors='coerce')
    nba_stats_y = nba_stats_y.rename(columns={'age': 'age_adj'})
    
    nba_stats_y = nba_stats_y[['player_name','season_x','age_adj','o_dpm','d_dpm','pid','is_observed']]
    nba_stats_y = nba_stats_y[nba_stats_y['pid'].notna()]
    
    #nba_stats_y = nba_stats_y[nba_stats_y['season_x']<=year]
    #nba_stats_y = age_curve_adj(nba_stats_y.copy(),year)
    return nba_stats_y

def dpm_forecaster_new(df):
    # ── constants ─────────────────────────────────────────────────────────────
    AGE_MIN, AGE_MAX = 19, 40
    AGES        = list(range(AGE_MIN, AGE_MAX + 1))
    TIER_NAMES  = ["allnba", "allstar", 'good', "rotation", "fringe", "marginal"]
    N_TIERS     = len(TIER_NAMES)

    # Four percentile cut-points on total_dpm → five tiers (top → bottom).
    # Evaluated per age so the same raw DPM can rank differently at 22 vs 35.
    PCTILE_CUTS     = [95, 90, 75, 50, 25, 5]
    AGE_WINDOW      = 1      # ± seasons included when computing per-age stats
    MIN_CELL_OBS    = 8      # fall back to global percentile if fewer obs at age
    RECENCY_DECAY   = 1.5    # exponential decay rate across observed seasons

    # ── Step 1: per-age tier boundaries ───────────────────────────────────────
    obs = df.dropna(subset=["o_dpm", "d_dpm"]).copy()
    obs["total_dpm"] = obs["o_dpm"] + obs["d_dpm"]
    obs["age_int"]   = obs["age_adj"].round().astype(int).clip(AGE_MIN, AGE_MAX)

    global_cuts = np.percentile(obs["total_dpm"], PCTILE_CUTS)

    # age_cuts[age] = [p90, p75, p50, p25] thresholds on total_dpm
    age_cuts = {}
    for age in AGES:
        window = obs[obs["age_int"].between(age - AGE_WINDOW,
                                            age + AGE_WINDOW)]["total_dpm"]
        age_cuts[age] = (np.percentile(window, PCTILE_CUTS)
                         if len(window) >= MIN_CELL_OBS else global_cuts.copy())

    def _tier_idx(total_dpm, age_int):
        cuts = age_cuts[int(np.clip(age_int, AGE_MIN, AGE_MAX))]
        if   total_dpm >= cuts[0]: return 0   # elite
        elif total_dpm >= cuts[1]: return 1   # good
        elif total_dpm >= cuts[2]: return 2   # rotation
        elif total_dpm >= cuts[3]: return 3   # fringe
        elif total_dpm >= cuts[4]: return 4   # fringe
        else:                      return 5   # marginal

    obs["tier_idx"] = [_tier_idx(r.total_dpm, r.age_int)
                       for r in obs.itertuples(index=False)]

    # ── Step 2: piecewise quadratic aging curves per tier ─────────────────────
    # Each tier curve is now TWO quadratics with a shared peak:
    #   young side (age <= peak):  y = peak_val + a_left  * (age - peak_age)^2
    #   old side   (age >  peak):  y = peak_val + a_right * (age - peak_age)^2
    # where a_left and a_right are trained separately from data.
    #
    # We enforce concave-down shapes and require older-age decay to be steeper
    # than younger-age rise (|a_right| >= ratio * |a_left|).
    #
    # Per-tier multipliers: (dpm_rise, dpm_decay, min_rise, min_decay)
    _AMP = {
        "allnba":   (1.05, 1.30, 1.00, 1.20),
        "allstar":  (1.10, 1.35, 1.05, 1.25),
        "good":     (1.15, 1.45, 1.10, 1.35),
        "rotation": (1.20, 1.55, 1.15, 1.45),
        "fringe":   (1.25, 1.70, 1.20, 1.60),
        "marginal": (1.30, 1.85, 1.25, 1.75),
    }
    _MIN_A_DPM = 0.0015
    _MIN_A_MIN = 0.30
    _OLDER_STEEPER_RATIO_DPM = 1.25
    _OLDER_STEEPER_RATIO_MIN = 1.20

    def _fit_quad_raw(ages_arr, vals_arr, fallback=0.0):
        """OLS quadratic fit; returns (a2, a1, a0) coefficients."""
        mask = np.isfinite(vals_arr) & np.isfinite(ages_arr)
        a, v = ages_arr[mask], vals_arr[mask]
        if len(np.unique(a)) >= 3:
            return np.polyfit(a, v, deg=2)
        elif len(np.unique(a)) >= 2:
            c = np.polyfit(a, v, deg=1)
            return np.array([0.0, c[0], c[1]])
        elif len(np.unique(a)) == 1:
            return np.array([0.0, 0.0, float(v.mean()) if len(v) else fallback])
        else:
            return np.array([0.0, 0.0, fallback])

    def _fit_side_curvature(side_ages, side_vals, peak_age, peak_val, min_abs_a):
        """Fit y - peak_val = a * (age - peak_age)^2 on one side of the peak."""
        mask = np.isfinite(side_ages) & np.isfinite(side_vals)
        x = side_ages[mask] - peak_age
        y = side_vals[mask] - peak_val
        if len(x) == 0:
            return -min_abs_a
        x2 = x ** 2
        denom = float(np.dot(x2, x2))
        if denom <= 1e-12:
            return -min_abs_a
        a = float(np.dot(x2, y) / denom)
        return a

    def _fit_piecewise_curve(
        ages_arr,
        vals_arr,
        rise_amp,
        decay_amp,
        min_abs_a,
        older_ratio,
        fallback=0.0,
    ):
        """
        Train two quadratics (young/old) with shared peak and steeper old decay.
        """
        raw = _fit_quad_raw(ages_arr, vals_arr, fallback=fallback)

        if abs(raw[0]) > 1e-10:
            peak_age = float(np.clip(-raw[1] / (2 * raw[0]), AGE_MIN, AGE_MAX))
            peak_val = float(np.poly1d(raw)(peak_age))
        else:
            valid_ages = ages_arr[np.isfinite(ages_arr)]
            peak_age = float(np.clip(np.median(valid_ages) if len(valid_ages) else 27.0,
                                     AGE_MIN, AGE_MAX))
            valid_vals = vals_arr[np.isfinite(vals_arr)]
            peak_val = float(valid_vals.mean()) if len(valid_vals) else float(fallback)

        left_mask  = ages_arr <= peak_age
        right_mask = ages_arr >= peak_age

        a_left = _fit_side_curvature(
            ages_arr[left_mask], vals_arr[left_mask], peak_age, peak_val, min_abs_a
        )
        a_right = _fit_side_curvature(
            ages_arr[right_mask], vals_arr[right_mask], peak_age, peak_val, min_abs_a
        )

        # Apply side-specific amplification
        a_left *= rise_amp
        a_right *= decay_amp

        # Enforce concave-down shape and steeper decay on the older side.
        a_left = -max(abs(a_left), min_abs_a)
        a_right = -max(abs(a_right), min_abs_a)
        a_right = -max(abs(a_right), abs(a_left) * older_ratio)

        return {
            "peak_age": peak_age,
            "peak_val": peak_val,
            "a_left": a_left,
            "a_right": a_right,
        }

    def _eval_piecewise(piece, age):
        dx = float(age) - piece["peak_age"]
        if age <= piece["peak_age"]:
            return piece["peak_val"] + piece["a_left"] * dx * dx
        return piece["peak_val"] + piece["a_right"] * dx * dx

    tier_curves      = []   # {age: {o_dpm, d_dpm, minutes, o_std, d_std, m_std}}
    tier_quad_params = []   # (piece_o, piece_d, piece_m) for diagnostics

    for t in range(N_TIERS):
        name_t = TIER_NAMES[t]
        dpm_rise, dpm_decay, min_rise, min_decay = _AMP[name_t]
        td = obs[obs["tier_idx"] == t]

        t_ages = td["age_int"].to_numpy(dtype=float)
        piece_o = _fit_piecewise_curve(
            t_ages,
            td["o_dpm"].to_numpy(dtype=float),
            rise_amp=dpm_rise,
            decay_amp=dpm_decay,
            min_abs_a=_MIN_A_DPM,
            older_ratio=_OLDER_STEEPER_RATIO_DPM,
            fallback=0.0,
        )
        piece_d = _fit_piecewise_curve(
            t_ages,
            td["d_dpm"].to_numpy(dtype=float),
            rise_amp=dpm_rise,
            decay_amp=dpm_decay,
            min_abs_a=_MIN_A_DPM,
            older_ratio=_OLDER_STEEPER_RATIO_DPM,
            fallback=0.0,
        )

        mins_obs = td["Minutes"].dropna()
        if len(mins_obs) > 0:
            m_ages = td.loc[mins_obs.index, "age_int"].to_numpy(dtype=float)
            m_vals = mins_obs.to_numpy(dtype=float)
        else:
            m_ages = np.array([], dtype=float)
            m_vals = np.array([], dtype=float)

        piece_m = _fit_piecewise_curve(
            m_ages,
            m_vals,
            rise_amp=min_rise,
            decay_amp=min_decay,
            min_abs_a=_MIN_A_MIN,
            older_ratio=_OLDER_STEEPER_RATIO_MIN,
            fallback=1200.0,
        )

        tier_quad_params.append((piece_o, piece_d, piece_m))

        # per-age residual std for Gaussian likelihood
        o_std_map, d_std_map, m_std_map = {}, {}, {}
        for age in AGES:
            cell = td[td["age_int"].between(age - AGE_WINDOW, age + AGE_WINDOW)]
            if len(cell) >= 3:
                o_std_map[age] = max(float(cell["o_dpm"].std(ddof=1)), 0.20)
                d_std_map[age] = max(float(cell["d_dpm"].std(ddof=1)), 0.15)
                m_std_map[age] = max(float(cell["Minutes"].std(ddof=1)), 120.0)
            else:
                o_std_map[age] = max(float(td["o_dpm"].std(ddof=1))
                                     if len(td) >= 2 else 0.50, 0.20)
                d_std_map[age] = max(float(td["d_dpm"].std(ddof=1))
                                     if len(td) >= 2 else 0.40, 0.15)
                m_std_map[age] = max(float(td["Minutes"].std(ddof=1))
                                     if len(td["Minutes"].dropna()) >= 2 else 300.0,
                                     120.0)

        tier_curves.append({
            age: {
                "o_dpm":   float(np.clip(_eval_piecewise(piece_o, age), -8.0, 8.0)),
                "d_dpm":   float(np.clip(_eval_piecewise(piece_d, age), -6.0, 6.0)),
                "minutes": float(np.clip(_eval_piecewise(piece_m, age),  0.0, 3500.0)),
                "o_std":   o_std_map[age],
                "d_std":   d_std_map[age],
                "m_std":   m_std_map[age],
            }
            for age in AGES
        })

    # ── print piecewise curvature diagnostics ──────────────────────────────────
    print("[aging curves] Piecewise quadratic curvatures by tier:")
    print(f"  {'tier':>9} | {'o_dpm':^39} | {'d_dpm':^39} | {'minutes':^39}")
    for t, tname in enumerate(TIER_NAMES):
        co, cd, cm = tier_quad_params[t]
        fmt = lambda p: (
            f"peak={p['peak_age']:.2f}, young_a={p['a_left']:+.4f}, old_a={p['a_right']:+.4f}"
        )
        print(f"  {tname:>9} | {fmt(co)} | {fmt(cd)} | {fmt(cm)}")

    # ── print tier threshold summary ──────────────────────────────────────────
    sample_ages = [20, 23, 26, 29, 32, 35, 38]
    print("\n[thresholds] Age-specific total-DPM tier boundaries (sample ages):")
    header = f"  {'tier':>9} | " + " | ".join(f"age {a}" for a in sample_ages)
    print(header)
    for i, name in enumerate(TIER_NAMES[:-1]):   # 4 cut-points
        row = f"  {name:>9} | " + " | ".join(
            f"{age_cuts[a][i]:+.2f}" for a in sample_ages
        )
        print(row)

    # ── helpers for per-player computation ───────────────────────────────────
    def _norm_logpdf(x, mu, sigma):
        """Gaussian log-density (no scipy needed)."""
        sigma = max(float(sigma), 1e-6)
        return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)

    def _tier_probs(obs_rows):
        """
        Soft tier probabilities from recency-weighted Gaussian likelihood.
        Uses o_dpm, d_dpm, Minutes and age.
        obs_rows: DataFrame sorted ascending by age_adj, with o_dpm & d_dpm.
        """
        if len(obs_rows) == 0:
            return np.ones(N_TIERS) / N_TIERS

        n = len(obs_rows)
        rw = np.exp(np.linspace(-RECENCY_DECAY, 0.0, n))
        rw /= rw.sum()

        log_p = np.zeros(N_TIERS)
        for i, (_, row) in enumerate(obs_rows.iterrows()):
            ai = int(np.clip(round(float(row["age_adj"])), AGE_MIN, AGE_MAX))
            o, d = float(row["o_dpm"]), float(row["d_dpm"])
            m = float(row["Minutes"]) if pd.notna(row["Minutes"]) else None
            for t in range(N_TIERS):
                c = tier_curves[t][ai]
                ll = (
                    _norm_logpdf(o, c["o_dpm"], c["o_std"])
                    + _norm_logpdf(d, c["d_dpm"], c["d_std"])
                )
                if m is not None:
                    ll += _norm_logpdf(m, c["minutes"], c["m_std"])
                log_p[t] += rw[i] * ll

        log_p -= log_p.max()
        probs = np.exp(log_p)
        return probs / probs.sum()

    def _residuals(obs_rows, rw):
        """
        Per-tier recency-weighted residual: player's observed value − tier curve.
        Returns arrays of shape (N_TIERS,) for o, d, minutes.
        """
        res_o = np.zeros(N_TIERS)
        res_d = np.zeros(N_TIERS)
        res_m = np.zeros(N_TIERS)
        for i, (_, row) in enumerate(obs_rows.iterrows()):
            ai = int(np.clip(round(float(row["age_adj"])), AGE_MIN, AGE_MAX))
            o, d = float(row["o_dpm"]), float(row["d_dpm"])
            m    = float(row["Minutes"]) if pd.notna(row["Minutes"]) else None
            for t in range(N_TIERS):
                c = tier_curves[t][ai]
                res_o[t] += rw[i] * (o - c["o_dpm"])
                res_d[t] += rw[i] * (d - c["d_dpm"])
                if m is not None:
                    res_m[t] += rw[i] * (m - c["minutes"])
        return res_o, res_d, res_m

    # ── per-player projection ─────────────────────────────────────────────────
    def _project_player(player_data):
        # build age → {o, d, m} lookup; for duplicate ages keep most-minutes row
        known = {}
        for _, row in player_data.sort_values("Minutes",
                                              ascending=False,
                                              na_position="last").iterrows():
            ai = int(np.clip(round(float(row["age_adj"])), AGE_MIN, AGE_MAX))
            if ai not in known:
                known[ai] = {
                    "o": row["o_dpm"]   if pd.notna(row["o_dpm"])   else None,
                    "d": row["d_dpm"]   if pd.notna(row["d_dpm"])   else None,
                    "m": float(row["Minutes"]) if pd.notna(row["Minutes"]) else None,
                    "season_x": int(round(float(row["season_x"])))
                    if pd.notna(row["season_x"]) else None,
                }

        # player-specific season-age mapping: season_x ≈ age + offset
        season_rows = player_data.dropna(subset=["season_x", "age_adj"])
        if len(season_rows) > 0:
            season_offset = float(
                np.median(
                    season_rows["season_x"].to_numpy(dtype=float)
                    - season_rows["age_adj"].to_numpy(dtype=float)
                )
            )
        else:
            season_offset = None

        obs_rows = (player_data.dropna(subset=["o_dpm", "d_dpm"])
                               .sort_values("age_adj")
                               .reset_index(drop=True))

        tp = _tier_probs(obs_rows)
        dominant = TIER_NAMES[int(np.argmax(tp))]

        n = len(obs_rows)
        if n > 0:
            rw = np.exp(np.linspace(-RECENCY_DECAY, 0.0, n)); rw /= rw.sum()
            res_o, res_d, res_m = _residuals(obs_rows, rw)
        else:
            res_o = res_d = res_m = np.zeros(N_TIERS)

        results = []
        for age in AGES:
            entry       = known.get(age)
            is_observed = entry is not None
            if is_observed and entry["season_x"] is not None:
                season_x = int(entry["season_x"])
            elif season_offset is not None:
                season_x = int(round(age + season_offset))
            else:
                season_x = None

            # ── DPM ──────────────────────────────────────────────────────────
            if is_observed and entry["o"] is not None:
                pred_o, pred_d = entry["o"], entry["d"]
                per_tier_od = {
                    TIER_NAMES[t]: (pred_o, pred_d)
                    for t in range(N_TIERS)
                }
            else:
                # Pairwise per-tier forecasts (o_dpm, d_dpm), then blended output.
                per_tier_od = {
                    TIER_NAMES[t]: (
                        float(tier_curves[t][age]["o_dpm"] + res_o[t]),
                        float(tier_curves[t][age]["d_dpm"] + res_d[t]),
                    )
                    for t in range(N_TIERS)
                }
                pred_o = float(sum(tp[t] * per_tier_od[TIER_NAMES[t]][0]
                                   for t in range(N_TIERS)))
                pred_d = float(sum(tp[t] * per_tier_od[TIER_NAMES[t]][1]
                                   for t in range(N_TIERS)))

            # ── Minutes ───────────────────────────────────────────────────────
            if is_observed and entry["m"] is not None:
                pred_m = entry["m"]
            else:
                pred_m = float(sum(
                    tp[t] * max(0.0, tier_curves[t][age]["minutes"] + res_m[t])
                    for t in range(N_TIERS)
                ))

            results.append({
                "season_x":    season_x,
                "age":         age,
                "o_dpm":       round(pred_o, 4),
                "d_dpm":       round(pred_d, 4),
                "dpm":         round(pred_o + pred_d, 4),
                "minutes":     round(max(0.0, pred_m), 1),
                "is_observed": is_observed,
                "tier":        dominant,
                **{f"p_{TIER_NAMES[t]}": round(float(tp[t]), 4)
                   for t in range(N_TIERS)},
                **{f"o_dpm_{tn}": round(float(per_tier_od[tn][0]), 4)
                   for tn in TIER_NAMES},
                **{f"d_dpm_{tn}": round(float(per_tier_od[tn][1]), 4)
                   for tn in TIER_NAMES},
            })

        return pd.DataFrame(results)

    # ── project all players ───────────────────────────────────────────────────
    df = df.copy()
    df["_gkey"] = df["player_name"].astype(str) + "||" + df["pid"].astype(str)
    groups = list(df.groupby("_gkey", sort=False))
    print(f"\nProjecting {len(groups):,} players with probabilistic tier curves ...")

    pieces = []
    for gkey, grp in groups:
        name, pid_str = gkey.split("||", 1)
        proj = _project_player(grp)
        proj["player_name"] = name
        proj["pid"]         = pid_str
        pieces.append(proj)

    out = pd.concat(pieces, ignore_index=True)
    prob_cols = [f"p_{t}" for t in TIER_NAMES]
    pair_cols = [f"o_dpm_{t}" for t in TIER_NAMES] + [f"d_dpm_{t}" for t in TIER_NAMES]
    out = out[["player_name", "pid", "season_x", "age", "o_dpm", "d_dpm", "dpm",
               "minutes", "is_observed", "tier"] + prob_cols + pair_cols]
    print(f"Done. {out.shape[0]:,} rows ({len(groups)} players × "
          f"{AGE_MAX - AGE_MIN + 1} ages)")
    return out
    return out

def adjust_dpm(group,adj_season):
    curr_age = group.loc[group['season_x']==group['season_x'].max(),'age'].mean()
    group.loc[group['age']<=curr_age,'decay'] = np.nan
    group.loc[group['age']<=curr_age,'offense'] = np.nan
    group.loc[group['age']<=curr_age,'defense'] = np.nan
    
    # Find first valid o_dpm and d_dpm
    base_off = group['o_dpm'].ffill()
    base_def = group['d_dpm'].ffill()
    base_pid = group['pid'].ffill()

    # Compute cumulative age curve adjustments
    cum_off = group['offense'].cumsum()
    cum_off = cum_off.fillna(0)
    cum_def = group['defense'].cumsum()
    cum_def = cum_def.fillna(0)

    # Fill missing o_dpm and d_dpm using base + cumulative adjustments
    group['o_dpm'] = base_off + cum_off #.where(group['age']>curr_age)
    group['d_dpm'] = base_def + cum_def #.where(group['age']>curr_age)

    # Fill pid forward
    group['pid'] = base_pid
    return group

def age_curve_adj(stats,c_s):
    # Load age curve data
    age_curve = pd.read_excel(f'{path}/nba_stats.xlsx', sheet_name='age curve', engine='openpyxl')

    # Create all player-age combinations
    combinations = list(itertools.product(stats['player_name'].unique(), range(19, 40)))
    combinations = pd.DataFrame(combinations, columns=['player_name', 'age_adj'])

    # Merge with stats and age curve
    combinations = combinations.merge(stats, on=['player_name', 'age_adj'], how='outer')
    combinations = combinations.merge(age_curve, left_on='age_adj', right_on='age')
    combinations = combinations.sort_values(by=['player_name', 'age'], ascending=True)
    """
    # Initialize previous values
    prev_name = ""
    prev_pid = np.nan
    prev_off = np.nan
    prev_def = np.nan
    
    # Iterate and adjust values
    for x in combinations.itertuples(index=False):
        pname, age_adj, *_ , o_dpm, d_dpm, pid, _, _, age_curve_off, age_curve_def = x

        if pd.isna(o_dpm) and pd.notna(prev_off) and pd.notna(prev_def) and (prev_name == pname):
            combinations.loc[(combinations['player_name'] == pname) & (combinations['age'] == age_adj), 'o_dpm'] = prev_off + age_curve_off
            combinations.loc[(combinations['player_name'] == pname) & (combinations['age'] == age_adj), 'd_dpm'] = prev_def + age_curve_def
            combinations.loc[(combinations['player_name'] == pname) & (combinations['age'] == age_adj), 'pid'] = prev_pid
            prev_off += age_curve_off
            prev_def += age_curve_def
        else:
            prev_off = o_dpm
            prev_def = d_dpm
            prev_pid = pid

        prev_name = pname
    """
    #combinations = combinations.groupby('player_name', group_keys=False).apply(adjust_dpm)
    combinations = combinations.groupby('player_name', group_keys=False).apply(
        lambda g: adjust_dpm(g, c_s)
    )

    return combinations

def outcomes(df):
    #nba_stats = extract_nba_stats(season)
    player_peaks = df.groupby('pid')[['o_dpm','d_dpm']].agg(lambda x: x.nlargest(5).mean())
    player_peaks = player_peaks.reset_index()
    player_bio = df.pivot_table(index='pid',values=['player_name','season_x'],aggfunc='first')
    player_bio = player_bio.reset_index()
    player_outcomes = player_bio.merge(player_peaks, on='pid', how='left')
    player_outcomes['dpm'] = 1*player_outcomes['o_dpm'] + 1*player_outcomes['d_dpm']
    player_outcomes['season_x'] -= 1
    player_outcomes = player_outcomes.merge(player_stats[['pid','player']].drop_duplicates(), on='pid', how='left')
    player_outcomes = player_outcomes[['pid', 'player', 'season_x', 'o_dpm', 'd_dpm', 'dpm']]
    return player_outcomes

def p_nba_scaling(league_stats,p_stats):
    #4% of players from college and INTL leagues make the nba
    scale = 0.04/(len(league_stats['pid'].unique())/len(p_stats['pid'].unique()))
    if(scale>2): scale = 2
    if(scale<0): scale = 0
    
    return scale

#%% ml model
def regression_draft_model(league_stats, pre_nba_data, pre_nba_raw, train_season_end):
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

    # ---- build target dpm from top-5 offensive/defensive comps ----
    off_comps = league_stats.groupby(["player_name", "pid"])["o_dpm"].apply(lambda x: x.nlargest(5))
    off_comps = off_comps.groupby(["player_name", "pid"]).mean().dropna().reset_index()

    def_comps = league_stats.groupby(["player_name", "pid"])["d_dpm"].apply(lambda x: x.nlargest(5))
    def_comps = def_comps.groupby(["player_name", "pid"]).mean().dropna().reset_index()

    dist2 = off_comps.copy()
    dist2["d_dpm"] = def_comps["d_dpm"]
    dist2["dpm"] = dist2["o_dpm"] + dist2["d_dpm"]
    dist2 = dist2[["pid", "player_name", "dpm"]]

    pre_nba_data[["player", "pid", "team", "season","intl"]] = pre_nba_raw[["player", "pid", "team", "season","intl"]]
    pre_nba_data = pre_nba_data.merge(dist2[["pid", "dpm"]], on=["pid"], how="left")
    df_sorted = pre_nba_data.sort_values(by=["season", "dpm"], ascending=[False, False])
    df_sorted["dpm"] = df_sorted["dpm"].fillna(-5.0)
    df_sorted.loc[df_sorted["dpm"] < -5, "dpm"] = -5.0

    df_sorted["makes NBA"] = np.where(df_sorted["dpm"] > -5, 1, 0)
    df_sorted["rotation"] = np.where(df_sorted["dpm"] >= -1, 1, 0)
    df_sorted["starter"] = np.where(df_sorted["dpm"] >= 0, 1, 0)
    df_sorted["all star"] = np.where(df_sorted["dpm"] >= 1, 1, 0)
    df_sorted["all nba"] = np.where(df_sorted["dpm"] >= 2, 1, 0)
    df_sorted["mvp"] = np.where(df_sorted["dpm"] >= 4.5, 1, 0)

    # five nested binary tiers, each rarer than the last
    tier_names = ["makes NBA", "rotation", "starter", "all star", "all nba", "mvp"]

    df_train = df_sorted[df_sorted["season"] <= train_season_end].copy()
    df_test = df_sorted[df_sorted["season"] == latest_season].copy()

    # deterministic, order-preserving feature list (drop the categorical "role")
    columns_considered = correl_columns + interaction_columns + ['adj_rating','intl']
    columns_considered.remove('role')
    #seen = set()
    #columns_considered = [c for c in (correl_columns + interaction_columns )
    #                      if c != "role" and not (c in seen or seen.add(c))]

    X = df_train[columns_considered].to_numpy(dtype=float)
    X_test = df_test[columns_considered].to_numpy(dtype=float)

    def make_model(spw):
        # strong regularization + shallow trees to avoid memorizing the rare class
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=3.0,
            reg_alpha=0.5,
            gamma=1.0,
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=spw,
            max_delta_step=1,
            random_state=42,
        )

    eps = 1e-6

    def to_logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit_tier(name):
        y = df_train[name].to_numpy(dtype=int)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pred_col = "pred_" + name.replace(" ", "_")

        # too few positives to stratify into folds -> emit base rate, skip modelling
        if n_pos < 5:
            base = float(y.mean())
            print(f"[{name}] positives = {n_pos} / {len(y)} -> too few to model; "
                  f"using base rate {base:.5f}")
            df_train[pred_col] = base
            df_test[pred_col] = base
            return

        # The full neg/pos ratio over-inflates and overfits the rare class; since we
        # calibrate afterwards, a milder sqrt weighting learns the minority signal
        # without distorting probabilities as severely.
        spw = float(np.sqrt(n_neg / max(n_pos, 1)))

        # leak-free out-of-fold raw scores on TRAIN
        n_splits = min(5, n_pos)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_raw = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in skf.split(X, y):
            m = make_model(spw)
            m.fit(X[tr_idx], y[tr_idx])
            oof_raw[va_idx] = m.predict_proba(X[va_idx])[:, 1]

        # Platt (sigmoid) calibrator fit on OOF scores: robust for rare events
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(to_logit(oof_raw), y)

        def calibrate(p):
            if len(p) == 0:
                return p
            return calibrator.predict_proba(to_logit(p))[:, 1]

        train_pred = calibrate(oof_raw)

        # final model on ALL training data, used to score the test seasons
        final_model = make_model(spw)
        final_model.fit(X, y)
        test_pred = calibrate(final_model.predict_proba(X_test)[:, 1])

        ap = average_precision_score(y, oof_raw)
        auc = roc_auc_score(y, oof_raw)
        brier = brier_score_loss(y, train_pred)
        print(f"[{name}] pos = {n_pos} / {len(y)} (base {y.mean():.4f}), spw = {spw:.2f} | "
              f"AUC-PR = {ap:.4f} | AUC-ROC = {auc:.4f} | Brier = {brier:.5f} | "
              f"mean pred = {train_pred.mean():.4f}")

        df_train[pred_col] = train_pred * (1-df_train['short']/1.5)
        df_test[pred_col] = test_pred * (1-df_test['short']/1.5)
        #df_test[pred_col] = np.where(df_test['short']>=1,test_pred/2,test_pred)

    pred_cols = []
    for name in tier_names:
        fit_tier(name)
        pred_cols.append("pred_" + name.replace(" ", "_"))

    def enforce_strict_monotone_survival(arr, base_gap=1e-3):
        """
        Keep cumulative tier probabilities monotone while preserving a small
        positive gap between adjacent tiers so the implied continuous
        distribution does not collapse entire intervals to zero mass.
        """
        surv = np.clip(np.asarray(arr, dtype=float), 0.0, 1.0).copy()
        if surv.ndim == 1:
            surv = surv.reshape(1, -1)

        surv = np.minimum.accumulate(surv, axis=1)
        n_cols = surv.shape[1]
        if n_cols <= 1:
            return surv

        for i in range(surv.shape[0]):
            row = surv[i].copy()
            gap = min(base_gap, max(row[0], 1e-6) / max(n_cols + 1, 1))
            for j in range(1, n_cols):
                upper = max(row[j - 1] - gap, 0.0)
                if row[j] >= upper:
                    row[j] = upper
            surv[i] = row
        return surv

    # probability must be <= the previous tier's
    for frame in (df_train, df_test):
        mono = enforce_strict_monotone_survival(frame[pred_cols].to_numpy(dtype=float))
        frame[pred_cols] = mono

    # derive dpm distribution moments/quantiles from tier probabilities
    # Treat the nested tier probabilities as survival values at fixed DPM
    # cutoffs and build a continuous piecewise-linear quantile function.
    # This is cheap, monotone, and distribution-like without per-row fitting.
    floor = -5.0
    lower_tail = min(float(df_train["dpm"].quantile(0.01)), floor - 0.75)
    upper_tail = max(float(df_train["dpm"].quantile(0.99)), 5.5)
    x_knots = np.array([lower_tail, -5.0, -1.0, 0.0, 1.0, 2.0, 4.5, upper_tail], dtype=float)

    def summarize_dpm_distribution(frame):
        surv = enforce_strict_monotone_survival(frame[pred_cols].to_numpy(dtype=float))
        cdf_core = 1.0 - surv
        mean = np.empty(len(cdf_core), dtype=float)
        p05 = np.empty(len(cdf_core), dtype=float)
        p25 = np.empty(len(cdf_core), dtype=float)
        p50 = np.empty(len(cdf_core), dtype=float)
        p75 = np.empty(len(cdf_core), dtype=float)
        p95 = np.empty(len(cdf_core), dtype=float)

        for i in range(len(cdf_core)):
            core = np.clip(cdf_core[i], 1e-4, 1.0 - 1e-4)
            probs = np.empty(len(core) + 2, dtype=float)
            probs[0] = 0.0
            probs[-1] = 1.0
            probs[1:-1] = core

            for j in range(1, len(probs)):
                if probs[j] <= probs[j - 1]:
                    probs[j] = min(probs[j - 1] + 1e-4, 1.0)

            if probs[-1] <= probs[-2]:
                probs[-2] = min(probs[-1] - 1e-4, max(probs[-3] + 1e-4, 0.0))

            # Quantile function is linear between knot probabilities.
            mean[i] = np.sum(0.5 * (x_knots[:-1] + x_knots[1:]) * np.diff(probs))
            p05[i] = np.interp(0.05, probs, x_knots)
            p25[i] = np.interp(0.25, probs, x_knots)
            p50[i] = np.interp(0.50, probs, x_knots)
            p75[i] = np.interp(0.75, probs, x_knots)
            p95[i] = np.interp(0.95, probs, x_knots)

        return mean, p05, p25, p50, p75, p95

    tr_mean, tr_p05, tr_p25, tr_p50, tr_p75, tr_p95 = summarize_dpm_distribution(df_train)
    te_mean, te_p05, te_p25, te_p50, te_p75, te_p95 = summarize_dpm_distribution(df_test)

    # keep pred_dpm as backward-compatible alias for mean
    df_train["pred_dpm"] = tr_mean
    df_test["pred_dpm"] = te_mean
    df_train["pred_dpm_mean"] = tr_mean
    df_test["pred_dpm_mean"] = te_mean
    df_train["pred_dpm_p05"] = tr_p05
    df_test["pred_dpm_p05"] = te_p05
    df_train["pred_dpm_p25"] = tr_p25
    df_test["pred_dpm_p25"] = te_p25
    df_train["pred_dpm_median"] = tr_p50
    df_test["pred_dpm_median"] = te_p50
    df_train["pred_dpm_p75"] = tr_p75
    df_test["pred_dpm_p75"] = te_p75
    df_train["pred_dpm_p95"] = tr_p95
    df_test["pred_dpm_p95"] = te_p95

    rmse = float(np.sqrt(np.mean((df_train["pred_dpm_mean"].to_numpy() - df_train["dpm"].to_numpy()) ** 2)))
    mae = float(np.mean(np.abs(df_train["pred_dpm_mean"].to_numpy() - df_train["dpm"].to_numpy())))
    med_ae = float(np.median(np.abs(df_train["pred_dpm_median"].to_numpy() - df_train["dpm"].to_numpy())))
    print(f"[dpm dist] RMSE(mean) = {rmse:.4f} | MAE(mean) = {mae:.4f} | MedAE(median) = {med_ae:.4f}")

    dpm_cols = ["pred_dpm", "pred_dpm_p05", "pred_dpm_p25", "pred_dpm_median", "pred_dpm_p75", "pred_dpm_p95"]
    train_out = df_train[["pid", "player", "team", "season", "dpm"] + dpm_cols + tier_names + pred_cols]
    test_out = df_test[["pid", "player", "team", "season"] + dpm_cols + pred_cols]
    return train_out, test_out

#%% get latest nba stats
nba_stats = extract_nba_stats(latest_season)
#draft_outcomes = outcomes(nba_stats)

correl_weights = get_analytical_weights(data)
#rotation_scale_factor = p_nba_scaling(nba_stats,player_stats)

print("ML model summary")
_,test = regression_draft_model(nba_stats.copy(),data_full.copy(),player_stats.copy(),latest_season-3)

test_reduced = test[['pid', 'player', 'team', 'season', 'pred_makes_NBA']]
test_reduced = test_reduced.rename(columns={"pred_makes_NBA": "pred"})
player_stats = player_stats.merge(test_reduced, on=['player','pid', 'team','season'], how='left')

made_nba = nba_stats[['pid']].drop_duplicates()
made_nba = made_nba.dropna()
made_nba['made_nba'] = 1
player_stats = player_stats.merge(made_nba, on=['pid'], how='left') #season
player_stats['made_nba'] = player_stats['made_nba'].fillna(0)

player_stats = player_stats[['player','age','pid', 'team','season','class', 'hgt','GP', 'mp','usg', 'TS%','ORB%','DRB%', 
                   'AST%','TO%', 'ast/tov','BLK%', 'blk_share','STL%', 'stl_share','pfr', 'ftr','FT%', 'dunkar',
                   'rimar', 'rim%','midar','mid%','3par', '3P%','ORtg', 'drtg','bpm','2par', '2P%','bpm_adj', 
                   'dunkasst','rimasst', 'midasst','3Passt','role','3p_regress_val', '3P%_regressed',
                   'made_nba', 'pred', 'mp_p', 'intl']]

del test_reduced,made_nba

#%% mahalanobis distance algo
def distance2(pid, yr, full_matrix, data_copy, weights_copy, print_df):
    data_copy = data_copy[data_copy.index.isin(full_matrix.index)]
    
    cov = np.ma.cov(np.ma.masked_invalid(data_copy), rowvar=False)    
    try:
        invcov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If covariance is singular, use pseudo-inverse
        invcov = np.linalg.pinv(cov)
    
    # Map skewed & zero-inflated data into a normal space while preserving rank & outliers
    #transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    #transformed_data = transformer.fit_transform(data_copy)
    
    # Fit Ledoit-Wolf Shrinkage on the transformed rank space
    #lw = LedoitWolf().fit(transformed_data)
    #cov = lw.covariance_
    #invcov = lw.precision_

    # Get player data
    player_data = full_matrix.loc[(full_matrix['pid'] == pid) & (full_matrix['season'] == yr)]
    """
    if(name == "Jalen Johnson"): player_data = full_matrix.loc[(full_matrix['pid'] == 73238) & (full_matrix['season'] == yr)]
    else: player_data = full_matrix.loc[(full_matrix['player'] == name) & (full_matrix['season'] == yr)]
    """
    player_index = player_data.index[0]
    player = data_copy.iloc[player_index] # data_copy
    pvec = np.ma.masked_invalid(np.array(player))

    data_array = np.ma.masked_invalid(data_copy.to_numpy()) # data_copy
    delta_matrix = data_array - pvec #data_array

    # adding custom weights to the covariance matrix
    weighted_invcov = weights_copy[:, None] * invcov * weights_copy[None, :]

    #temp = np.einsum('ij,jk,ik->i', delta_matrix, invcov, delta_matrix)
    temp = np.einsum('ij,jk,ik->i', delta_matrix, weighted_invcov, delta_matrix)
    temp = np.clip(temp, 0.0, None)
    dist_array = np.sqrt(temp)

    dist_array[player_index] = 0

    full_matrix['mdist'] = dist_array
    full_matrix = full_matrix[['player', 'team', 'season', 'hgt', 'bpm', 'mdist', 'pid']]
    full_matrix['score'] = 1 / (full_matrix['mdist'] ** 2) # np.exp(-full_matrix['mdist']*full_matrix['mdist']/2)
    full_matrix = full_matrix.sort_values(by='score', ascending=False)

    #if(len(full_matrix[full_matrix['mdist']<=3])>100): full_matrix = full_matrix.head(100)
    #elif(len(full_matrix[full_matrix['mdist']<=3])<50): full_matrix = full_matrix.head(50)
    #else: full_matrix = full_matrix[full_matrix['mdist']<=3]
    full_matrix = full_matrix.head(150) #150

    if(print_df == 0): full_matrix = full_matrix[(full_matrix['mdist']==0)|(~(full_matrix['pid'].isin(full_matrix.loc[full_matrix['season']>=yr,'pid'])))]
    #print(full_matrix[['player','team','season']].head(5))
    return full_matrix

#%% individual player comps analysis
def fleishman_coeffs(mean, std, skew, kurt):
    def equations(vars):
        a, b, c, d = vars
        eq1 = b**2 + 6*b*d + 2*c**2 + 15*d**2 - 1
        eq2 = 2*c*(b**2 + 24*b*d + 105*d**2 + 2) - skew
        eq3 = b**4 + 24*b**3*d + 144*b**2*d**2 + 12*b**2*c**2 + 720*b*d**3 + 120*b*c**2*d + 36*c**4 + 1680*d**4 + 12*c**2 + 3 - kurt
        eq4 = a
        return [eq1, eq2, eq3, eq4]
    
    #print(mean,std,skew,kurt)
    """
    c_guess = 0.10007 * skew + 0.00844 * pow(skew,3)
    d_guess = 0.95357 - 0.05679 * kurt + 0.0352 * skew + 0.00133 * pow(skew,2)
    b_guess = 0.30978 - 0.31655 * d_guess
    a_guess = mean - c_guess * pow(std,2)
    initial_guess = [a_guess,b_guess,c_guess,d_guess]
    """
    initial_guess = [0, 1, 0, 0]
    #initial_guess = [0, 0, 1, 0]
    a, b, c, d = fsolve(equations, initial_guess)
    #print(a, b, c, d)
    return a, b, c, d

def generate_fleishman_distribution(n_samples, mean, std, skew, kurt):
    a, b, c, d = fleishman_coeffs(mean, std, skew, kurt)
    z = np.random.normal(0, 1, n_samples)
    x = a + b*z + c*z**2 + d*z**3
    x = mean + x*std
    x = np.clip(x, -5, 8) #-2.5
    #x = x.tolist()
    return x

def skew_kurt_error(a, target_skew, target_kurt):
    from scipy.stats import skewnorm, skew, kurtosis
    from scipy.optimize import minimize_scalar
    # The skewnorm distribution's moments depend only on 'a' for the standard form (loc=0, scale=1)
    actual_skew = skewnorm.stats(a, moments='s')
    actual_kurt = skewnorm.stats(a, moments='k') # 'k' is Fisher's excess kurtosis

    # We return the sum of squared errors to minimize
    return (actual_skew - target_skew)**2 + (actual_kurt - target_kurt)**2

def get_a_from_skew(skew):
    # Ensure skew is within theoretical limits (-0.995 to 0.995)
    skew = np.clip(skew, -0.994, 0.994)
    # Calculate d based on the relationship between skewness and shape
    delta = np.sign(skew) * ( (np.pi/2) * (abs(skew)**(2/3)) / (abs(skew)**(2/3) + ((4-np.pi)/2)**(2/3)) )**0.5
    # Calculate alpha (a)
    a = delta / np.sqrt(1 - delta**2)
    return a

def skewed_normal_distributions(n_samples, off_mean, off_std, off_skew, off_kurt, def_mean, def_std, def_skew, def_kurt, c):
    import scipy.stats as stats
    
    if(np.isnan(off_skew)): off_skew = 1e-9
    if(np.isnan(off_kurt)): off_kurt = 1e-9
    if(np.isnan(def_skew)): def_skew = 1e-9
    if(np.isnan(def_kurt)): def_kurt = 1e-9
    if(np.isnan(c)): c = 0
    
    #print(n_samples)
    #print(off_mean, off_std, off_skew, off_kurt)
    #print(def_mean, def_std, def_skew, def_kurt)
    #print(c)
    
    #artificially impose limits
    if(c > 0.5): c = 0.5
    if(c < -0.5): c = -0.5
    if(off_kurt > 5): off_kurt = 5
    if(def_kurt > 5): def_kurt = 5
    
    try: 
        off_a = get_a_from_skew(off_skew)
    except: off_a = 0
    try: 
        def_a = get_a_from_skew(def_skew)
    except: def_a = 0
            
    dist1 = stats.skewnorm(a=off_a, loc=off_mean, scale=off_std)
    dist2 = stats.skewnorm(a=def_a, loc=def_mean, scale=def_std)
    
    cov_matrix = [[1, c],[c, 1]]
    norm_samples = np.random.multivariate_normal([0, 0], cov_matrix, n_samples)
    uniform_samples = stats.norm.cdf(norm_samples)
    
    var1 = dist1.ppf(uniform_samples[:, 0])
    var2 = dist2.ppf(uniform_samples[:, 1])

    var1 = (var1 - np.mean(var1)) * (1+0.1*off_kurt) + np.mean(var1)
    var2 = (var2 - np.mean(var2)) * (1+0.1*def_kurt) + np.mean(var2)
    
    #var1 = np.clip(var1, -3, 6) #-2.5
    #var2 = np.clip(var2, -2, 4) #-1.5
    
    pdata = np.vstack((var1, var2)).T
    #['dpm'] = 
    pdata = 1*pdata[:, 0] + 1*pdata[:, 1]
    pdata = np.nan_to_num(pdata, nan=-5)
    pdata = np.clip(pdata, -5, 10)
    #print(pdata)
    return pdata

def _skew_to_delta(skew):
    skew = np.clip(skew, -0.994, 0.994)
    s23 = np.abs(skew) ** (2.0 / 3.0)
    delta = np.sign(skew) * np.sqrt((np.pi / 2.0) * s23 / (s23 + ((4.0 - np.pi) / 2.0) ** (2.0 / 3.0)))
    return np.clip(delta, -1.0 + 1e-12, 1.0 - 1e-12)

def _skewnorm_excess_kurtosis(delta):
    pi = np.pi
    return 8.0 * (pi - 3.0) * delta**4 / (pi - 2.0 * delta**2) ** 2

def _apply_kurtosis_mixture(x, mu, k_target, delta):
    if k_target <= 0.0:
        return x

    k_base = _skewnorm_excess_kurtosis(delta)
    k_needed = max(0.0, k_target - k_base)

    if k_needed <= 0.0:
        return x

    # variance-mixture strength
    kappa = k_needed / 3.0
    kappa = max(min(kappa, 1.0), -1.0)
    #print(kappa)
    # two-point mixture (mean-preserving)
    u = np.random.rand(len(x))
    s1 = np.sqrt(1.0 + np.sqrt(kappa))
    s2 = np.sqrt(1.0 - np.sqrt(kappa))
    scale = np.where(u < 0.5, s1, s2)
    #print()
    #print(mu)
    #print(scale)
    #print(x)

    return mu + scale * (x - mu)

def _apply_kurtosis_mixture_improved(x, mu, k_target, delta):
    if k_target <= 0.0: return x    
    k_base = _skewnorm_excess_kurtosis(delta)
    k_residual = max(0.0, k_target - k_base)
    
    if k_residual < 1e-6: return x    
    if k_residual >= 2.0:
        # High kurtosis regime (subgaussian mixture)
        alpha_sq = (k_residual - 2.0) / 6.0
        alpha = np.sqrt(np.clip(alpha_sq, 0.0, 0.95))
    else:
        # Low kurtosis regime (approximate)
        # k ≈ 2α² / (1 - α²/2) => α² ≈ k / (2 + k)
        alpha_sq = k_residual / (2.0 + k_residual)
        alpha = np.sqrt(alpha_sq)
    
    s1_sq = 1.0 + alpha
    s2_sq = 1.0 - alpha
    
    # Safety bounds
    s2_sq = max(s2_sq, 0.05)
    s1_sq = 2.0 - s2_sq  # Re-enforce constraint
    
    s1 = np.sqrt(s1_sq)
    s2 = np.sqrt(s2_sq)
    
    # Apply mixture
    u = np.random.rand(len(x))
    scale = np.where(u < 0.5, s1, s2)
    
    return mu + scale * (x - mu)

def skewed_normal_distributions_fast(n_samples, off_mean, off_std, off_skew, off_kurt, def_mean, def_std, def_skew, def_kurt, c):
    
    off_skew, off_kurt, def_skew, def_kurt, c = np.nan_to_num([off_skew, off_kurt, def_skew, def_kurt, c], nan=0.0)

    #print(n_samples)
    #print(off_mean, off_std, off_skew, off_kurt)
    #print(def_mean, def_std, def_skew, def_kurt)
    #print(c)

    c = np.clip(c, -0.5, 0.5)
    off_kurt = min(off_kurt, 5.0) #5
    def_kurt = min(def_kurt, 5.0) #5

    delta1 = _skew_to_delta(off_skew)
    delta2 = _skew_to_delta(def_skew)

    s1 = np.sqrt(1.0 - delta1 * delta1)
    s2 = np.sqrt(1.0 - delta2 * delta2)

    U = np.random.randn(n_samples)
    V = np.random.randn(n_samples)
    sc = np.sqrt(1.0 - c * c)

    Z0 = U
    Z1 = c * U + sc * V
    Z2 = -sc * U + c * V

    var1 = off_mean + off_std * (delta1 * np.abs(Z0) + s1 * Z2)
    var2 = def_mean + def_std * (delta2 * np.abs(Z1) + s2 * Z2)
    #print(var1.mean(),var2.mean())

    var1 = _apply_kurtosis_mixture_improved(var1, off_mean, off_kurt, delta1)
    var2 = _apply_kurtosis_mixture_improved(var2, def_mean, def_kurt, delta2)
    #print(var1.mean(),var2.mean(),delta1,delta2)
    
    pdata = var1 + var2
    pdata = np.nan_to_num(pdata, nan=-5.0)
    return pdata #np.clip(pdata, -5.0, 10.0)

def clustering(start,end,n):
    #features = ['median', '95th']
    #features = ['p >= rotation', 'p >= starter', 'p >= all star', 'p >= all nba']
    features = ['p >= rotation', 'p >= all star']
    
    i = start; df = []
    while(i<end+1):
        dfn = pd.read_excel(f'{path}/results.xlsx',f'{i}')
        first_column_name = dfn.columns[0]
        dfn.drop(columns=[first_column_name], inplace=True)
        df.append(dfn)
        i+=1
    
    df = pd.concat(df)
    df.reset_index(drop=True, inplace=True)
    
    X = df[features]
    X[features] = X[features].apply(zscore)
    
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10) # 'k-means++' is a smart initialization method
    kmeans.fit(X)
    
    centroids_sorted_idx = np.argsort(kmeans.cluster_centers_.mean(axis=1))[::-1]
    relabel_map = np.zeros(kmeans.n_clusters, dtype=int)
    for l, old_idx in enumerate(centroids_sorted_idx):
        relabel_map[old_idx] = l
        
    sorted_labels = relabel_map[kmeans.labels_]
    df['Cluster'] = sorted_labels
    
    df['Cluster'] = df['Cluster'].clip(lower=1)
    df = df.drop(columns=['consensus'])
    df = df.set_index('Cluster')
    
    i = start; df_clustered = []
    while(i<end+1):
        dfn = df[df['season']==i]
        df_clustered.append(dfn)
        i+=1
        
    centroids_df = df.pivot_table(values=features,index='Cluster',aggfunc="mean")
    return df_clustered, centroids_df

def list_to_df(l):
    l = pd.DataFrame(l)
    l.columns = l.iloc[0];l = l.drop(0)
    l = l.apply(pd.to_numeric, errors='ignore')
    return l

def pr_gain_area(y_true,y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    order = np.argsort(-y_scores)
    y_true = y_true[order]

    P = np.sum(y_true)
    N = len(y_true) - P
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)

    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (P + eps)
    mask = (precision > 1e-10) & (recall > 1e-10)
    precision = precision[mask]
    recall = recall[mask]
    
    pi = P / (P + N)
    precision_gain = (precision - pi) / ((1 - pi) * precision + eps)
    recall_gain = (recall - pi) / ((1 - pi) * recall + eps)

    # Add (0,0) anchor as required by PRG curve definition
    precision_gain = np.concatenate(([0], precision_gain))
    recall_gain = np.concatenate(([0], recall_gain))
    mask = (
        np.isfinite(precision_gain) &
        np.isfinite(recall_gain) &
        (np.abs(precision_gain) < 100) &
        (np.abs(recall_gain) < 100)
    )

    precision_gain = precision_gain[mask]
    recall_gain = recall_gain[mask]

    # ---- HANDLE CROSSING POINTS (CRITICAL STEP) ----
    # Insert points where recall_gain crosses 0
    new_rg = []
    new_pg = []

    for i in range(len(recall_gain) - 1):
        rg1, rg2 = recall_gain[i], recall_gain[i+1]
        pg1, pg2 = precision_gain[i], precision_gain[i+1]
        new_rg.append(rg1)
        new_pg.append(pg1)
        # If crossing zero, interpolate
        if rg1 * rg2 < 0:
            t = rg1 / (rg1 - rg2)
            rg0 = 0
            pg0 = pg1 + t * (pg2 - pg1)
            new_rg.append(rg0)
            new_pg.append(pg0)

    new_rg.append(recall_gain[-1])
    new_pg.append(precision_gain[-1])

    recall_gain = np.array(new_rg)
    precision_gain = np.array(new_pg)

    # Sort by recall gain
    order = np.argsort(recall_gain)
    recall_gain = recall_gain[order]
    precision_gain = precision_gain[order]
    
    # Remove duplicates (VERY IMPORTANT)
    _, unique_idx = np.unique(recall_gain, return_index=True)
    recall_gain = recall_gain[unique_idx]
    precision_gain = precision_gain[unique_idx]

    # ---- TRAPEZOIDAL INTEGRATION ----
    area = 0.0
    for i in range(len(recall_gain) - 1):
        dx = recall_gain[i+1] - recall_gain[i]
        # skip pathological segments
        if abs(dx) > 10:
            continue
        avg_height = (precision_gain[i] + precision_gain[i+1]) / 2
        if abs(avg_height) > 100:
            continue
        area += dx * avg_height
    return area

def model_accuracy_measurement(stats_df,start,end,results_file):
    from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score
    from scipy.stats import spearmanr, kendalltau
    
    draft_outcomes = outcomes(stats_df)
    draft_outcomes = draft_outcomes.dropna()
    observed_outcomes = [['year','sample size','observed rotation','observed starter','observed all star','observed all nba',
                          'observed mvp', 'model rotation','model starter','model all star','model all nba','model mvp']]
    correlations = [['year','sample size','correlation','rmse']]
    spearman = [['year','sample size','mean DPM rho','rotation rho','starter rho','allstar rho','allnba rho']]
    kendall = [['year','sample size','mean DPM tau','rotation tau','starter tau','allstar tau','allnba tau']]
    brier_scores = [['year','sample size','rotation brier score','starter brier score','all star brier score','all nba brier score']]
    log_loss_score = [['year','sample size','rotation log loss','starter log loss','all star log loss','all nba log loss']]
    roc_auc = [['year','sample size','rotation roc auc','starter roc auc','all star roc auc','all nba roc auc']]
    pr_auc = [['year','sample size','rotation pr auc','starter pr auc','all star pr auc','all nba pr auc']]
    pr_gain = [['year','sample size','rotation pr gain','starter pr gain','all star pr gain','all nba pr gain']]
    
    i = start
    while(i<=end):
        model_pred = pd.read_excel(f'{path}/{results_file}.xlsx',f'{i}')
        model_pred = model_pred.merge(draft_outcomes[['pid','season_x','dpm']], on='pid', how='left')
        
        model_pred['dpm'] = model_pred['dpm'].fillna(-5) #-4
        model_pred[['makes NBA','rotation','starter','all star','all nba','mvp']] = model_pred[['makes NBA','rotation','starter','all star','all nba','mvp']].fillna(0)
        
        #model_pred['dpm'] = np.where(model_pred['season_x'] < i, -4, model_pred['dpm'])
        #model_pred['a_bust'] = np.where(model_pred['dpm'] <= -4, 1, 0)
        model_pred['a_rotation'] = np.where((model_pred['dpm'] >= -1), 1, 0) #& (model_pred['dpm'] <= 0)
        model_pred['a_starter'] = np.where((model_pred['dpm'] >= 0), 1, 0) #& (model_pred['dpm'] <= 1)
        model_pred['a_all star'] = np.where((model_pred['dpm'] >= 1), 1, 0) #& (model_pred['dpm'] <= 2)
        model_pred['a_all nba'] = np.where((model_pred['dpm'] >= 2), 1, 0) #& (model_pred['dpm'] <= 4.5)
        model_pred['a_mvp'] = np.where(model_pred['dpm'] >= 4.5, 1, 0)
        model_pred = model_pred.drop_duplicates(subset='pid')
        
        #manual fix for jared harper
        model_pred.loc[model_pred['pid']==45330,'dpm'] = -3
        
        #check if this is needed
        model_pred = model_pred[~(model_pred['season_x']<i)]
        model_pred2 = model_pred[(model_pred['season_x']==i)&(model_pred['dpm']>-5)]
        try: allnba_log_loss = log_loss(model_pred['a_all nba'], model_pred['all nba'])
        except ValueError: allnba_log_loss = 0
        
        observed_outcomes.append([i,len(model_pred),
                                  model_pred['a_rotation'].sum(),model_pred['a_starter'].sum(),
                                  model_pred['a_all star'].sum(),model_pred['a_all nba'].sum(),model_pred['a_mvp'].sum(),
                                  model_pred['rotation'].sum(),model_pred['starter'].sum(),
                                  model_pred['all star'].sum(),model_pred['all nba'].sum(),model_pred['mvp'].sum()])
        
        correlations.append([i,len(model_pred),model_pred['dpm'].corr(model_pred['mean DPM']),
                             np.sqrt(((model_pred['dpm'] - model_pred['mean DPM']) ** 2).mean())])
        
        spearman.append([i,len(model_pred2),
                        spearmanr(model_pred2['mean DPM'], model_pred2['dpm'])[0],
                        spearmanr(model_pred2['a_rotation'], model_pred2['dpm'])[0],
                        spearmanr(model_pred2['a_starter'], model_pred2['dpm'])[0],
                        spearmanr(model_pred2['a_all star'], model_pred2['dpm'])[0],
                        spearmanr(model_pred2['a_all nba'], model_pred2['dpm'])[0]])
        
        kendall.append([i,len(model_pred2),
                        kendalltau(model_pred2['mean DPM'], model_pred2['dpm'])[0],
                        kendalltau(model_pred2['a_rotation'], model_pred2['dpm'])[0],
                        kendalltau(model_pred2['a_starter'], model_pred2['dpm'])[0],
                        kendalltau(model_pred2['a_all star'], model_pred2['dpm'])[0],
                        kendalltau(model_pred2['a_all nba'], model_pred2['dpm'])[0]])
        
        brier_scores.append([i,len(model_pred),
                             brier_score_loss(model_pred['a_rotation'], model_pred['rotation']),
                             brier_score_loss(model_pred['a_starter'], model_pred['starter']),
                             brier_score_loss(model_pred['a_all star'], model_pred['all star']),
                             brier_score_loss(model_pred['a_all nba'], model_pred['all nba'])])
           
        log_loss_score.append([i,len(model_pred),                  
                             log_loss(model_pred['a_rotation'], model_pred['rotation']),
                             log_loss(model_pred['a_starter'], model_pred['starter']),
                             log_loss(model_pred['a_all star'], model_pred['all star']),
                             allnba_log_loss])
        
        roc_auc.append([i,len(model_pred),                     
                             roc_auc_score(model_pred['a_rotation'], model_pred['rotation'], average='weighted'),
                             roc_auc_score(model_pred['a_starter'], model_pred['starter'], average='weighted'),
                             roc_auc_score(model_pred['a_all star'], model_pred['all star'], average='weighted'),
                             roc_auc_score(model_pred['a_all nba'], model_pred['all nba'], average='weighted')])
        
        pr_auc.append([i,len(model_pred),                     
                             average_precision_score(model_pred['a_rotation'], model_pred['rotation']),
                             average_precision_score(model_pred['a_starter'], model_pred['starter']),
                             average_precision_score(model_pred['a_all star'], model_pred['all star']),
                             average_precision_score(model_pred['a_all nba'], model_pred['all nba'])])
        
        pr_gain.append([i,len(model_pred),                     
                             pr_gain_area(model_pred['a_rotation'], model_pred['rotation']),
                             pr_gain_area(model_pred['a_starter'], model_pred['starter']),
                             pr_gain_area(model_pred['a_all star'], model_pred['all star']),
                             pr_gain_area(model_pred['a_all nba'], model_pred['all nba'])])
        
        i+=1
    
    observed_outcomes = list_to_df(observed_outcomes)
    correlations = list_to_df(correlations)
    spearman = list_to_df(spearman)
    kendall = list_to_df(kendall)
    brier_scores = list_to_df(brier_scores)
    log_loss_score = list_to_df(log_loss_score)
    roc_auc = list_to_df(roc_auc)
    pr_auc = list_to_df(pr_auc)
    pr_gain = list_to_df(pr_gain)
    return observed_outcomes,correlations,spearman,kendall,brier_scores,log_loss_score,roc_auc,pr_auc,pr_gain
    
def weighted_mean(var, wts):
    return np.average(var, weights=wts)

def weighted_variance(var, wts):
    return np.average((var - weighted_mean(var, wts))**2, weights=wts)

def weighted_skew(var, wts):
    return (np.average((var - weighted_mean(var, wts))**3, weights=wts) / weighted_variance(var, wts)**(1.5))

def weighted_kurtosis(var, wts):
    return (np.average((var - weighted_mean(var, wts))**4, weights=wts) / weighted_variance(var, wts)**(2))

def calibrated_probability(n_s,n_t,p0=0.04,m=1,gamma=1.1,inflection=0.25,steepness=3,scale_high=1.15):
    p = (n_s + m*p0) / (n_t + m)
    eps = 1e-9
    p = np.clip(p,eps,1-eps)
    #p_penalized = np.power(p, gamma)
    #p_final = np.clip(p_penalized*scale_high, eps, 1-eps)
    transformed = 1/(1+ np.exp(-steepness * (p - inflection)))
    t_min = 1/(1+ np.exp(steepness * inflection))
    t_max = 1/(1+ np.exp(-steepness * (1 - inflection)))
    transformed = (transformed-t_min)/(t_max-t_min)
    p_final = np.clip(transformed*scale_high, eps, 1-eps)
    return p_final

def player_comp_analysis(x,year,p_stats,league_stats,cor_weights,print_val):
    #dist = distance2(x, year, p_stats.copy(), data.copy(), print_val)
    dist = distance2(x, year, p_stats, data, cor_weights, print_val)
    dist = dist.drop_duplicates(subset=['player'], keep='first')
    dist['weights'] = np.exp(-dist['mdist'])
    
    #pid = dist['pid'].values[0]
    p_name = dist['player'].values[0]
    team = dist['team'].values[0]
    #print(team)
    bpm = dist['bpm'].values[0]
    hgt = dist['hgt'].values[0]
    dist['hgt_pct'] = norm.pdf(dist['hgt'], loc=hgt, scale=3.4) #1-np.abs(dist['hgt'] - hgt).rank(pct=True)
    dist['bpm_pct'] = np.abs(dist['score']).rank(pct=True)
    dist.reset_index(inplace=True)
    p_class = p_stats.iloc[dist['index'].values[0]]['class']
    p_age = p_stats.iloc[dist['index'].values[0]]['age']
    #p_mp = p_stats.iloc[dist['index'].values[0]]['mp']
    p_mpp = p_stats.iloc[dist['index'].values[0]]['mp_p']
    p_gp = p_stats.iloc[dist['index'].values[0]]['GP']
    p_intl = p_stats.iloc[dist['index'].values[0]]['intl']
    
    if(print_val==0): dist = dist.loc[(dist['season']<year)] #| ((dist['player']==x) & (dist['season']==year))]
    else: dist #= dist.loc[((dist['player']==x) & (dist['season']==year))]
    
    comps = league_stats[league_stats['pid'].isin(dist['pid'])]
    nba_comps = len(comps['pid'].unique()) #- 1
    #comps = comps.groupby('player_name')['dpm'].apply(lambda x: x.nlargest(5))
    samples = 100000
    
    off_comps = comps.groupby('player_name')['o_dpm'].apply(lambda x: x.nlargest(5))
    off_comps = off_comps.groupby(level=0).mean()
    off_comps = off_comps.dropna()
    off_comps[off_comps < -3] = -3 #-2.5 old value
    
    def_comps = comps.groupby('player_name')['d_dpm'].apply(lambda x: x.nlargest(5))
    def_comps = def_comps.groupby(level=0).mean()
    def_comps = def_comps.dropna()
    def_comps[def_comps < -2] = -2 #-1.5 old value
    """
    off_comps = comps.groupby(['pid','player_name'])['o_dpm'].apply(lambda x: x.nlargest(5))
    off_comps = off_comps.groupby(level=0).mean()
    off_comps = off_comps.reset_index()
    off_comps = off_comps.drop_duplicates(subset=['pid'])
    dist = dist.merge(off_comps, on='pid', how='left')
    dist['o_dpm'] = dist['o_dpm'].fillna(-3)
    
    def_comps = comps.groupby(['pid','player_name'])['d_dpm'].apply(lambda x: x.nlargest(5))
    def_comps = def_comps.groupby(level=0).mean()
    def_comps = def_comps.reset_index()
    def_comps = def_comps.drop_duplicates(subset=['pid'])
    dist = dist.merge(def_comps, on='pid', how='left')
    dist['d_dpm'] = dist['d_dpm'].fillna(-2)
    
    exclude_pids = dist[dist['season']>=year]['pid'].to_list()
    dist_comps = dist[~dist['pid'].isin(exclude_pids)]
    dist_comps = dist_comps.drop_duplicates(subset=['pid'])
    dist_comps['score'] = np.exp(-dist_comps['mdist']*dist_comps['mdist']/2)
    """
    
    tot_comps = len(dist)
    
    try:        
        if(p_intl == 0 and p_gp <= 10): padding = np.exp(-((1-min(p_gp/25,1))))
        else: padding = 1

        p_nba1 = p_stats.loc[(p_stats['pid']==x)&(p_stats['season']==year),'pred'].values[0]
        #p_nba0 = p_stats.loc[(p_stats['pid']==x)&(p_stats['season']==year),'pred_big'].values[0]
        p_nba2 = nba_comps/tot_comps
        p_nba2 = calibrated_probability(nba_comps * padding, tot_comps)
        
        p_nba =  0.2*p_nba1*padding + 0.8*p_nba2*padding #0*p_nba0*padding
        #p_nba *= np.exp((p_nba-1)/2.5) # padding * rotation_scale_factor
        if(p_nba>=1): p_nba = 1
        
    except ZeroDivisionError: 
        p_nba = 0; padding = 0
    
    #non_nba_comps = int(tot_comps*(1-p_nba))
    #try: non_nba_comps = int((len(off_comps)/p_nba) - len(off_comps))
    #except ZeroDivisionError: non_nba_comps = tot_comps
    
    N = 150 #int(np.log(len(p_stats)/18000)*60) #int(90/rotation_scale_factor) #35
    N2 = 10 #35
    replacements = int(max(N2-len(off_comps),2))
    
    comps_list = off_comps.to_list() + [-3] * replacements
    try: comps_list = np.partition(comps_list, -N)[-N:]
    except ValueError: comps_list
    
    skewness_off = skew(comps_list) *np.exp(min(1.5*p_nba2-1,0))
    kurtosis_off = scipy.stats.kurtosis(comps_list) *np.exp(min(1.5*p_nba2-1,0))
    variance_off = np.var(comps_list) *np.exp(min(1.5*p_nba2-1,0))
    mean_off = np.mean(comps_list)
    
    comps_list = def_comps.to_list() + [-2] * replacements
    try: comps_list = np.partition(comps_list, -N)[-N:]
    except ValueError: comps_list
    
    skewness = skew(comps_list) *np.exp(min(1.5*p_nba2-1,0))
    kurtosis = scipy.stats.kurtosis(comps_list) *np.exp(min(1.5*p_nba2-1,0))
    variance = np.var(comps_list) *np.exp(min(1.5*p_nba2-1,0))
    mean = np.mean(comps_list)
    
    try: cor = off_comps.corr(def_comps)
    except: cor = 0
    
    if(print_val == 1):
        
        distribution = skewed_normal_distributions(samples, mean_off, variance_off**0.5, skewness_off, kurtosis_off, mean, variance**0.5, skewness, kurtosis, cor)
        
        bench = np.sum(np.array(distribution) >= -1)/(samples) #-1, 0, -1.75
        starter = np.sum(np.array(distribution) >= 0)/(samples) #0, 1, -0.25
        allstar = np.sum(np.array(distribution) >= 1)/(samples) #1, 2.5, 0.75
        allnba = np.sum(np.array(distribution) >= 2)/(samples) #2, 4, 1.5
        mvp = np.sum(np.array(distribution) >= 4.5)/(samples) #4.5, 7.5, 3.75
        
        p_5 = round(np.percentile(distribution, 5),2) #4
        p_50 = round(np.percentile(distribution, 50),2) #4
        meand = round(np.mean(distribution),2) #4
        p_95 = round(np.percentile(distribution, 95),2) #4
                    
        #plot of histogram
        plt.hist(distribution, bins = 200, edgecolor='green') # 'bins' controls the number of bins, 'edgecolor' adds a border
        plt.gca().yaxis.set_major_formatter(PercentFormatter((samples*samples)))

        # Best-fit curve
        #plt.xlim(-4, 6)
        #xmin, xmax = plt.xlim()
        #x2 = np.linspace(xmin, xmax, 100)
        #pdf = dist.pdf(x2)
        #plt.plot(x2, pdf, 'r', linewidth=2)

        #plt.xlim(-4, 6)
        plt.title(f'{x} {year}')
        plt.xlabel('DARKO peak')
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.show()
        
        print()
        print(x)
        print(p_name,round(p_age,2))
        print(team,year)
        print('BPM',round(bpm,2))
        print()
        #print("Bust rate",round(1-bench,4))
        print("NBA rate",round(p_nba,4))
        print("Rotation Rate",round(bench,4))
        print("Starter Rate",round(starter,4))
        print("All Star Rate",round(allstar,4))
        print("All NBA Rate",round(allnba,4))
        print("MVP Rate",round(mvp,4))
        #print("bpm percentile among comps",bpm_gap)
        #print()
        #print("Comp 1 - ",c1)
        #print("Comp 2 - ",c2)
        #print("Comp 3 - ",c3)
        #print()
        #print("offense")
        #print("mean - ",mean_off)
        #print("variance - ",variance_off)
        #print("skew - ",skewness_off)
        #print("kurt - ",kurtosis_off)
        #print("defense")
        #print("mean - ",mean)
        #print("variance - ",variance)
        #print("skew - ",skewness)
        #print("kurt - ",kurtosis)
        print()
        print("P5 DPM",p_5)
        print("median DPM",p_50)
        print("mean DPM",meand)
        print("P95 DPM",p_95)
        
        off_comps = comps.groupby(['pid','player_name'])['o_dpm'].apply(lambda x: x.nlargest(5))
        off_comps = off_comps.groupby(level=0).mean()
        off_comps = off_comps.reset_index()
        dist = dist.merge(off_comps, left_on=['pid'], right_on=['pid'], how='left')
        def_comps = comps.groupby(['pid','player_name'])['d_dpm'].apply(lambda x: x.nlargest(5))
        def_comps = def_comps.groupby(level=0).mean()
        def_comps = def_comps.reset_index()
        dist = dist.merge(def_comps, left_on=['pid'], right_on=['pid'], how='left')
        dist = dist[['pid','player', 'team', 'season', 'hgt', 'bpm', 'mdist', 'o_dpm', 'd_dpm']]
        dist['dpm'] = dist['o_dpm'] + dist['d_dpm']
        return dist
    else:
        if(p_class < 2): w = p_class
        elif(p_class == 2): w = 1.8
        elif(p_class == 3): w = 2.4
        elif(p_class == 4): w = 2.5
        else: w = 0
                
        p_mins = p_mpp * min(p_gp,40) * 40 #p_mp
        p_weight = max((p_mins/3600) * 1 * 10000, 20) #* (0.5+p_nba) #100000 is the old value
        p_mins = round(p_mins,0)
        p_weight = np.ceil(p_weight).astype(int)
        
        try:
            #distribution_off = generate_fleishman_distribution(p_weight, mean_off, variance_off**0.5, skewness_off, kurtosis_off)            
            #distribution_def = generate_fleishman_distribution(p_weight, mean, variance**0.5, skewness, kurtosis)            
            #distribution = sum_of_unique_combinations(distribution_off,distribution_def)
            
            distribution = skewed_normal_distributions_fast(int(p_weight * p_nba), mean_off, variance_off**0.5, skewness_off, kurtosis_off, mean, variance**0.5, skewness, kurtosis, cor)
            distribution = np.append(distribution, np.array([-5.] * int(p_weight * (1-p_nba))))
            #print(distribution)
            
            #p_5 = round(np.percentile(distribution, 5),2)
            #p_50 = round(np.percentile(distribution, 50),2)
            meand = round(np.mean(distribution),2)
            #p_95 = round(np.percentile(distribution, 95),2)
            
            p_nba = round(p_nba,2)
            #p_nba0 = round(p_nba0,2)
            p_nba1 = round(p_nba1,2)
            p_nba2 = round(p_nba2,2)
            
            if(p_class < 1):
                print(p_name,int(year),p_nba1,p_nba2,"HS equivalent season, excluded")
                return [p_name, x, team, p_class, p_age, year, p_weight, []]
            elif(len(distribution)==0):
                print(p_name,int(year),p_weight,p_nba1,p_nba2,"No NBA comps")
                return [p_name, x, team, p_class, p_age, year, p_weight, np.full(1, -5.0, dtype=float)]
            else:
                print(p_name,int(year),p_weight,p_nba1,p_nba2,meand)
                return [p_name, x, team, p_class, p_age, year, p_weight, distribution]
        
        except:
            print(p_name,int(year),"Error")
            return [p_name, x, team, p_class, p_age, year, p_weight, np.full(1, -5.0, dtype=float)]
    
def calculate_percentile(data_list, percentile): return np.percentile(data_list, percentile)

def mdist_list(year, p_stats, cor_weights, print_val, get_all_player_stats):
    start_time = datetime.now()
    #nba_stats_year = extract_nba_stats(year)
    nba_stats_year = nba_stats
    print()
    print(f"{year} nba stats extracted")
    
    if(get_all_player_stats == 1):
        #p_stats['adj_rating'] = data['adj_rating']
        """
        seniors = p_stats[(p_stats['season']==year)&(p_stats['GP']>=10)&
                          (((p_stats['mp']>=12) & (p_stats['bpm']>=-1) & (p_stats['class']==1))|
                          ((p_stats['mp']>=18) & (p_stats['bpm']>=2.5) & (p_stats['class']==2))|
                          ((p_stats['mp']>=21) & (p_stats['bpm']>=3) & (p_stats['class']==3))|
                          ((p_stats['mp']>=24) & (p_stats['bpm']>=4) & (p_stats['class']==4)))]
        """
        seniors = p_stats[(p_stats['season']==year)&(p_stats['GP']>=2)&(p_stats['mp']>=2)&(p_stats['class']<=4)]
        #seniors = seniors[seniors['pid']==-32861]
        names_list = seniors['pid'].to_list()
        #p_stats = p_stats.drop(columns=['adj_rating'])
    else:
        list_p = pd.read_excel(f'{path}/nba_stats.xlsx','draft list')[f'{year}']
        list_p = list_p.dropna()
        df_p = p_stats[p_stats['player'].isin(list_p)]
        #names_list = list_p['Player'].dropna().to_list()
        names_list = df_p['pid'].to_list()
        
    names_list = list(set(names_list))
    names_list.sort()
    #exceptions = list(set(names_list)-set(p_stats.player))
    #print("names not in player data")
    #print(exceptions)
    print()
    tot_len = len(names_list)
    result = [['pid','player','team','class','age','season',
               'nba','rotation','starter','all star','all nba','mvp',
               'P5 DARKO','P25 DARKO','P50 DARKO','P75 DARKO','P95 DARKO','mean DARKO']]

    if(print_val == 0):  p_stats = p_stats[p_stats['season'] <= year]
    grouped = p_stats[p_stats['season']==year].groupby('pid') #check if multiple seasons are needed
    
    for p in names_list:
        print(tot_len)
    
        if p not in grouped.groups:
            tot_len -= 1
            continue
    
        df_name = grouped.get_group(p)
    
        full_dist = []
        pteam = pclass = page = pseason = None
        p_weight = 0
        
        for x, y in df_name[['pid', 'season']].values:
            player_list = player_comp_analysis(x, y, p_stats, nba_stats_year, cor_weights, print_val)
    
            pname = player_list[0]
            pteam = player_list[2]
            pclass = player_list[3]
            page = player_list[4]
            pseason = player_list[5]
            #ppid = player_list[1] #check this
            p_weight += player_list[6]
    
            try:
                full_dist.extend(player_list[7])  # MUCH faster than +
            except (AttributeError, TypeError):
                pass
    
        if not full_dist:
            tot_len -= 1
            continue
    
        arr = np.array(full_dist)  # convert ONCE
        
        try: p_5  = np.percentile(arr, 5)
        except: p_5 = -5
        try: p_25  = np.percentile(arr, 25)
        except: p_25 = -5
        try: p_50 = np.percentile(arr, 50)
        except: p_50 = -5
        try: p_75 = np.percentile(arr, 75)
        except: p_75 = -5
        try: p_95 = np.percentile(arr, 95)
        except: p_95 = -5
        try: meand = arr.mean()
        except: meand = -5
        
        old_len = len(arr)
        arr = arr[arr > -5]
        new_len = len(arr)
        
        #pnba     = np.sum(arr > -4) / p_weight #-5
        pnba     = new_len/old_len
        pbench   = np.mean(arr >= -1) * pnba #/ p_weight #-1
        pstarter = np.mean(arr >= 0) * pnba #/ p_weight #0
        pallstar = np.mean(arr >= 1) * pnba #/ p_weight #1
        pallnba  = np.mean(arr >= 2) * pnba #/ p_weight #2
        pmvp     = np.mean(arr >= 4.5) * pnba #/ p_weight #4.5
    
        result.append([x,pname,pteam,pclass,page,pseason,pnba,pbench,pstarter,pallstar,pallnba,pmvp,p_5,p_25,p_50,p_75,p_95,meand])
    
        tot_len -= 1
    
    result = pd.DataFrame(result)
    result.columns = result.iloc[0];result = result.drop(0)
    result = result.apply(pd.to_numeric, errors='ignore')
    result = result[result['class']>0]

    pivot = result.copy()
    print()
    print("nba caliber players",round(pivot['nba'].sum(),2))
    print("rotation caliber players",round(pivot['rotation'].sum(),2))
    print("starter caliber players",round(pivot['starter'].sum(),2))
    print("all star caliber players",round(pivot['all star'].sum(),2))
    print("all nba caliber players",round(pivot['all nba'].sum(),2))
    print("mvp claiber players",round(pivot['mvp'].sum(),2))
    print()
    
    #pivot['P50 DARKO'] = round(pivot['P50 DARKO'],2)
    #pivot['P5 DARKO'] = round(pivot['P5 DARKO'],2)
    #pivot['P95 DARKO'] = round(pivot['P95 DARKO'],2)
    #pivot['mean DARKO'] = round(pivot['mean DARKO'],2)
    
    #pivot['nba'] = round(pivot['nba'],4)
    #pivot['rotation'] = round(pivot['rotation'],4)
    #pivot['starter'] = round(pivot['starter'],4)
    #pivot['all star'] = round(pivot['all star'],4)
    #pivot['all nba'] = round(pivot['all nba'],4)
    #pivot['mvp'] = round(pivot['mvp'],4)
    
    pivot = pivot[['pid','player','team','age','season','mean DARKO','nba','rotation','starter','all star','all nba','mvp','P5 DARKO','P25 DARKO','P50 DARKO','P75 DARKO','P95 DARKO']]
    pivot.columns = ['pid','player','team','age','season','mean DPM','makes NBA','rotation','starter','all star','all nba','mvp','P5 DPM','P25 DPM','median DPM','P75 DPM','P95 DPM']
    pivot = pivot.drop_duplicates(subset='pid')
    pivot = pivot.set_index('pid')
    print("total minutes for run",round((datetime.now()-start_time).total_seconds()/60,2))
    return pivot

def hyperparameter_tuning(n,test):
    #from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr, kendalltau

    draft_outcomes = outcomes(extract_nba_stats(2026))
    draft_outcomes = draft_outcomes.dropna()

    i = 0; full_draft_list = [['iteration','parameters','observed rotation','observed all star','model rotation','model all star',
                               'correlation','rmse', 'rho rotation', 'rho allstar','tau rotation', 'tau allstar']]
    full_draft_results = []

    while(i<n):
        print("iteration",i)
        if(i==0): correl_weights = get_analytical_weights(data)
        else:correl_weights = get_analytical_weights_randomized(data,i)
        
        print(correl_weights)
        #draft_list = mdist_list(latest_season, player_stats.copy(), correl_weights.copy(), 0, 0)
        draft_list = ensemble_draft_model(test,0.6,0)

        draft_list = draft_list.merge(draft_outcomes[['pid','season_x','dpm']], on='pid', how='left')
        draft_list['dpm'] = draft_list['dpm'].fillna(-5) #-4
        #draft_list['dpm'] = np.where(draft_list['season_x'] < i, -4, draft_list['dpm'])
        #draft_list['a_bust'] = np.where(draft_list['dpm'] <= -1, 1, 0)
        draft_list['a_rotation'] = np.where((draft_list['dpm'] >= -1), 1, 0) #& (draft_list['dpm'] <= 0)
        draft_list['a_starter'] = np.where((draft_list['dpm'] >= 0), 1, 0) #& (draft_list['dpm'] <= 1)
        draft_list['a_all star'] = np.where((draft_list['dpm'] >= 1), 1, 0) #& (draft_list['dpm'] <= 2)
        draft_list['a_all nba'] = np.where((draft_list['dpm'] >= 2), 1, 0) #& (draft_list['dpm'] <= 4.5)
        draft_list['a_mvp'] = np.where(draft_list['dpm'] >= 4.5, 1, 0)
        draft_list = draft_list.drop_duplicates(subset='pid')
        
        draft_list = draft_list[(draft_list['season']>=i-1)]
        draft_list[['makes NBA','rotation','starter','all star','all nba','mvp']] = draft_list[['makes NBA','rotation','starter','all star','all nba','mvp']].fillna(0)
        
        full_draft_list.append([i,correl_weights,
                                draft_list['a_rotation'].sum(),draft_list['a_all star'].sum(),draft_list['rotation'].sum(),draft_list['all star'].sum(),
                                draft_list['dpm'].corr(draft_list['mean DPM']),
                                np.sqrt(((draft_list['dpm'] - draft_list['mean DPM']) ** 2).mean()),
                                spearmanr(draft_list['a_rotation'], draft_list['mean DPM'])[0],
                                spearmanr(draft_list['a_all star'], draft_list['mean DPM'])[0],
                                kendalltau(draft_list['a_rotation'], draft_list['mean DPM'])[0],
                                kendalltau(draft_list['a_all star'], draft_list['mean DPM'])[0]])
        
        full_draft_results.append(draft_list)
        i+=1
        
    full_draft_list = list_to_df(full_draft_list)
    return full_draft_results, full_draft_list

#%% ensemble model
def ensemble_draft_model(draft_list2,w,flag):
    draft_list = mdist_list(latest_season, player_stats.copy(), correl_weights.copy(), 0, flag)
    #_,draft_list2 = regression_draft_model(nba_stats.copy(),data_full.copy(),player_stats.copy(),latest_season-3)
    
    draft_list = draft_list.reset_index()
    draft_list_full = draft_list.merge(draft_list2, on=['pid','player','team','season'], how='left')

    draft_list_full['makes NBA'] = w*draft_list_full['makes NBA'] + (1-w)*draft_list_full['pred_makes_NBA']
    draft_list_full['rotation'] = w*draft_list_full['rotation'] + (1-w)*draft_list_full['pred_rotation']
    draft_list_full['starter'] = w*draft_list_full['starter'] + (1-w)*draft_list_full['pred_starter']
    draft_list_full['all star'] = w*draft_list_full['all star'] + (1-w)*draft_list_full['pred_all_star']
    draft_list_full['all nba'] = w*draft_list_full['all nba'] + (1-w)*draft_list_full['pred_all_nba']
    draft_list_full['mvp'] = w*draft_list_full['mvp'] + (1-w)*draft_list_full['pred_mvp']

    #verify this
    draft_list_full['mean DPM'] = +6 * draft_list_full['mvp'] + \
                                  +3.5 * (draft_list_full['all nba'] - draft_list_full['mvp']) + \
                                  +1.5 * (draft_list_full['all star'] - draft_list_full['all nba']) + \
                                  +0.5 * (draft_list_full['starter'] - draft_list_full['all star']) + \
                                  -0.5 * (draft_list_full['rotation'] - draft_list_full['starter']) + \
                                  -2 * (draft_list_full['makes NBA'] - draft_list_full['rotation']) + \
                                  -5 * (1 - draft_list_full['makes NBA'])
                                
    #draft_list_full['mean DPM'] = w*draft_list_full['mean DPM'] + (1-w)*draft_list_full['pred_dpm']
    draft_list_full['P5 DPM'] = w*draft_list_full['P5 DPM'] + (1-w)*draft_list_full['pred_dpm_p05']
    draft_list_full['P25 DPM'] = w*draft_list_full['P25 DPM'] + (1-w)*draft_list_full['pred_dpm_p25']
    draft_list_full['median DPM'] = w*draft_list_full['median DPM'] + (1-w)*draft_list_full['pred_dpm_median']
    draft_list_full['P75 DPM'] = w*draft_list_full['P75 DPM'] + (1-w)*draft_list_full['pred_dpm_p75']
    draft_list_full['P95 DPM'] = w*draft_list_full['P95 DPM'] + (1-w)*draft_list_full['pred_dpm_p95']
    
    draft_list_full['makes NBA'] = round(draft_list_full['makes NBA'],4)
    draft_list_full['rotation'] = round(draft_list_full['rotation'],4)
    draft_list_full['starter'] = round(draft_list_full['starter'],4)
    draft_list_full['all star'] = round(draft_list_full['all star'],4)
    draft_list_full['all nba'] = round(draft_list_full['all nba'],4)
    draft_list_full['mvp'] = round(draft_list_full['mvp'],4)
    
    draft_list_full['mean DPM'] = round(draft_list_full['mean DPM'],2)
    draft_list_full['P5 DPM'] = round(draft_list_full['P5 DPM'],2)
    draft_list_full['P25 DPM'] = round(draft_list_full['P25 DPM'],2)
    draft_list_full['median DPM'] = round(draft_list_full['median DPM'],2)
    draft_list_full['P75 DPM'] = round(draft_list_full['P75 DPM'],2)
    draft_list_full['P95 DPM'] = round(draft_list_full['P95 DPM'],2)
    
    print()
    print("Ensemble model summary")
    print()
    print("nba caliber players",round(draft_list_full['makes NBA'].sum(),2))
    print("rotation caliber players",round(draft_list_full['rotation'].sum(),2))
    print("starter caliber players",round(draft_list_full['starter'].sum(),2))
    print("all star caliber players",round(draft_list_full['all star'].sum(),2))
    print("all nba caliber players",round(draft_list_full['all nba'].sum(),2))
    print("mvp claiber players",round(draft_list_full['mvp'].sum(),2))
    print()
    
    draft_list_full = draft_list_full[['pid', 'player', 'team', 'age', 'season', 'mean DPM', 'makes NBA', 'rotation',
                                       'starter', 'all star', 'all nba', 'mvp', 'median DPM', 'P75 DPM', 'P95 DPM']] #'P5 DPM', 'P25 DPM',
    draft_list_full = draft_list_full.set_index('pid')
    return draft_list_full

#%% call the player comparision function

#pcomps = player_comp_analysis(66098, latest_season, player_stats.copy(), nba_stats.copy(), correl_weights.copy(), 1) #Darryn Peterson

#draft_list = mdist_list(latest_season, player_stats.copy(), correl_weights.copy(), 0, 0)

#clustered_list = clustering(2016,2026,9)

#validation = model_accuracy_measurement(nba_stats,2014,latest_season-1,'results')

#full_summary, full_results = hyperparameter_tuning(30,test)

draft_list = ensemble_draft_model(test,0.6,0) #weight to mdist model, full vs representative
