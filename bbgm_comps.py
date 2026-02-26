# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:26:36 2026

@author: Subramanya.Ganti
"""
#%% initializations and functions
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

path = 'C:/Users/Subramanya.Ganti/Downloads/Sports/basketball/bbgm'

def rate_stats(df):
    df['AtRimr'] = df['AtRimFGA']/df['FGA']
    df['LowPostr'] = df['LowPostFGA']/df['FGA']
    df['MidRanger'] = df['MidRangeFGA']/df['FGA']
    return df

def distance2(pid, yr, full_matrix, test_matrix, data_copy, print_df):
    # Keep rows in data_copy where the index is present in full_matrix index
    data_copy = data_copy[data_copy.index.isin(full_matrix.index)]
    # Compute the covariance matrix and its inverse
    cov = np.ma.cov(np.ma.masked_invalid(data_copy), rowvar=False)    
    # Compute inverse covariance matrix safely
    try:
        invcov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If covariance is singular, use pseudo-inverse
        invcov = np.linalg.pinv(cov)

    # Get player data
    player_data = test_matrix.loc[(test_matrix['pid'] == pid) & (test_matrix['Season'] == yr)]

    player_index = player_data.index[0]
    #modify this line
    #player = data_copy.iloc[player_index]
    player = player_data[data_copy.columns]
    # Mask invalid values in the player vector
    pvec = np.ma.masked_invalid(np.array(player))

    # Convert data_copy to numpy array and mask invalid values
    data_array = np.ma.masked_invalid(data_copy.to_numpy())

    # Compute delta matrix by subtracting player vector from all rows
    delta_matrix = data_array - pvec

    # Compute Mahalanobis distances using einsum
    temp = np.einsum('ij,jk,ik->i', delta_matrix, invcov, delta_matrix)
    #print(temp.min(), temp.max())
    temp = np.clip(temp, 0.0, None)
    #print(temp.min(), temp.max())
    dist_array = np.sqrt(temp)

    # Set distance to self as 0
    dist_array[player_index] = 0

    # Add distances to full_matrix
    full_matrix['mdist'] = dist_array
    #prevent same item from influencing predictions
    full_matrix = full_matrix[full_matrix['mdist']>0]
    full_matrix = full_matrix[['Name', 'Age','Pos','Team', 'Season', 'mdist','Ovr', 'Pot', 'Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins',
                               'Dnk', 'FT.1', '2Pt', '3Pt', 'oIQ', 'dIQ', 'Drb', 'Pss', 'Reb', 'VORP']]
    full_matrix['score'] = 1 / (full_matrix['mdist'] ** 2) # np.exp(-full_matrix['mdist']*full_matrix['mdist']/2)
    full_matrix = full_matrix.sort_values(by='score', ascending=False)

    # Filter and sort
    score_mean = full_matrix[1:]['score'].mean()
    score_std = full_matrix[1:]['score'].std()
    full_matrix = full_matrix.loc[full_matrix['score'] >= (score_mean + 4 * score_std)] #4 or 3.75
    """
    copy = full_matrix.copy()
    if(len(copy.loc[copy['score'] >= (score_mean + 4 * score_std)])<50):
        full_matrix = full_matrix.head(50)
    else:
        full_matrix = full_matrix.loc[full_matrix['score'] >= (score_mean + 4 * score_std)] #4 or 3.75
    """
    if(print_df == 1): print(full_matrix.mean(numeric_only=True))
        
    return full_matrix

def collate(test_df, training_df, data):
    results = [['pid','player','pos','age','season','w_Age', 'w_Season', 'mdist', 'Ovr', 'Pot', 'Hgt',
           'Str', 'Spd', 'Jmp', 'End', 'Ins', 'Dnk', 'FT.1', '2Pt', '3Pt', 'oIQ','dIQ', 'Drb', 'Pss', 'Reb', 'VORP', 'score']]
    for p in test_df.values:
        print(p[1],p[7])
        try:
            dist = distance2(p[0], p[7], training_df, test_df, data.copy(), 0)
            dist2 = dist.mean(numeric_only=True)
            results.append([p[0],p[1],p[2],p[4],p[7]] + dist2.to_list())
        except:
            results.append([p[0],p[1],p[2],p[4],p[7],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    #print(results)
    results = pd.DataFrame(results[1:], columns=results[0])
    results = results.drop(['w_Season', 'mdist', 'score'], axis=1)
    return results

#%% data aggregation
training = pd.read_csv(f'{path}/training_set.csv',sep=',',low_memory=False)
test = pd.read_csv(f'{path}/test_set.csv',sep=',',low_memory=False)
test['Age'] += 2084 - test['Season'] + 1

training = rate_stats(training)
test = rate_stats(test)

data = training[['Age','MP','AtRimr', 'AtRimFGP','LowPostr', 'LowPostFGP', 'MidRanger','MidRangeFGP','3P%','3PAr','FTr', 'FT%',
                 'ORB%', 'DRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%','+/-', 'On-Off', 'ORtg', 'DRtg', 'WS/48', 'OBPM','DBPM']]

correlation_matrix = data.corr()

#%% comparision
#dist = distance2(36, 2026, training, test, data.copy(), 1)
predictions = collate(test, training, data)
#predictions = predictions.merge(test[['pid','Salary']], on='pid', how='left')
predictions = predictions.pivot_table(index=['pid', 'player','pos'],values=['age', 'season', 'w_Age', 'Ovr', 'Pot', 'Hgt', 'Str', 'Spd',
                                                                            'Jmp', 'End', 'Ins', 'Dnk', 'FT.1', '2Pt', '3Pt', 'oIQ', 'dIQ',
                                                                            'Drb', 'Pss', 'Reb','VORP'], aggfunc='mean')
predictions = predictions.reset_index()
predictions = predictions[['pid', 'player', 'pos','age', 'w_Age', 'Ovr', 'Pot', 'Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins', 'Dnk',
                           'FT.1', '2Pt', '3Pt', 'oIQ', 'dIQ', 'Drb', 'Pss', 'Reb','VORP']]
