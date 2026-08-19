import pandas as pd
import os
'''

    Fills in missing timing values

'''
year = 2026
df = pd.read_csv(f"data/Driver_race_data_{year}.csv")
df = df.sort_values(by=["Year","Round","GridPos"],ascending=[True,True,True])

time_cols = ['AvgFPTime','QualyTime', 'QualTimeDelta']
print(df.isnull().sum())

def smart_impute_grid(group):
    
    gaps = group[time_cols].diff()
    mean_gaps = gaps.mean()
    
   
    mean_gaps = mean_gaps.fillna(0) 
    
   
    for i in range(len(group)):
        # Target rows where the time is NaN
        if pd.isna(group.iloc[i]['QualyTime']) or pd.isna(group.iloc[i]['AvgFPTime']):
            if i > 0: 
                
                group.iloc[i, group.columns.get_indexer(time_cols)] = (
                    group.iloc[i - 1][time_cols] + mean_gaps
                )
            else:
               
                group[time_cols] = group[time_cols].bfill()
                
    return group

df_cleaned = df.groupby(['Year', 'Round'], group_keys=False).apply(smart_impute_grid)

df_cleaned['IsAccurate'] = df_cleaned['IsAccurate'].fillna(False).astype(bool)


df_cleaned = df_cleaned.sort_values(by=["Year","Round","FinishPos"],ascending=[True,True,True])
df_cleaned.to_csv(f"data/Driver_race_data_{year}.csv",index = False)


dfnew = pd.read_csv(f"data/Driver_race_data_{year}.csv")
print(dfnew.isnull().sum())

