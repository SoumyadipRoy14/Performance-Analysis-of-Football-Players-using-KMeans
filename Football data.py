#!/usr/bin/env python
# coding: utf-8

# # A Data-Driven Clustering and Over/Underperformance Analysis of Football Players

# Build a data-driven framework to:
# 
# 1.Segment players into performance clusters using unsupervised learning (e.g., KMeans)
# 
# 2.Identify overperformers and underperformers compared to expected contributions
# 
# 3.Highlight “hidden gems” — players with high output relative to their playing time and expectations, but possibly low media presence (non-top-ranked)

# What types of players exist in the dataset based on their performance profiles?
# 
# Which players are consistently outperforming expectations (xG + xAG)?
# 
# Are there underrated players based on contribution vs minutes played?

# In[1]:


import pandas as pd


# In[77]:


football=pd.read_csv('big_5_players_stats_2023_2024.csv')
football


# In[22]:


football.head()


# In[8]:


for col in football.columns:
 print(col)


# In[9]:


football.shape


# In[11]:


football['Nation'].value_counts()


# In[13]:


football['Squad'].value_counts()


# In[19]:


import numpy as np


# In[78]:


football['Playing Time_90s']


# In[79]:


football = football[pd.to_numeric(football['Playing Time_90s'], errors='coerce').notna()]


# In[80]:


football.reset_index(inplace=True)
football


# In[81]:


football['Playing Time_90s']=football['Playing Time_90s'].astype(float)
football['Playing Time_90s']


# In[82]:


football = football[football['Playing Time_90s'] >= 5.0]
football


# In[83]:


football['Per 90 Minutes_G+A']=football['Per 90 Minutes_G+A'].astype(float)
football['Per 90 Minutes_xG+xAG']=football['Per 90 Minutes_xG+xAG'].astype(float)

football['Performance_CrdY']=football['Performance_CrdY'].astype(float)
football['Performance_CrdR']=football['Performance_CrdR'].astype(float)
football['Performance_G+A']=football['Performance_G+A'].astype(float)


# In[84]:


# Feature engineering
football['Overperformance_per90'] = football['Per 90 Minutes_G+A'] - football['Per 90 Minutes_xG+xAG']
football['Output_Index'] = football['Per 90 Minutes_G+A'] * np.log(football['Playing Time_90s'])
football['Risk_Factor'] = (football['Performance_CrdY'] + 2 * football['Performance_CrdR']) / football['Performance_G+A'].replace(0, np.nan)


# In[92]:


football['Expected_npxG+xAG']=football['Expected_npxG+xAG'].astype(float)
football['Progression_PrgC']=football['Progression_PrgC'].astype(float)
football['Progression_PrgP']=football['Progression_PrgP'].astype(float)
football['Progression_PrgR']=football['Progression_PrgR'].astype(float)


# In[93]:


features = [
    'Per 90 Minutes_G+A', 'Per 90 Minutes_xG+xAG',
    'Expected_npxG+xAG', 'Progression_PrgC',
    'Progression_PrgP', 'Progression_PrgR'
]
X = football[features].fillna(0)


# In[94]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[95]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[97]:


kmeans = KMeans(n_clusters=5, random_state=42)
football['Cluster'] = kmeans.fit_predict(X_scaled)


# In[98]:


big_clubs = pd.read_csv('big_clubs.csv')
big_clubs


# In[100]:


football['Cluster']


# In[101]:


football['Cluster'].value_counts()


# In[103]:


top_gems = football[
    (football['Overperformance_per90'] > 0.3) &
    (~football['Squad'].isin(big_clubs['Squad'])) &  # example big clubs
    (football['Cluster'].isin([2, 3]))  # example cluster with elite performance
]


# After clustering players using KMeans, not all clusters are equally talented or relevant depending on your goal (e.g., scouting attacking threats, finding underrated creators).
# 
# Clusters 2 and 3 were visually or statistically identified (via metrics like G+A/90, xG, progressions, etc.) as "elite" performers — i.e., top scorers, playmakers, or all-around offensive threats.

# In[104]:


print(top_gems[['Player', 'Squad', 'Per 90 Minutes_G+A', 'Per 90 Minutes_xG+xAG', 'Overperformance_per90', 'Cluster']])


# In[109]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[114]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=football, x='Per 90 Minutes_G+A', y='Per 90 Minutes_xG+xAG', hue='Cluster', palette='Set2')
plt.title('Player Clusters: Actual vs Expected Goal Contributions per 90')
plt.xlabel('G+A per 90')
plt.ylabel('xG + xAG per 90')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[119]:


football['Efficiency'] = football['Per 90 Minutes_G+A'] * football['Playing Time_90s']


# In[127]:


football['Playing Time_Min']=football['Playing Time_Min'].astype(float)
football['Age']=football['Age'].astype(float)


# In[129]:


underrated_gems = football[
    (football['Playing Time_90s'] < 10) &
    (football['Playing Time_Min'] < 900) &
     (football['Age'] < 30) &
    (football['Per 90 Minutes_G+A'] > 0.4)
].sort_values(by='Efficiency', ascending=False)


# In[125]:


print("Underrated High Performers (Limited minutes, high output):")
print(underrated_gems[['Player', 'Squad', 'Playing Time_90s', 'Playing Time_Min','Per 90 Minutes_G+A', 'Efficiency']])


# In[130]:


football['Player_Score'] = (
    0.4 * football['Per 90 Minutes_G+A'] +
    0.25 * football['Overperformance_per90'] +
    0.2 * football['Efficiency'] / football['Efficiency'].max() +
    0.2 * football['Output_Index'] / football['Output_Index'].max() -
    0.15 * football['Risk_Factor'].fillna(0)
)
football


# In[135]:


football['Rank'] = football['Player_Score'].rank(ascending=False)
football


# In[140]:


# **Sort top performers by Player Score**
top_ranked = football.sort_values(by='Player_Score', ascending=False)[
    ['Player', 'Squad', 'Per 90 Minutes_G+A', 'Overperformance_per90', 'Efficiency', 'Output_Index', 'Risk_Factor', 'Player_Score', 'Rank']
].head(50)


# In[141]:


print("Top Ranked Players Based on Weighted Score:")
print(top_ranked)

