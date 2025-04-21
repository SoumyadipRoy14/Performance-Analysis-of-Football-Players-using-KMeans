# Performance-Analysis-of-Football-Players-using-KMeans
A Data-Driven Clustering and Over/Underperformance Analysis of Football Players

**Objective**:- The goal of this project is to design an analytical pipeline to assess and rank football players based on advanced performance metrics. The framework focuses on identifying both elite and underrated talents using unsupervised learning and custom scoring logic. The key components of this objective include: 

1.Segment players into performance clusters using unsupervised learning (KMeans)
Used K-Means to group players based on key stats (G+A/90, xG+xAG/90, progressions etc.) into performance-based clusters.

2.Identify Overperformers from Smaller Clubs
Tried to figured out the players outside big clubs who exceed expected output and belong to Cluster 2 & 3, which showed elite attacking profiles.

3.Plot the Contributed Goals+Assists with Expected Goals+Assists based on the Clusters
Plotted G+A/90 vs. xG+xAG/90, color-coded by cluster, to highlight overachievers and high-impact players.

4.Find the underrated players based on contribution, Age, Matches & minutes played.
Filtered young players (<30) with limited minutes but high G+A/90 to identify underutilized, high-efficiency gems.


5.Find the top 50 highest-performing players based on a weighted score.
A custom formula is used combining output, overperformance, efficiency, and risk to rank the top 50 players.
