# Performance-Analysis-of-Football-Players-using-KMeans
A Data-Driven Clustering and Over/Underperformance Analysis of Football Players

🧠 Objective
The goal of this project is to build a data-driven analytical framework to assess, segment, and rank football players from Europe's top 5 leagues (2023–24 season) using advanced statistics and unsupervised learning techniques. This analysis can help clubs, scouts, or analysts identify high-impact or underrated players with data-backed justification.

🔍 Key Components
1. Player Clustering via KMeans (Unsupervised Learning)
Players are segmented into 5 clusters using KMeans based on offensive metrics like:

Goals + Assists per 90 (G+A/90)
Expected goals + assists (xG+xAG)
Progressions (carries, passes, dribbles)

🎯 Why Clustering? To automatically group players with similar styles/impact and identify elite profiles (Cluster 2 & 3, as statistically and visually validated).

2. Overperformers from Smaller Clubs
Identifies players:

With higher actual output than expected (Overperformance_per90 > 0.2)
From non-big clubs (not in Manchester City, Real Madrid, Bayern, etc. extracted from big_clubs.csv)
Belonging to elite clusters (2 or 3)

🔎 Useful for clubs looking to find hidden gems outperforming their xG models without being in top-tier clubs.

3. Visual Analysis: G+A vs. xG+xAG
A 2D scatter plot visualizes:

Actual vs expected goal contributions
Cluster-wise player distribution
Helps identify both consistent and overachieving players

4. Underrated Talents (Low Minutes, High Output)
Filtered players:

Age under 30
Less than 10 full matches played (Playing Time_90s < 10)
But high impact (G+A per 90 > 0.5)

🧨 These are potentially underused or emerging stars with excellent returns in limited playtime.

5. Top 50 Players via Weighted Scoring System
A custom Player_Score metric was designed combining:

G+A per 90 (40%)
Overperformance vs xG (25%)
Efficiency (20%) — Output × Time
Output Index (20%) — Volume-adjusted impact
Risk Penalty (−15%) — Cards vs Contribution

🥇 Generates a ranked leaderboard of top-performing players, beyond just raw goals/assists.

🧮 Dataset
📦 Source: Kaggle - Big 5 European Soccer Player Statistics (2023–24)
https://www.kaggle.com/datasets/mamounkabbaj/2023-2024-big-5-european-soccer-player-statistics/data

Covers leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1

📊 Features

#Available Features:-

Rank: The rank of the player based on performance metrics.
Player: Name of the player.
Nation: Nationality of the player.
Position: Playing position of the player.
Squad: Club the player belongs to.
Competition: League the player is competing in.
Age: Age of the player.
Year_Born: Year the player was born.
Playing Time_MP: Matches played.
Playing Time_Starts: Matches started.
Playing Time_Min: Minutes played.
Playing Time_90s: Equivalent of 90-minute matches played.
Performance_Gls: Goals scored.
Performance_Ast: Assists.
Performance_G+A: Goals plus assists.
Performance_G-PK: Goals excluding penalties.
Performance_PK: Penalty kicks made.
Performance_PKatt: Penalty kicks attempted.
Performance_CrdY: Yellow cards.
Performance_CrdR: Red cards.
Expected_xG: Expected goals.
Expected_npxG: Non-penalty expected goals.
Expected_xAG: Expected assists.
Expected_npxG+xAG: Non-penalty expected goals plus expected assists.
Progression_PrgC: Progressive carries.
Progression_PrgP: Progressive passes.
Progression_PrgR: Progressive dribbles.
Per 90 Minutes_Gls: Goals per 90 minutes.
Per 90 Minutes_Ast: Assists per 90 minutes.
Per 90 Minutes_G+A: Goals plus assists per 90 minutes.
Per 90 Minutes_G-PK: Goals excluding penalties per 90 minutes.
Per 90 Minutes_G+A-PK: Goals plus assists excluding penalties per 90 minutes.
Per 90 Minutes_xG: Expected goals per 90 minutes.
Per 90 Minutes_xAG: Expected assists per 90 minutes.
Per 90 Minutes_xG+xAG: Expected goals plus expected assists per 90 minutes.
Per 90 Minutes_npxG: Non-penalty expected goals per 90 minutes.
Per 90 Minutes_npxG+xAG: Non-penalty expected goals plus expected assists per 90 minutes.

#Features Used:-

Category---Features Used
Performance---G+A/90, xG+xAG/90, Efficiency
Risk---Yellow Cards, Red Cards
Progression---PrgC, PrgP, PrgR
Experience---Playing Time (Min, 90s), Matches
Others---Age, Squad, Cluster

🛠️ Tools & Libraries
Python 3
Pandas, NumPy for data handling
Sklearn for clustering and scaling
Matplotlib, Seaborn for visualization

📌 Final Takeaways
✅ Identified elite performers using unsupervised clustering.
✅ Surface overperformers from underrated clubs.
✅ Created a custom scoring system for advanced ranking.
✅ Discovered underrated but high-potential players.
✅ Delivered visuals and metrics for clear insights.

📁 Files Included

big_5_players_stats_2023_2024.csv – Raw dataset
big_clubs.csv – List of major clubs for filtering
Football data.py – Main notebook/script with analysis
README.md – Project summary and insights

