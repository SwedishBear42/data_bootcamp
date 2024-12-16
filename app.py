dimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import streamlit as st
import statsmodels.api as sm
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from scipy.stats import f_oneway
from scipy.stats import spearmanr


# Automatically install requirements
requirements_file = "requirements.txt"
if os.path.exists(requirements_file):
    subprocess.run(["pip", "install", "-r", requirements_file])

# Set custom page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        * {
            font-family: 'Garamond', serif;
        }
        .css-18e3th9 {
            font-family: 'Garamond', serif;  /* Adjusting Streamlit's main font */
        }
        .css-1d391kg {
            font-family: 'Garamond', serif; /* Adjusting Streamlit's sidebar font */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fantasy Football Data Analysis")
st.header("By Nico Lupin")
st.subheader("nml5639@nyu.edu")


#Description
st.markdown("""#### Problem Statement
One of my longest standing and most exciting hobbies has been my fantasy football league which is now entering its 6th year with the same group of friends from middle school. Now as a I become more and more interested in data analytics, analyzing fantasy football data has become a compelling objective as one of the more data driven aspects of being a football fan. For my final project, I am going to be building predictive models capable of uncovering the strategies and actions that have consistently led to winning outcomes in my league. This involves draft choices, player statistics, and draft order as well as an anlysis of each individual team in the league to identify patterns and factors that have led to their outperformance. By analyzing player data, team composition and construction, and weekly performance, this project will seek to construct the first comprehensive data-driven understanding of my fantasy football team dynamics with the hope that next year, I can finally win the league after once again missing playoff contention this year. 
Dataset Description
The dataset utilized is sourced from a github library called "espn_api" which interfaces with the actual ESPN Fantasy Football API to extract comprehensive data from each one of my league's seasons which have each been assigned their own object below. The datasets encompass weekly player statistics, weekly team statistics, and general league statistics for the last 4 seasons of football, 2021-2024
Source: https://github.com/cwendt94/espn-api. I then cleaned the data myself and have saved the combined data to 3 csv's on my computer that combine the past 4 years into comprehensive player level, team-level, and matchup data. These datasets encompass 762 player rows, 12 teams per year, and 384 matchups over 4 years. They can be linked with the season, team_id, and player_id columns.
""")

# Load CSV Files from GitHub
@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/SwedishBear42/data_bootcamp/main/"  # Update with your GitHub details
    players_df = pd.read_csv(base_url + "players.csv")
    teams_df = pd.read_csv(base_url + "teams.csv")
    
    return players_df, teams_df

# Load Data
players_df, teams_df = load_data()
# #### Player Analysis Data Description

st.header("Player Analysis Data Description")

st.markdown("""
We will first inspect the dataframes and create a few visual plots that can be used to help us understand the structure and makeup of the data.
""")

# Display the first few rows of teams_df and players_df
st.subheader("Dataframe Heads")
st.write("### Teams DataFrame")
st.dataframe(teams_df.head())

st.write("### Players DataFrame")
st.dataframe(players_df.head())

# Display the data type and sample entries of 'eligible_slots'
st.subheader("Eligible Slots Sample")
st.write(players_df['eligible_slots'].head(10))

# Distribution of Total Points
st.subheader("Distribution of Total Points")
fig1, ax1 = plt.subplots()
sns.histplot(players_df, x='total_points', ax=ax1)
ax1.set_title('Distribution of Total Points')
st.pyplot(fig1)

# Distribution of Eligible Slots
st.subheader("Distribution of Eligible Slots")
fig2, ax2 = plt.subplots()
sns.histplot(players_df, x='eligible_slots', ax=ax2)
ax2.set_title('Distribution of Eligible Slots')
st.pyplot(fig2)

# ANOVA and Position Rank Distribution by Drafted Team
st.subheader("Position Rank Distribution by Drafted Team")

# Filter data for valid drafted team and position rank
valid_players = players_df.dropna(subset=['drafted_team_id', 'pos_rank'])

# Perform ANOVA
groups = [group['pos_rank'].values for _, group in valid_players.groupby('drafted_team_id')]
f_stat, p_value = f_oneway(*groups)

# Calculate medians and sort drafted teams by median position rank
medians = valid_players.groupby('drafted_team_id')['pos_rank'].median().sort_values()
sorted_teams = medians.index  # Sorted order of drafted_team_id by median

st.write(f"**F-statistic:** {f_stat:.4f}, **P-value:** {p_value:.4e}")

# Boxplot to visualize position rank distribution by team
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=valid_players, 
    x='drafted_team_id', 
    y='pos_rank',
    order=sorted_teams,
    ax=ax3
)
ax3.set_title('Position Rank Distribution by Drafted Team')
ax3.set_xlabel('Drafted Team ID')
ax3.set_ylabel('Position Rank')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
st.pyplot(fig3)

# Trade Frequency and Final Standing Correlation
st.subheader("Relationship Between Number of Trades and Final Standing")

# Count the # of trades for each team_id in players_df
trade_counts = players_df[players_df['acquisition_type'] == 'TRADE'].groupby('team_id').size().reset_index(name='trade_count')

# Merge trade counts with teams_df to include final standings
teams_with_trades = teams_df[['team_id', 'season', 'final_standing']].merge(
    trade_counts, 
    on='team_id', 
    how='inner'
)

# Correlation
correlation, p_val = spearmanr(teams_with_trades['trade_count'], teams_with_trades['final_standing'])
st.write(f"**Spearman Correlation:** {correlation:.4f}, **P-value:** {p_val:.4e}")

# Scatterplot to visualize the relationship
fig4, ax4 = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    data=teams_with_trades, 
    x='trade_count', 
    y='final_standing',
    ax=ax4
)
ax4.set_title('Relationship Between Number of Trades and Final Standing', fontsize=16)
ax4.set_xlabel('Number of Trades', fontsize=14)
ax4.set_ylabel('Final Standing', fontsize=14)
ax4.grid(True)
st.pyplot(fig4)

# #### Average Total Points by Draft Round for QBs

st.header("Average Total Points by Draft Round for QBs")

st.markdown("""
This bar chart helps us understand the total points scored by QBs based on their average drafted round.
""")

# Step 1: Filter QB players and calculate stats by round
qbs = players_df[players_df['eligible_slots'].str.contains('QB')].dropna(subset=['round_num', 'total_points'])

# Step 2: Group QBs by round_num
round_stats = qbs.groupby('round_num')['total_points'].agg(['mean', 'count'])

# Step 3: Identify where the dropoff in performance happens (visual inspection)
fig5, ax5 = plt.subplots(figsize=(12, 6))

# Barplot: Average total points per round
sns.barplot(x=round_stats.index, y=round_stats['mean'], ax=ax5)

# Overlay: Number of QBs drafted as text annotations
for i, count in enumerate(round_stats['count']):
    ax5.text(i, round_stats['mean'].iloc[i] + 5, f'{count}', ha='center', fontsize=10, color='black')

# Formatting
ax5.set_title('Average Total Points by Draft Round for QBs')
ax5.set_xlabel('Draft Round')
ax5.set_ylabel('Average Total Points')
ax5.axhline(qbs['total_points'].mean(), color='red', linestyle='--', label='Overall Average Total Points')
ax5.legend()

st.pyplot(fig5)

# #### Correlation Between Final Standing and Various Metrics

st.header("Correlation Between Final Standing and Various Metrics")

st.markdown("""
This section explores the correlation between the final standings of teams and different performance metrics, including points scored, points against, draft projections, and point differential.
""")

# Calculate point differential
teams_df['point_differential'] = teams_df['points_for'] - teams_df['points_against']

# Setting up the figure with subplots
fig6, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: points_for vs final_standing
sns.scatterplot(x='points_for', y='standing', data=teams_df, ax=axes[0, 0])
axes[0, 0].set_title('Points For vs Final Standing')
axes[0, 0].set_xlabel('Points For')
axes[0, 0].set_ylabel('Final Standing')

# Plot 2: points_against vs final_standing
sns.scatterplot(x='points_against', y='standing', data=teams_df, color='orange', ax=axes[0, 1])
axes[0, 1].set_title('Points Against vs Final Standing')
axes[0, 1].set_xlabel('Points Against')
axes[0, 1].set_ylabel('Final Standing')

# Plot 3: draft_projected_rank vs final_standing
sns.scatterplot(x='draft_projected_rank', y='standing', data=teams_df, color='green', ax=axes[1, 0])
axes[1, 0].set_title('Draft Projected Rank vs Final Standing')
axes[1, 0].set_xlabel('Draft Projected Rank')
axes[1, 0].set_ylabel('Final Standing')

# Plot 4: point_differential vs final_standing
sns.scatterplot(x='point_differential', y='standing', data=teams_df, color='purple', ax=axes[1, 1])
axes[1, 1].set_title('Point Differential vs Final Standing')
axes[1, 1].set_xlabel('Point Differential')
axes[1, 1].set_ylabel('Final Standing')

# Adjust layout for better spacing
plt.tight_layout()
st.pyplot(fig6)

st.markdown("""
These relationships depict that ESPN's projected final standings on draft day are statistically insignificant and should not be taken at face value. The other scatter plots provide more meaningful insights, with the point differential scatter plot showing the clearest relationship. This makes sense as wins are predicated on point differential. Comparing the points_for and points_against scatter plots helps us understand the variance among teams and the factors we can control, which will guide the rest of our analysis.
""")

# #### Regression Analysis: Points For and Points Against vs Final Standing

st.header("Regression Analysis: Points For and Points Against vs Final Standing")

st.markdown("""
This section explores whether regressing both `points_for` and `points_against` can determine which variable has more weight in predicting the `final_standing` of teams.
""")

# Standardize variables for comparability
scaler = StandardScaler()
teams_df[['points_for', 'points_against', 'standing']] = scaler.fit_transform(
    teams_df[['points_for', 'points_against', 'standing']]
)

# Define independent and dependent variables
X = teams_df[['points_for', 'points_against']]
y = teams_df['standing']

# Add a constant to the model
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display regression summary
st.subheader("Regression Model Summary")
with st.expander("View Model Summary"):
    st.text(model.summary().as_text())

# Regression plots for each variable
st.subheader("Regression Plots")

fig7, axes7 = plt.subplots(1, 2, figsize=(14, 6))

# Plot regression for points_for vs final_standing
sns.regplot(x='points_for', y='standing', data=teams_df, ax=axes7[0])
axes7[0].set_title('Points For vs Final Standing')
axes7[0].set_xlabel('Points For (Standardized)')
axes7[0].set_ylabel('Final Standing (Standardized)')

# Plot regression for points_against vs final_standing
sns.regplot(x='points_against', y='standing', data=teams_df, ax=axes7[1], color='orange')
axes7[1].set_title('Points Against vs Final Standing')
axes7[1].set_xlabel('Points Against (Standardized)')
axes7[1].set_ylabel('Final Standing (Standardized)')

plt.tight_layout()
st.pyplot(fig7)

st.markdown("""
Based on the regression coefficients, we can calculate the relative contribution of each variable by dividing the absolute value of each coefficient by the sum of the coefficients. 
- **Points For** contributes approximately 55% of the variance in the two-variable model to the final standing.
- **Points Against** contributes approximately 45%.

This indicates that while the strength of our own team (`points_for`) is slightly more influential in dictating our final standing, a significant portion (`45%`) is influenced by factors outside our control (`points_against`).
Since this analysis focuses on using statistics to construct the optimal fantasy football strategy, the emphasis will be on maximizing our own point totals.
""")
# #### Relationship Between Draft Number and Total Points for RBs & WRs

st.header("Univariate & Multivariate Linear Regression Models")

st.markdown("""
This section delves into whether a player's relative strength can be predicted from the onset and how preconceived notions of their performance influence their actual performance. Multiple regressions are run to determine if specific stats denote total points outperformance or position rank, and whether draft position serves as a good predictor for ultimate success. Additionally, we examine if a player's actual NFL team performance affects individual player outperformance.
""")
# Section 2: Draft Number vs Total Points (RB & WR)
st.write("### Draft Number vs Total Points (RB & WR)")
positions_of_interest = ["RB, RB/WR", "WR, RB/WR"]

# Filter RB/WR Data
filtered_players = players_df[players_df['eligible_slots'].isin(positions_of_interest)].copy()
filtered_players.dropna(subset=['total_points', 'round_num', 'round_pick'], inplace=True)
filtered_players['draft_number'] = filtered_players['round_num'] * filtered_players['round_pick']

# Drafted Players
drafted_players = filtered_players[filtered_players['draft_number'] > 0].copy()
undrafted_players = filtered_players[filtered_players['draft_number'] == 0].copy()

st.write(f"Number of drafted players: {drafted_players.shape[0]}")
st.write(f"Number of undrafted players: {undrafted_players.shape[0]}")

# Plot Drafted Players vs Undrafted Players
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# Drafted
sns.scatterplot(
    data=drafted_players,
    x='draft_number',
    y='total_points',
    hue='eligible_slots',
    palette='viridis',
    ax=axes[0],
    s=100, alpha=0.7
)
axes[0].set_title('Drafted Players')
axes[0].set_xlabel('Draft Number')
axes[0].set_ylabel('Total Points')

# Undrafted
sns.scatterplot(
    data=undrafted_players,
    x='draft_number',
    y='total_points',
    hue='eligible_slots',
    palette='viridis',
    ax=axes[1],
    s=100, alpha=0.7
)
axes[1].set_title('Undrafted Players')
axes[1].set_xlabel('Draft Number')

plt.tight_layout()
st.pyplot(fig)

# Top 10 Undrafted Players
st.write("#### Top 10 Undrafted Players")
if 'team_name' in undrafted_players.columns:
    top_10_undrafted = undrafted_players.sort_values(by='total_points', ascending=False).head(10)
    st.dataframe(top_10_undrafted[['player_id', 'player_name', 'eligible_slots', 'season', 'team_name', 'total_points']])
else:
    st.error("Missing required column: 'team_name'")

# Section 3: Draft Number Linear Regression
st.write("### Linear Regression: Draft Number vs Total Points")

# Regression Analysis for Drafted Players
X_drafted = drafted_players[['draft_number']]  # Ensure X is a DataFrame
X_drafted = sm.add_constant(X_drafted)        # Adds the constant term
y_drafted = drafted_players['total_points']

# Fit the OLS regression model
model_drafted_1 = sm.OLS(y_drafted, X_drafted).fit()

# Display the regression summary in an expander
with st.expander("View Model Summary"):
    st.text(model_drafted_1.summary())

# Regression Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=drafted_players,
    x='draft_number',
    y='total_points',
    scatter_kws={'s': 100, 'alpha': 0.5},
    line_kws={'color': 'red'},
    ax=ax
)
plt.title('Draft Number vs Total Points')
plt.xlabel('Draft Number')
plt.ylabel('Total Points')

# Add Regression Annotations
slope = model_drafted_1.params['draft_number']
intercept = model_drafted_1.params['const']
r_squared = model_drafted_1.rsquared
p_value = model_drafted_1.pvalues['draft_number']

annotation = f"Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nR-squared: {r_squared:.3f}\nP-value: {p_value:.3e}"
plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
             backgroundcolor='white', verticalalignment='top')
st.pyplot(fig)

# Calculate Percentage of Points Above the Regression Line
y_pred = model_drafted_1.predict(X_drafted)          # Correct model reference
above_line = np.sum(y_drafted > y_pred)
total_points = len(y_drafted)
percentage_above = (above_line / total_points) * 100

# Display the percentage
st.write(f"**Percentage of points above the regression line:** {percentage_above:.2f}%")
st.markdown("""
This means that 47% of the players that have been drafted had a higher value than their predicted draft value! This should dispel fears during a draft that all the "good" players have been taken. Now let's investigate if this same relationship exists with Quarterbacks.
""")

# Section 4: QB Draft Relationship
st.write("### QB Draft Relationship")
positions_of_interest_qb = ["QB"]
filtered_qbs = players_df[players_df['eligible_slots'].isin(positions_of_interest_qb)].copy()
filtered_qbs.dropna(subset=['total_points', 'round_num', 'round_pick'], inplace=True)
filtered_qbs['draft_number'] = filtered_qbs['round_num'] * filtered_qbs['round_pick']

drafted_qbs = filtered_qbs[filtered_qbs['draft_number'] > 0].copy()
st.write(f"Number of drafted QBs: {drafted_qbs.shape[0]}")

# Regression for QBs
X_qb = drafted_qbs['draft_number']
y_qb = drafted_qbs['total_points']
X_qb = sm.add_constant(X_qb)

model_qb = sm.OLS(y_qb, X_qb).fit()
with st.expander("View Model Summary"):
    st.text(model_qb.summary())

# Regression Plot for QBs
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=drafted_qbs,
    x='draft_number',
    y='total_points',
    scatter_kws={'s': 100, 'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.title('Draft Number vs Total Points (QBs)')
plt.xlabel('Draft Number')
plt.ylabel('Total Points')

# QB Regression Annotations
slope_qb = model_qb.params['draft_number']
intercept_qb = model_qb.params['const']
r_squared_qb = model_qb.rsquared
p_value_qb = model_qb.pvalues['draft_number']

annotation_qb = f"Slope: {slope_qb:.2f}\nIntercept: {intercept_qb:.2f}\nR-squared: {r_squared_qb:.3f}\nP-value: {p_value_qb:.3e}"
plt.annotate(annotation_qb, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
             backgroundcolor='white', verticalalignment='top')
st.pyplot(fig)



# #### Creating Position-specific DataFrames

st.header("Creating Position-specific DataFrames")

# Define the positions of interest for creating separate DataFrames
positions = ['QB', 'TE, RB/WR/TE', 'RB, RB/WR', 'WR, RB/WR', 'D/ST', 'K']
position_dfs = {}

# Expander to display the number of players per position
with st.expander("View Number of Players per Position"):
    # Loop through each position
    for pos in positions:
        pos_df = players_df[players_df['eligible_slots'] == pos].copy()
        pos_df.dropna(subset=['total_points', 'round_num', 'round_pick'], inplace=True)
        pos_df['draft_number'] = pos_df['round_num'] * pos_df['round_pick']
        position_dfs[pos] = pos_df
        
        st.write(f"**Number of {pos} players:** {pos_df.shape[0]}")

# #### Encoding 'acquisition_type' Variable

st.header("Encoding 'acquisition_type' Variable")

# Acquisition_type is a categorical variable with ADD, TRADE, DRAFTED so let's encode it
encoder = OneHotEncoder(drop='first', sparse_output=False)

def encode_acquisition_type(df):
    if 'acquisition_type' in df.columns:
        acquisition_type = df[['acquisition_type']]
        
        encoded = encoder.fit_transform(acquisition_type)
        
        encoded_cols = encoder.get_feature_names_out(['acquisition_type'])
        
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        
        df = pd.concat([df, encoded_df], axis=1)
        
        df.drop('acquisition_type', axis=1, inplace=True)
    
    return df

# Expander to display encoding process
with st.expander("View Acquisition Type Encoding"):
    # Loop through each position and encode 'acquisition_type'
    for pos, df in position_dfs.items():
        position_dfs[pos] = encode_acquisition_type(df)
        st.write(f"**Encoded 'acquisition_type' for {pos} players.**")

    st.markdown("### Example Encoded DataFrame")
    selected_pos = st.selectbox("Select a position to view its encoded DataFrame", positions)
    st.dataframe(position_dfs[selected_pos].head())

    # #### Defining Player Statistic Variables

st.header("Defining Player Statistic Variables")

# Defining the specific player statistic variables for the subsequent analysis
common_vars = [
    'drafted_team_id',
    'draft_number',
    'actual_teamWin',
    'actual_teamLoss',
    'actual_pointsScored',
    'actual_fumbles',
    'percent_owned',
    'percent_started',
    'on_team_id'
]

# Position-specific variables
position_specific_vars = {
    'QB': [
        'actual_passingAttempts',
        'actual_passingCompletions',
        'actual_passingIncompletions',
        'actual_passingYards',
        'actual_passingTouchdowns',
        'actual_passingInterceptions',
        'actual_passingCompletionPercentage',
        'actual_turnovers'
    ],
    'TE, RB/WR/TE': [
        'actual_receivingReceptions',
        'actual_receivingYards',
        'actual_receivingTouchdowns',
        'actual_receivingTargets',
        'actual_receivingYardsAfterCatch',
        'actual_receivingYardsPerReception'
    ],
    'WR, RB/WR': [
        'actual_receivingReceptions',
        'actual_receivingYards',
        'actual_receivingTouchdowns',
        'actual_receivingTargets',
        'actual_receivingYardsAfterCatch',
        'actual_receivingYardsPerReception',
    ],
    'RB, RB/WR': [
        'actual_rushingTouchdowns',
        'actual_rushingAttempts',
        'actual_rushingYards',
        'actual_rushingYardsPerAttempt'
    ],
    'D/ST': [
        'actual_defensiveTouchdowns',
        'actual_defensiveInterceptions',
        'actual_defensiveSacks',
        'actual_defensiveForcedFumbles',
        'actual_defensivePointsAllowed',
        'actual_defensiveYardsAllowed',
        'actual_defensiveTotalTackles'
    ],
    'K': [
        'actual_madeFieldGoals',
        'actual_attemptedFieldGoals',
        'actual_missedFieldGoals',
        'actual_madeExtraPoints',
        'actual_attemptedExtraPoints',
        'actual_missedExtraPoints'
    ]
}

position_vars = {}

# Loop through each position and define variables
for pos in position_dfs.keys():
    vars_list = common_vars.copy()
    vars_list += position_specific_vars.get(pos, [])
    encoded_cols = [col for col in position_dfs[pos].columns if 'acquisition_type_' in col]
    vars_list += encoded_cols
    # Exclude 'total_points' as it's the dependent variable
    vars_list = [var for var in vars_list if var != 'total_points']
    position_vars[pos] = vars_list
    st.write(f"**Variables for {pos}:**")
    st.write(vars_list)

    # #### Section 4: Regression Analysis for Player Positions

st.header("Section 4: Regression Analysis for Player Positions")

st.markdown("""
In this section, we aim to identify the best combination of player statistics that can predict the total points for each respective position with the highest statistical significance. Using the `find_all_regressions` function, we systematically evaluate all possible combinations of up to 3 independent variables and select the model with the highest adjusted R² to penalize for overfitting. Generally, we are looking to find that the independent variables identified are the player statistics that directly contribute the most to points, such as touchdowns, reception yards, and rushing yards.
""")

# #### Defining the Regression Function

def find_all_regressions(df, independent_vars, dependent_var='total_points', max_vars=3):
    """
    Finds all possible regressions for combinations of independent variables (up to max_vars)
    and calculates Adjusted R-squared for each.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - independent_vars (list): List of independent variable names.
    - dependent_var (str): The name of the dependent variable.
    - max_vars (int): Maximum number of variables in the regression.

    Returns:
    - results (list): A list of tuples containing:
        (variable combination, adjusted R-squared)
    """
    from itertools import combinations
    results = []

    for k in range(1, max_vars + 1):
        for combo in combinations(independent_vars, k):
            subset = df[list(combo) + [dependent_var]].replace([np.inf, -np.inf], np.nan).dropna()

            # Define independent and dependent variables
            X = subset[list(combo)]
            y = subset[dependent_var]
            X = sm.add_constant(X)

            # Fit the model
            try:
                model = sm.OLS(y, X).fit()
                adj_r2 = model.rsquared_adj

                # Append the combination and Adjusted R-squared to results
                results.append((combo, adj_r2))
            except Exception as e:
                continue

    return results

# #### Performing Regression Analysis for Each Position

st.subheader("Regression Analysis Results")

# Initialize a dictionary to store all regression results for each position
all_regression_results = {}

# Initialize a dictionary to store the top 10 regression results for each position
top_10_regression_results = {}

# Iterate through each position
for pos, df in position_dfs.items():
    st.markdown(f"### Regression Analysis for **{pos}**")
    
    vars_list = position_vars.get(pos, [])
    
    if not vars_list:
        st.warning(f"No independent variables found for position {pos}. Skipping regression analysis.")
        continue
    
    # Perform regression analysis
    with st.spinner(f"Performing regression analysis for {pos}..."):
        regression_results = find_all_regressions(
            df,
            independent_vars=vars_list,
            dependent_var='total_points',
            max_vars=3
        )
    
    # Check if any regression results were obtained
    if not regression_results:
        st.warning(f"No successful regression models found for position {pos}.")
        continue
    
    # Store all regression results
    all_regression_results[pos] = regression_results
    
    # Convert the results to a DataFrame for easier manipulation
    try:
        results_df = pd.DataFrame(regression_results, columns=['Variable_Combination', 'Adjusted_R2'])
    except Exception as e:
        st.error(f"Error creating DataFrame for position {pos}: {e}")
        continue
    
    # Sort the DataFrame by Adjusted R² in descending order
    results_df_sorted = results_df.sort_values(by='Adjusted_R2', ascending=False)
    
    # Extract the top 10 models
    top_10 = results_df_sorted.head(10)
    
    # Store the top 10 regression results
    top_10_regression_results[pos] = top_10
    
    # Display the number of regression models found
    st.write(f"**Number of regression models found for {pos}:** {len(regression_results)}")
    
    # Display the top 10 regression models
    with st.expander(f"Top 10 Regression Models for {pos}"):
        st.table(top_10.style.format({
            'Variable_Combination': lambda x: ', '.join(x),
            'Adjusted_R2': "{:.3f}".format
    }))

# #### Summary of Regression Findings

st.markdown("""
We found that across all positions, `actual_pointsScored` is a consistent predictor, which makes sense because the points scored are based on the amount of points they score in real life. Though it's only one factor, it is clearly a very strong factor, though there may be signs of multicollinearity connected to this relationship. There was lower predictability for D/ST due to a significant number of external factors such as opposing offenses and a lot more moving parts that determine their scoring, which was not present, not to mention a smaller sample size. Now, let's plot the primary predictor for each position.""")

# #### Section 5: Identifying the Best Regression Models for Each Position

st.header("Section 5: Identifying the Best Regression Models for Each Position")

st.markdown("""
In this section, we identify the best regression model for each player position by selecting the model with the highest Adjusted R² from the top 10 models previously identified. This helps in understanding which combination of player statistics best predicts the total points for each position.
""")

# Initialize a dictionary to store the best regression results for each position
best_regression_results_specific = {}

# Initialize a list to collect results for display
best_models_display = []

# Iterate through each position and its top 10 regression results
for pos, top_10 in top_10_regression_results.items():
    if top_10.empty:
        st.warning(f"No successful regression models found for position {pos}. Skipping.")
        continue
    
    # Extract the top model (first row) for the current position
    best_combo = top_10.iloc[0]['Variable_Combination']
    best_adj_r2 = top_10.iloc[0]['Adjusted_R2']
    
    # Prepare the data for the best combination
    subset = position_dfs[pos][list(best_combo) + ['total_points']].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Check if subset has enough data points
    if subset.shape[0] < len(best_combo) + 1:
        st.warning(f"Not enough data to fit the best model for position {pos}. Skipping.")
        continue
    
    # Define independent and dependent variables
    X = subset[list(best_combo)]
    y = subset['total_points']
    X = sm.add_constant(X)  # Adds the constant term
    
    # Fit the OLS regression model
    try:
        model = sm.OLS(y, X).fit()
        
        # Store the results in the dictionary
        best_regression_results_specific[pos] = {
            'Best Combination': best_combo,
            'Adjusted R-squared': model.rsquared_adj,
            'Model': model
        }
        
        # Collect data for display
        best_models_display.append({
            'Position': pos,
            'Best Combination': ', '.join(best_combo),
            'Adjusted R²': f"{model.rsquared_adj:.3f}"
        })
    
    except Exception:
        # Silently skip the regression if it fails to fit
        continue

# Convert the collected best models data into a DataFrame for display

best_models_df = pd.DataFrame(best_models_display)

st.subheader("Best Regression Models by Position")
st.table(best_models_df)
# #### Section 6: Plotting the Best Regression Models

st.header("Section 6: Plotting the Best Regression Models")

st.markdown("""
In this section, we visualize the best regression model for each player position. The plots display the relationship between the primary predictor and the total points, along with the regression line and key statistics.
""")

# Convert the collected best models data into a DataFrame for display
best_models_display = []
for pos, details in best_regression_results_specific.items():
    best_models_display.append({
        'Position': pos,
        'Best Combination': ', '.join(details['Best Combination']),
        'Adjusted R²': f"{details['Adjusted R-squared']:.3f}"
    })

best_models_df = pd.DataFrame(best_models_display)

st.subheader("Best Regression Models by Position")
st.table(best_models_df)

# Plotting the Best Models
st.subheader("Visualization of Best Regression Models")

for pos, result in best_regression_results_specific.items():
    best_combo = result['Best Combination']
    best_model = result['Model']
    
    # Select the primary predictor (first variable in the combination)
    primary_predictor = best_combo[0]
    
    # Prepare the data for plotting
    subset = position_dfs[pos][list(best_combo) + ['total_points']].replace([np.inf, -np.inf], np.nan).dropna()
    
    if subset.empty:
        st.warning(f"No data available to plot for position {pos}.")
        continue
    
    # Define independent and dependent variables
    X_plot = subset[primary_predictor]
    y_plot = subset['total_points']
    
    # Create the regression plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.regplot(
        x=primary_predictor,
        y='total_points',
        data=subset,
        scatter=True,
        fit_reg=True,
        scatter_kws={'s': 100, 'alpha': 0.5},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax
    )
    ax.set_title(f'Linear Regression: {primary_predictor.replace("_", " ").title()} vs. Total Points for {pos}', fontsize=16)
    ax.set_xlabel(primary_predictor.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Total Points', fontsize=14)
    
    # Extract regression parameters
    slope = best_model.params.get(primary_predictor, None)
    intercept = best_model.params.get('const', None)
    r_squared = best_model.rsquared_adj
    p_value = best_model.pvalues.get(primary_predictor, None)
    
    # Add annotations if parameters are available
    if slope is not None and intercept is not None and p_value is not None:
        annotation_text = (
            f"Slope: {slope:.2f}\n"
            f"Intercept: {intercept:.2f}\n"
            f"Adjusted R-squared: {r_squared:.4f}\n"
            f"P-value: {p_value:.3e}"
        )
        ax.annotate(
            annotation_text,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            backgroundcolor='white',
            verticalalignment='top'
        )
    st.pyplot(fig)

st.markdown("""Overall, points scored seems to be the biggest predictor. The explanation in why the QB plot seems much more fitted is because of the weighting that QBs points are calculated. Touchdowns are 4 points while 100 yards passing is also 4 points. On the contrary, for RBs, WRs, and TEs, touchdowns are 6 points while 10 yards reception or 10 yards rushing is 1 point. Receivers also receive half a point for every reception. This scoring discrepancy is observed clearly in the plots. Additionally, kickers usually get 3 points for every field goal and is scaled up for longer field goals. Defenses on the other hand, obtain points through a myriad of ways including holding opponents to certain scores and yardage and only get 2 points for every interceptions. However, an interception in football usually signifies a turnover in possession and can be an important indicator for relative strength. However, its clear that the variables involved are too broad. Lets look at the best regression models when considering position specific variables. """)

# #### Section 7: Position-Specific Regressions

st.header("Section 7: Position-Specific Variable Regressions")

st.markdown("""
In this section, we identify the best combination of player-specific statistics that can predict the total points for each respective position with the highest statistical significance. Using the `find_all_regressions` function, we systematically evaluate all possible combinations of up to 3 independent variables and select the models with the highest Adjusted R². This approach helps in determining the player statistics that directly contribute the most to total points.
""")

# ##### Initializing Dictionaries to Store Regression Results

# Initialize dictionaries to store regression results
all_regression_results_specific = {}
top_10_regression_results_specific = {}

# Initialize a list to collect top models for display
top_models_display_specific = []

# ##### Performing Regression Analysis for Each Position

st.subheader("Regression Analysis Results by Position")

# Iterate through each position and perform regression analysis
for pos, df in position_dfs.items():
    st.markdown(f"### Regression Analysis for **{pos}**")
    
    # Use position-specific variables only
    vars_list_specific = position_specific_vars.get(pos, [])
    
    # Check if there are enough variables to perform regression
    if not vars_list_specific:
        st.info(f"No position-specific variables available for **{pos}**. Skipping regression.")
        continue
    
    # Perform the regression analysis to get all combinations and their Adjusted R²
    regression_results = find_all_regressions(
        df,
        independent_vars=vars_list_specific,
        dependent_var='total_points',
        max_vars=3  # Use up to 3 variables
    )
    
    # Display the number of regression models found
    st.write(f"**Number of regression models found:** {len(regression_results)}")
    
    if regression_results:
        # Optionally display the first few regression results for debugging
        # st.write(f"**First few regression results for {pos}:**")
        # st.write(regression_results[:3])
        pass
    
    if not regression_results:
        st.warning(f"No successful regression models found for **{pos}**.")
        continue
    
    # Store all regression results
    all_regression_results_specific[pos] = regression_results
    
    # Convert the results to a DataFrame for easier manipulation
    try:
        results_df = pd.DataFrame(regression_results, columns=['Variable_Combination', 'Adjusted_R2'])
    except Exception as e:
        st.error(f"Error creating DataFrame for **{pos}**: {e}")
        continue
    
    # Sort the DataFrame by Adjusted R² in descending order
    results_df_sorted = results_df.sort_values(by='Adjusted_R2', ascending=False)
    
    # Extract the top 10 models
    top_10 = results_df_sorted.head(10)
    
    # Store the top 10 regression results
    top_10_regression_results_specific[pos] = top_10
    
    # Append to the display list
    top_models_display_specific.append({
        'Position': pos,
        'Best Combination': ', '.join(top_10.iloc[0]['Variable_Combination']),
        'Adjusted R²': f"{top_10.iloc[0]['Adjusted_R2']:.3f}"
    })
    
    # Display the top 10 Adjusted R² for the current position
    with st.expander(f"Top 10 Regression Models for {pos}"):
        st.table(top_10.style.format({
            'Variable_Combination': lambda x: ', '.join(x),
            'Adjusted_R2': "{:.3f}".format
        }))

    # ##### Displaying the Best Regression Models by Position

if top_models_display_specific:
    best_models_specific_df = pd.DataFrame(top_models_display_specific)
    st.markdown("### Best Regression Models by Position")
    st.table(best_models_specific_df)
else:
    st.info("No best regression models were identified for any position.")

# ##### Plotting the Best Regression Models

st.subheader("Visualization of Best Regression Models")

for pos, top_10 in top_10_regression_results_specific.items():
    # Extract the top variable combination
    best_combo = top_10.iloc[0]['Variable_Combination']  # First entry is the best
    
    # Prepare the data for the plot
    subset = position_dfs[pos][list(best_combo) + ['total_points']].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Pick the primary predictor for the regression line (first variable in the combination)
    primary_predictor = best_combo[0]
    
    # Check if there is sufficient data
    if subset.empty:
        st.warning(f"No data available to plot for **{pos}**.")
        continue
    
    # Define independent and dependent variables
    X_plot = subset[primary_predictor]
    y_plot = subset['total_points']
    
    # Fit the model again for plotting purposes (optional if already stored)
    X_plot_const = sm.add_constant(X_plot)
    try:
        model = sm.OLS(y_plot, X_plot_const).fit()
    except Exception:
        st.warning(f"Failed to fit the best model for **{pos}** during plotting.")
        continue
    
    # Create the regression plot within an expander for better organization
    with st.expander(f"View Regression Plot for **{pos}**"):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.regplot(
            x=primary_predictor,
            y='total_points',
            data=subset,
            scatter=True,
            fit_reg=True,
            scatter_kws={'s': 100, 'alpha': 0.5},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax
        )
        ax.set_title(f'Linear Regression: {primary_predictor.replace("_", " ").title()} vs. Total Points for {pos}', fontsize=16)
        ax.set_xlabel(primary_predictor.replace('_', ' ').title(), fontsize=14)
        ax.set_ylabel('Total Points', fontsize=14)
        
        # Extract regression parameters
        slope = model.params.get(primary_predictor, None)
        intercept = model.params.get('const', None)
        r_squared = model.rsquared_adj
        p_value = model.pvalues.get(primary_predictor, None)
        
        # Add annotations if parameters are available
        if slope is not None and intercept is not None and p_value is not None:
            annotation_text = (
                f"Slope: {slope:.2f}\n"
                f"Intercept: {intercept:.2f}\n"
                f"Adjusted R-squared: {r_squared:.4f}\n"
                f"P-value: {p_value:.3e}"
            )
            ax.annotate(
                annotation_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                backgroundcolor='white',
                verticalalignment='top'
            )
        
        # Display the plot in Streamlit
        st.pyplot(fig)

st.markdown("""These regression models yield much more interested results. The best predictor of QB points is actually stat line that yields no points: attempted passes. Tight Ends have the best adjusted R^2 from receiving yards which makes sense since tight ends have variability in their usage and often aren't targeted for the end zone. The prudent fantasy manager would look for tight ends that may be undervalued on a receiving yards stand point. Running backs often are the highest scoring players in the game and my model has found that rushing touchdowns is the best predictor in contrast to rushing yards. Drafting undervalued running backs that are used in red zone plays could be a prduent move. Wide receivers were found to have highly predictable performance based on receiving yards. The full regression model with an adjusted 0.9053 actually had three variables: receiving yards, touchdowns and yards after catch highlighting the importance of playmaking ability after the reception. Finally, defenses had interceptions again while kickers had attempted field goals which is very interesting considering kickers only get points from made field goals. Because of the relative accuracy of NFL kickers, it would serve the fantasy manager to find kickers who play on teams with high volumes of non-conversion in their opponents territory. 
""")
    
# #### Section 8: Random Forest Draft Prediction Model

st.header("Fantasy Football Draft Prediction")
# ##### Function Definitions

def prepare_position_data(players_df, position_vars):
    """
    Prepare position-specific data by selecting relevant variables and handling missing values.
    """
    position_dfs = {}
    for position, vars_list in position_vars.items():
        vars_list_clean = [var for var in vars_list if var != 'total_points']
        # Ensure 'position' column exists
        if 'position' not in players_df.columns:
            st.error("The DataFrame does not contain a 'position' column.")
            continue
        position_df = players_df[players_df['position'] == position].dropna(subset=['total_points'])
        # Check if all variables exist in the DataFrame
        missing_vars = [var for var in vars_list_clean + ['total_points'] if var not in position_df.columns]
        if missing_vars:
            st.warning(f"Missing variables for position **{position}**: {missing_vars}. These will be skipped.")
            vars_list_clean = [var for var in vars_list_clean if var in position_df.columns]
        position_df = position_df[vars_list_clean + ['total_points']]
        position_dfs[position] = position_df
    return position_dfs

def train_position_models(position_dfs):
    """
    Train Random Forest models for each position and evaluate them using MAE and R².
    """
    models = {}
    metrics = {}  # Dictionary to store MAE and R² for each position
    for position, df in position_dfs.items():
        st.markdown(f"#### Training Random Forest for **{position}**")
        
        X = df.drop(columns=['total_points'])
        y = df['total_points']
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store the model and metrics
        models[position] = rf
        metrics[position] = {'MAE': mae, 'R²': r2}
        
        # Display metrics within an expander
        with st.expander(f"View Metrics for {position}"):
            st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
            st.write(f"**R-squared (R²):** {r2:.2f}")
        
        # Plot Feature Importances within an expander
        with st.expander(f"Feature Importances for {position}"):
            feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
            feature_importances = feature_importances.sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax)
            ax.set_title(f'Feature Importances for {position}', fontsize=16)
            ax.set_xlabel('Importance Score', fontsize=14)
            ax.set_ylabel('Features', fontsize=14)
            st.pyplot(fig)
        
        st.markdown("---")  # Separator between positions
    
    return models, metrics

def predict_points_2024(players_df, models, position_vars):
    """
    Use trained models to predict points for the 2024 season.
    """
    predictions = []
    for position, model in models.items():
        features = [var for var in position_vars[position] if var in players_df.columns]
        pos_data = players_df[players_df['position'] == position][features].fillna(0)
        preds = model.predict(pos_data)
        temp_df = players_df[players_df['position'] == position].copy()
        temp_df['projected_points'] = preds
        predictions.append(temp_df)
    return pd.concat(predictions)

def simulate_snake_draft_intelligent(player_pool, my_team_position=1, num_teams=12, rounds=15):
    """
    Simulate a snake draft where all teams operate intelligently.
    Each team evaluates players based on projected points and positional needs.
    """
    draft_order = list(range(1, num_teams + 1))
    reverse_order = draft_order[::-1]
    teams = {team: [] for team in draft_order}
    position_constraints = {
        'QB': 2, 'RB': 4, 'WR': 4, 'TE': 2, 'K': 1, 'D/ST': 1
    }
    team_positions = {team: position_constraints.copy() for team in draft_order}
    # Assuming 'season' and 'projected_points' columns exist
    if 'season' not in player_pool.columns or 'projected_points' not in player_pool.columns:
        st.error("The player pool DataFrame must contain 'season' and 'projected_points' columns.")
        return teams[my_team_position], teams
    
    player_pool = player_pool[player_pool['season'] == 2024].sort_values(by='projected_points', ascending=False).copy()
    player_pool['drafted'] = False
    
    def get_best_player(pos, available_players):
        """Get the best available player for a specific position."""
        pos_players = available_players[available_players['position'] == pos]
        if not pos_players.empty:
            return pos_players.iloc[0]
        return None
    
    for round_num in range(1, rounds + 1):
        # Determine draft order for this round (snake draft logic)
        current_order = draft_order if round_num % 2 != 0 else reverse_order
        for team_id in current_order:
            # Filter available players
            available_players = player_pool[player_pool['drafted'] == False]
    
            # Evaluate best positional fit for the team
            best_pick = None
            for position, need in team_positions[team_id].items():
                if need > 0:  # If the team still needs players for this position
                    best_pick = get_best_player(position, available_players)
                    if best_pick is not None:
                        break
    
            # If no positional need is found, pick the highest projected player
            if best_pick is None:
                best_pick = available_players.iloc[0]
    
            # Add the selected player to the team
            teams[team_id].append(best_pick)
            player_pool.at[best_pick.name, 'drafted'] = True
            team_positions[team_id][best_pick['position']] -= 1
    
            # Display draft pick for the user's team within an expander
            if team_id == my_team_position:
                with st.expander("Your Draft Picks"):
                    st.write(f"**Round {round_num}, Team {team_id}:** {best_pick['player_name']} ({best_pick['position']}) - {best_pick['projected_points']} pts")
            else:
                # Optionally, display other teams' picks or skip for brevity
                pass
    
    # Return the user's team and all teams
    return teams[my_team_position], teams
def calculate_positional_adjusted_value(players_df, replacement_levels):
    """
    Calculate Positional Adjusted Value (PAV) for each player based on replacement level.
    """
    players_df['PAV'] = 0  # Initialize PAV column
    
    for position, replacement_level in replacement_levels.items():
        pos_df = players_df[players_df['position'] == position]
        if pos_df.empty:
            st.warning(f"No players found for position **{position}** to calculate PAV.")
            continue
        # Handle cases where there are fewer players than the replacement level
        if len(pos_df) < replacement_level:
            avg_replacement_points = pos_df['projected_points'].min()
        else:
            avg_replacement_points = pos_df.nlargest(replacement_level, 'projected_points')['projected_points'].min()
        players_df.loc[players_df['position'] == position, 'PAV'] = (
            players_df.loc[players_df['position'] == position, 'projected_points'] - avg_replacement_points
        )
    
    return players_df
def simulate_snake_draft_with_pav(pav_player_pool, my_team_position=1, num_teams=12, rounds=15):
    """
    Simulate a snake draft using Positional Adjusted Value (PAV).
    Each team evaluates players based on PAV and positional needs.
    """
    draft_order = list(range(1, num_teams + 1))
    reverse_order = draft_order[::-1]
    teams = {team: [] for team in draft_order}
    position_constraints = {
        'QB': 2, 'RB': 6, 'WR': 6, 'TE': 2, 'K': 1, 'D/ST': 1
    }
    team_positions = {team: position_constraints.copy() for team in draft_order}
    
    # Ensure required columns exist
    if 'season' not in pav_player_pool.columns or 'PAV' not in pav_player_pool.columns:
        st.error("The player pool DataFrame must contain 'season' and 'PAV' columns.")
        return teams[my_team_position], teams
    
    # Filter for the 2024 season and sort by PAV descending
    pav_player_pool = pav_player_pool[pav_player_pool['season'] == 2024].sort_values(by='PAV', ascending=False).copy()
    pav_player_pool['drafted'] = False
    
    def get_best_player(pos, available_players_1):
        """Get the best available player for a specific position."""
        pos_players = available_players_1[available_players_1['position'] == pos]
        if not pos_players.empty:
            return pos_players.iloc[0]
        return None
    
    for round_num in range(1, rounds + 1):
        # Determine draft order for this round (snake draft logic)
        current_order = draft_order if round_num % 2 != 0 else reverse_order
        for team_id in current_order:
            # Filter available players
            available_players_1 = pav_player_pool[pav_player_pool['drafted'] == False]
    
            # Evaluate best positional fit for the team
            best_pick = None
            for position, need in team_positions[team_id].items():
                if need > 0:  # If the team still needs players for this position
                    best_pick = get_best_player(position, available_players_1)
                    if best_pick is not None:
                        break
    
            # If no positional need is found, pick the highest PAV player
            if best_pick is None:
                best_pick = available_players_1.iloc[0] if not available_players_1.empty else None
    
            if best_pick is not None:
                # Add the selected player to the team
                teams[team_id].append(best_pick)
                pav_player_pool.at[best_pick.name, 'drafted'] = True
                team_positions[team_id][best_pick['position']] -= 1
    
                # Display draft pick for the user's team within an expander
                if team_id == my_team_position:
                    with st.expander("Your Draft Picks with PAV"):
                        st.write(f"**Round {round_num}, Team {team_id}:** {best_pick['player_name']} ({best_pick['position']}) - {best_pick['projected_points']} pts, PAV: {best_pick['PAV']:.2f} pts")
            else:
                st.warning(f"Team {team_id} could not draft a player in round {round_num} due to constraints.")
                continue
    
    # Return the user's team and all teams
    return teams[my_team_position], teams
# ##### Main Execution

def main():
    st.title("Fantasy Football Draft Prediction Model")
    
    st.markdown("""
    #### Random Forest Draft Prediction Model
    
    In this section, we expand on the relationships found in the initial regression analysis and build a more sophisticated model to aid in drafting a successful team. Instead of relying solely on ESPN's projected points, we utilize specific historical data, including draft data from your league, to tailor a drafting model specifically suited to the league's dynamics.
    """)

    # ##### Define Position-Specific Variables
    st.subheader("Define Position-Specific Variables")
    
    # Define position-specific projected statistics
    position_specific_vars_projected = {
    'QB': [
        'projected_passingAttempts',
        'projected_passingCompletions',
        'projected_passingIncompletions',
        'projected_passingYards',
        'projected_passingTouchdowns',
        'projected_passingInterceptions',
        'projected_passingCompletionPercentage',
        'projected_turnovers',
    ],
    'RB': [
        'projected_rushingAttempts',
        'projected_rushingYards',
        'projected_rushingTouchdowns',
        'projected_rushingYardsPerAttempt',
        'projected_receivingReceptions',
        'projected_receivingYards',
        'projected_receivingTouchdowns',
        'projected_fumbles',
    ],
    'WR': [
        'projected_receivingReceptions',
        'projected_receivingYards',
        'projected_receivingTouchdowns',
        'projected_receivingTargets',
        'projected_receivingYardsPerReception',
        'projected_fumbles',
    ],
    'TE': [
        'projected_receivingReceptions',
        'projected_receivingYards',
        'projected_receivingTouchdowns',
        'projected_fumbles',
    ],
    'K': [
        'projected_madeFieldGoals',
        'projected_attemptedFieldGoals',
        'projected_missedFieldGoals',
        'projected_madeExtraPoints',
        'projected_attemptedExtraPoints',
        'projected_missedExtraPoints',
    ],
    'D/ST': [
        'projected_defensiveTouchdowns',
        'projected_defensiveInterceptions',
        'projected_defensiveSacks',
        'projected_defensiveForcedFumbles',
        'projected_defensiveTotalTackles',
        'projected_defensivePointsAllowed',
        'projected_defensiveYardsAllowed',
    ]
}
    # Add common variables
    common_vars = []
    position_vars = {pos: common_vars + pos_vars for pos, pos_vars in position_specific_vars_projected.items()}
    
    # ##### Prepare Position Data
    st.subheader("Prepare Position Data")
    
    with st.expander("View Prepared Position Data"):
        position_dfs = prepare_position_data(players_df, position_vars)
        for pos, df in position_dfs.items():
            st.markdown(f"**{pos}**:")
            st.write(f"Number of players: {df.shape[0]}")
            st.dataframe(df.head())
    
    # ##### Train Random Forest Models
    st.subheader("Train Random Forest Models")
    
    models, metrics = train_position_models(position_dfs)
    
    # ##### Display Model Performance Metrics
    st.subheader("Model Performance Metrics")
    
    with st.expander("View All Model Metrics"):
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.rename_axis("Position").reset_index()
        st.table(metrics_df.style.format({
            'MAE': "{:.2f}",
            'R²': "{:.2f}"
        }))
    
    # ##### Predict Points for 2024
    st.subheader("Predict Points for 2024 Season")
    
    predicted_df = predict_points_2024(players_df, models, position_vars)
    
    with st.expander("View Predicted Points"):
        st.write("**Sample Predicted Points:**")
        st.dataframe(predicted_df[['player_name', 'position', 'projected_points']].head())
    
    # ##### Simulate the Intelligent Draft
    st.subheader("Simulate the Intelligent Snake Draft")
    
    with st.expander("Configure Draft Settings"):
        my_team_position = st.number_input("Select Your Team Position (1-12)", min_value=1, max_value=12, value=1, step=1)
        num_teams = st.number_input("Number of Teams in the League", min_value=2, max_value=20, value=12, step=1)
        rounds = st.number_input("Number of Rounds in the Draft", min_value=1, max_value=30, value=15, step=1)
    
        # ##### Initialize session state variables
    if 'initial_draft_done' not in st.session_state:
        st.session_state.initial_draft_done = False

    if 'pav_draft_done' not in st.session_state:
        st.session_state.pav_draft_done = False

    # Update session state after initial draft simulation
    if st.button("Run Draft Simulation") and not st.session_state.initial_draft_done:
        with st.spinner("Simulating the draft..."):
            my_team, all_teams = simulate_snake_draft_intelligent(
                predicted_df, 
                my_team_position=int(my_team_position), 
                num_teams=int(num_teams), 
                rounds=int(rounds)
            )
        st.success("Draft simulation completed!")
    
        # ##### Display Your Team
        st.markdown("### Your Final Drafted Team")
        my_team_df = pd.DataFrame(my_team)
        my_team_df_display = my_team_df[['player_name', 'position', 'projected_points']]
        st.table(my_team_df_display)
        total_points = my_team_df['projected_points'].sum()
        st.write(f"**Total Projected Points:** {total_points:.2f}")
    
        # ##### Display All Teams
        with st.expander("View All Teams"):
            for team_id, team_players in all_teams.items():
                st.markdown(f"#### Team {team_id}")
                team_df = pd.DataFrame(team_players)
                team_df_display = team_df[['player_name', 'position', 'projected_points']]
                st.table(team_df_display)
                team_total = team_df['projected_points'].sum()
                st.write(f"**Total Projected Points:** {team_total:.2f}")
                st.markdown("---")
    
        # Update session state
        st.session_state.initial_draft_done = True
         # ##### Add the final markdown
        st.markdown("""This draft simulator for 2024 trained on data for the last 3 years finds that in the 1st round, players all pick QBs which is not at all the norm. This comes from how the random forest models are being built and ran. Each position has its own random forest model that is built from specific dataframes that have their position specific variables and total fantasy points. Then the data is split into 80% training, 20% test subsets and we are running 100 trees in the Random Forest Regressor. The model is finding which player statistics contribute the most to total points. The trained Random Forest model is then used to predict projected fantasy points for the 2024 season for every 2024 player. 
    
                Then, I run a snake draft where each round is reverse order for the 12 teams involved. There are specific position requirements encoded. For each draft pick, the code will remove players already drafted by a team and only consider players whose positions the team still needs. Then they will rank the remaining player based on total projected points and draft the player with the highest value. As a result, we can clearly see that this will lead to QBs having the highest draft order since they have the highest value. With that being said, we need to find another metric that better incorporates the opportunity cost of drafting a player compared to what other teams may do. 

                As a result, this next model will sort players by an adjusted-position value which takes into account the positional scarcity by comparing each players projected points to the average replacement-level player at that same position. Replacement-level players are players that typically would not start and we would subtract those points from each player's projected points. Here is a new draft with that:""")
        
    # ##### Positional Adjusted Value (PAV) and PAV-based Draft Simulation
    if st.session_state.initial_draft_done and not st.session_state.pav_draft_done:
        if st.button("Calculate PAV and Run PAV-based Draft Simulation"):
            with st.spinner("Calculating PAV and simulating the draft based on PAV..."):
                replacement_levels = {
                    'QB': 12, 
                    'RB': 30, 
                    'WR': 30, 
                    'TE': 12, 
                    'K': 12, 
                    'D/ST': 12
                }
            
                # Calculate PAV
                predicted_df_pav = calculate_positional_adjusted_value(predicted_df.copy(), replacement_levels)
            
                # Repopulate player_pool for PAV-based draft simulation
                pav_player_pool = predicted_df_pav.copy()
                pav_player_pool['drafted'] = False
            
                # Run PAV-based draft simulation
                my_team_pav, all_teams_pav = simulate_snake_draft_with_pav(
                    pav_player_pool, 
                    my_team_position=int(my_team_position), 
                    num_teams=int(num_teams), 
                    rounds=int(rounds)
                )
        
            st.success("PAV-based draft simulation completed!")
        
            # ##### Display Your Team based on PAV
            st.markdown("### Your Final Drafted Team (Based on PAV)")
            if my_team_pav:
                my_team_pav_df = pd.DataFrame(my_team_pav)
                my_team_pav_df_display = my_team_pav_df[['player_name', 'position', 'projected_points', 'PAV']]
                st.table(my_team_pav_df_display)
                total_pav_points = my_team_pav_df['projected_points'].sum()
                st.write(f"**Total Projected Points with PAV:** {total_pav_points:.2f}")
            else:
                st.info("No players were drafted to your team based on PAV.")
        
            # ##### Display All Teams based on PAV
            with st.expander("View All Teams Based on PAV"):
                for team_id, team_players in all_teams_pav.items():
                    st.markdown(f"#### Team {team_id}")
                    if team_players:
                        team_df = pd.DataFrame(team_players)
                        team_df_display = team_df[['player_name', 'position', 'projected_points', 'PAV']]
                        st.table(team_df_display)
                        team_total = team_df['projected_points'].sum()
                        st.write(f"**Total Projected Points:** {team_total:.2f}")
                    else:
                        st.write("No players drafted to this team.")
                    st.markdown("---")
            st.markdown("""Interestingly, for both models, the first 5 picks did not change at all while the last 10 picks were markedly different. Regardless, both draft simulations were roughly believable! Perhaps, if the QB was given some sort of handicap or if this was a machine learning model in which the computer was fed many simulations of real drafts, it would learn not to draft a QB first. Or perhaps, drafting a QB first, especially Lamar Jackson is actually a good play and my league and I have been doing it wrong all of these years.""")

       
            # Update session state
            st.session_state.pav_draft_done = True
if __name__ == "__main__":
    main()

# ##### Display Final Summary and Conclusion After PAV-based Draft Simulation
st.markdown("""
## Summary of Findings

In my analysis of my ESPN Fantasy Football league, I evaluated various predictive models and positional metrics to optimize player selection and draft strategies. The results demonstrated strong predictive performance for positions like running backs and wide receivers, while there was more variability in positions like defenses/special teams and quarterbacks. The models used were Linear Univariate and Multivariate Regression as well as Random Forest Regression.

### Key Findings:
1. **Strong Predictive Accuracy for RBs and WRs:**
   - Achieved high adjusted R² values.
   - Highlighted statistics that may not seem intuitive in determining total points, such as touchdowns vs. receiving yards for running backs and wide receivers respectively.
   
2. **Volume Metrics as Top Predictors:**
   - Consistently emerged as key predictors across all positions.
   
3. **Optimal Draft Strategies:**
   - **Early-Round Picks:** Running backs (RBs) and wide receivers (WRs) offer the greatest return on early-round draft picks due to their positional scarcity and high variability among replacement-level players.
   
4. **Quarterback (QB) Drafting Flexibility:**
   - Later rounds can be utilized to draft QBs without substantial drop-offs in projected performance.

## Next Steps
1. **Expand the Dataset:**
   - Include players that were not on rosters at the end of the season.
   - Incorporate data from other public leagues to compile as much information as possible.
   
2. **Enhance Model Features:**
   - Add game schedule strength as a contextual feature in the Random Forest Model.
   - Incorporate injury health data as a factor in player drafting decisions.
   
3. **Analyze Mid-Season Transactions:**
   - Evaluate trades and waiver wire adds/drops to assess their effectiveness.
   - While the number of trades alone showed no bearing on final standings, the quality of trades needs to be measured.

## Conclusion

This analysis proved to be particularly insightful. As a long-time fantasy football fan, this study helped answer questions that I had been wondering for years. Chiefly, **is my league good at drafting** and **is ESPN any good at predicting who drafted well?** The answer to both of these questions is **no**, which means next year, I hope that the model I started for this project can become quite comprehensive. Perhaps it can draft for me once I incorporate these additional factors, leading to my first fantasy championship victory.
""")

