The collection of documents above is a predictor of what indicators boost the likelihood of success (games won) in the NBA using 
ML methodology.







# **Table of Contents**
I. Introduction <br>
II. Methodology <br>
&nbsp A. Finding A Data Source <br>
  B. BoxScore: Sourcing Game Box Scores <br>
  C. Sourcing Team Win-Loss Results <br>
  D. Explanation of tools and libraries utilized in the Jupyter Notebook <br>
  E. Detailed walkthrough of the analysis workflow <br>
  F. Explanation of statistical methods, machine learning algorithms, or other techniques applied<br>
<
III. Results and Analysis	10<br>
  A. Presentation of findings from the Jupyter Notebook<br>
  B. Interpretation of results and their implications<br>
  C. Comparison with existing literature or benchmarks, if applicable	<br>
  D. Comparing the various methods<br>
IV. Challenges Encountered	
  A. Identification of technical challenges faced during the project	
  B. Reflection on lessons learned and improvements for future projects








# **Introduction**
Every year, people from all backgrounds from sports fanatics to data scientists come together to make predictions on the NBA finals and place bets on the outcomes of basketball games. In this project, we developed quantitative methods of measuring each team's historical performances and assessed the most relevant factors in determining the outcome of an NBA game.

# **Methodology**
## A. Finding A Data Source

Data on NBA teams, games, and players come in a myriad of modified formats and factors. Since the NBA is quite a popular league, many independent actors take it upon themselves to convert the raw performance of each athlete into unique metrics, visuals, and databases, creating heatmaps of shot selection or even their own statistical metrics of performance. Maintaining the integrity of this project required us to filter through the noise and assess which sources are of greatest significance. Through research on the standard amongst sports enthusiasts, we found open access to the most reliable source for recorded data (i.e. uncalculated metrics like games won, rebounds, assists, etc.): An API of game, player, and team data owned and made publicly available by the NBA. Within this API are vast repositories of information dating back to the 1996-1997 season. Here we took a bifurcated approach on determining relevant factors. On one hand we had discovered a method of determining future success based on historical wins, but moving in this direction dejected the most intuitive approach of assessing a combination of box-score metrics. So, we split our research into two approaches. Moving forward, this paper will also be split in two to properly assess the methodology of our research as a whole. 

## B. BoxScore: Sourcing Game Box Scores
When constructing our Box Score dataset, we pulled information from two endpoints: leaguegamefinder.md and teamestimatedmetrics.md. 

**leaguegamefinder.md:** Composed of 6 datasets, this endpoint gave a box score of individual game data. Using the first dataset, we created a dataframe of games from April 16th, 2024 (the most recent game played) dating back to the start of the 2019-2020 season (the earliest game in the dataset). This dataframe contained the following information:

SEASON_ID: Season identification number
TEAM_ID: Team identification number (all info is related to the team_id)
TEAM_ABBREVIATION: Team three-letter abbreviation
TEAM_NAME: Team Name
GAME_ID: Game identification number
MATCHUP: The teams competing against each other
WL: Win or Loss
MIN: Total game duration
PTS: Total points made
FGM: Total field goals made
FGA: Total field goal attempts
FG_PCT: Team field goal percentage made
FG3M: Total three-pointers made
FG3A: Total three-pointer attempts
FG3_PCT: Team three-pointer percentage made
FTM: Total free throws made
FTA: Total free throw attempts
FT_PCT: Team free throw percentage
OREB: Total offensive rebounds
DREB: Total defensive rebounds
REB: Total rebounds
AST: Total assists
STL: Total steals
BLK: Total blocks
TOV: Total turnovers
PF: Total personal fouls
PLUS_MINUS: Team plus/minus differential; how many points was the game or lost by



**teamestimatedmetrics.md:** Composed of one dataset, this endpoint contains information on team ratings and rankings. It is composed of the following columns:

TEAM_NAME: Team name
TEAM_ID: Team identification number (all info is related to the team_id)
GP: Total games played
W: Total wins
L: Total losses
W_PCT: Win percentage
MIN: Total games duration this season
E_OFF_RATING: Effective offensive rating
E_DEF_RATING:Effective defensive rating
E_NET_RATING: Effective net rating
E_PACE: Effective pace (calculation of possessions per game)
E_AST_RATIO: Effective assists ratio (assists per 100 possessions)
E_OREB_PCT: Effective offensive rebounds percentage
E_DREB_PCT: Effective defensive rebound percentage
E_REB_PCT: Effective rebound percentage
E_TM_TOV_PCT: Effective turnover percentage
GP_RANK: Games played rank
W_RANK: Wins rank
L_RANK: Losses rank
W_PCT_RANK: Win percentage rank
MIN_RANK: Total duration rank
E_OFF_RATING_RANK: Effective offensive rating rank
E_DEF_RATING_RANK: Effective defensive rating rank
E_NET_RATING_RANK: Effective team rating rank
E_AST_RATING_RANK: Effective assist rating rank
E_AST_RATIO_RANK: Effective assist ratio rank
E_OREB_PCT_RANK: Effective offensive rebound percentage rank
E_DREB_PCT_RANK: Effective defensive rebound percentage rank
E_REB_PCT_RANK: Effective rebound percentage rank
E_TM_TOV_PCT_RANK: Effective turnover percentage rank
E_PACE_RANK: Effective pace rank




## C. BoxScore: Data Wrangling
After sourcing these two datasets, we joined the team ratings/rankings onto the game data on the team name, effectively creating a dataframe (13795, 57) where each row of games included the team’s rating and ranking. To ensure that our data accurately accounted for all NBA teams, we checked the unique team abbreviation and found that it was the exact number of teams (30). Next, we inspected the type of data within our table and found nothing out of the ordinary. After clearing four rows that contained null values, we transformed the win-loss column from W/L to 1/0, changing the data type to integer as well. Finally, we took the box score data and aggregated the means of each team in the following categories: ['FG_PCT', 'FG3_PCT', 'FT_PCT','TOV','STL','BLK','REB','PF', 'PLUS_MINUS']. After joining these values to the table, we began building our model.


Sourcing Team Win-Loss Results
After some initial data wrangling from NBA API, we created two datasets: 'gameLogs' and 'nbaHomeWinLossModelDataset'. In these two datasets, we selected relevant columns that we thought were key to our modeling portion of the project. We created new columns that were calculated from the raw data, such as total win percentage and last game home win percentage from the original columns. The primary goal of data wrangling was to distill the raw game data throughout the seasons and aggregate the game data to find overall winning percentages of teams against each other and their performances throughout the seasons.

We eventually arrived at 'nbaHomeWinLossModelDataset' with the following fields and field descriptions.

Field Name
Annotation
HOME_LAST_GAME_OE
Offensive efficiency in the home team's last game
HOME_LAST_GAME_HOME_WIN_PCTG
Home team's winning percentage in their last home game
HOME_NUM_REST_DAYS
Number of rest days for the home team before the current game
HOME_LAST_GAME_AWAY_WIN_PCTG
Home team's winning percentage in their last away game
HOME_LAST_GAME_TOTAL_WIN_PCTG
Home team's total winning percentage across all games before the current game
HOME_LAST_GAME_ROLLING_SCORING_MARGIN
Home team's average scoring margin over last 3 games
HOME_LAST_GAME_ROLLING_OE
Home team's offensive efficiency averaged over last 3 games
HOME_W
Indicates if the home team won the last game (uses binary values)
SEASON
The season during which the game is played
AWAY_LAST_GAME_OE
Offensive efficiency in the away team's last game
AWAY_LAST_GAME_HOME_WIN_PCTG
Away team's winning percentage in their last game played at home
AWAY_NUM_REST_DAYS
Number of rest days for the away team before the current game
AWAY_LAST_GAME_AWAY_WIN_PCTG
Away team's winning percentage in their last game played away from home
AWAY_LAST_GAME_TOTAL_WIN_PCTG
Away team's total winning percentage across all games before the current game
AWAY_LAST_GAME_ROLLING_SCORING_MARGIN
Away team's average scoring margin over last 3 games
AWAY_LAST_GAME_ROLLING_OE
Away team's offensive efficiency averaged over last 3 games






## D. Explanation of tools and libraries utilized in the Jupyter Notebook
In our modeling training, we utilized several powerful libraries and classes essential for machine learning tasks in Python. We used Pandas, Matplotlib.pyplot, and Scikit-learn. Within Scikit-learn, we leveraged specific classes tailored for machine learning tasks. RandomForestClassifier has enabled effective classification and regression through its ensemble learning method based on decision trees. For regularization and feature selection in high-dimensional datasets, we utilized Lasso and Ridge linear regression techniques. LogisticRegression has proven valuable for binary classification tasks, offering simplicity and interpretability. Hyperparameter tuning has been optimized using GridSearchCV, enhancing model performance. Lastly, StandardScaler has been crucial for preprocessing data by standardizing features, ensuring uniform scales and improving the efficiency and convergence of many machine learning algorithms, particularly in the context of big data analysis.

## E. Detailed walkthrough of the analysis workflow
Our workflow process was split across several notebooks, but it follows the sequence of data gathering, data wrangling, exploratory data analysis, model building and final conclusion. In the model building phase, we split the data into test and training with the game result as our y variable and the rest as the x variables. We split the test and training data into a 1:2 ratio split. 

## F. Explanation of statistical methods, machine learning algorithms, or other techniques applied
In the 'game_wins_modeling' notebook, we employed various ML algorithms and regression methods: Logistic Regression, Lasso Regression, Ridge Regression, Decision Trees, and Random Forest. Among these methods, Logistic Regression was the most accurate and precise predictor and proved to be quite useful due to its effectiveness at predicting a binary variable (Win or Loss). We measured 'accuracy' as the ability to predict a game's final score in the testing data. This was calculated as a proportion of correctly predicted games to the total amount of games.


# III. Results and Analysis

## A. BoxScore: Model Training
Using a train size of 75%, we build training and testing dataframes. Our list of constant variables toggled quite a bit as we assessed relevant factors in improving the model’s accuracy. After utilizing linear probability to determine the effect of each variable, we arrived at a final list of relevant variables: ['E_OFF_RATING', 'PLUS_MINUS_y', 'E_REB_PCT', 'FTA', 'FG3_PCT_x'] (_y indicates a team data average, _x indicates an individual game statistic). 

## B. BoxScore: Visualizations and EDA
For the most part, the visualizations of these variables are unremarkable. Since our factors are already aggregated to the teams, what we see in our visualizations are just indicators of frequency which doesn’t allow us to better understand which type of teams might find success. Since the win/loss data is binary, we would also be unable to glean anything from a scatter plot relating the variables to the wins/losses.





## C. Presentation of findings from the Jupyter Notebook
From our models, we have the following summary results.

game_wins_modeling Data
Logistic Regression:
Accuracy: 0.6158038147138964
Precision: 0.6569920844327177
Recall: 0.6209476309226932
F1: 0.6384615384615385

Lasso Regression:
Accuracy:  0.547683923705722
Precision:  0.5771812080536913
Recall:  0.6433915211970075
F1:  0.608490566037736
Mean Absolute Error (MAE): 0.45231607629427795
Mean Squared Error (MSE): 0.45231607629427795
Root Mean Squared Error (RMSE): 0.6725444790452733
R-squared (R2): -0.8249271715605879

Ridge Regression:
Accuracy:  0.553133514986376
Precision:  0.5879518072289157
Recall:  0.6084788029925187
F1:  0.5980392156862745
Mean Absolute Error (MAE): 0.44686648501362397
Mean Squared Error (MSE): 0.44686648501362397
Root Mean Squared Error (RMSE): 0.668480728977002
R-squared (R2): -0.8029400972044363

Decision Tree:
Training accuracy :  0.6339825386165212
Testing accuracy :  0.6076294277929155

Random Forest:
Accuracy: 0.5885558583106267

BoxScore Model Training Data
Logistic Regression:
Accuracy:  0.541207196749855
Precision:  0.5418006430868167
Recall:  0.5820379965457686
F1:  0.5611990008326395
Lasso Regression:
Accuracy:  0.548752176436448
Precision:  0.542642924086223
Recall:  0.6666666666666666
F1:  0.598295014208215
Mean Absolute Error (MAE): 0.4512478235635519
Mean Squared Error (MSE): 0.4512478235635519
Root Mean Squared Error (RMSE): 0.6717498221537181
R-squared (R2): -0.8051104703905938

# D. Interpretation of results and their implications
Logistic Regression:
An accuracy score of 0.61 signified that 61% of our forecasts were correct relative to the total number of guesses. A precision score of 0.65 indicated that our model accurately predicted 65% of forecasts in the positive direction (i.e., when the team wins). The recall score, quantifying the model's ability to correctly identify positive class instances, stood at 0.62, representing the ratio tp/(tp + fn), where tp denotes true positives and fn represents false negatives. This metric assumes significance, particularly in scenarios where the cost of false negatives is substantial, such as predicting cancer from healthcare data.

Given the objective of predicting the winning team in games and the absence of tangible costs associated with false negatives, optimizing the model for precision rather than recall is deemed optimal. With an F1 score of 0.63, it was evident that the model could be further refined to yield improved results. Presently, indicator scores fell within the 0.60-0.65 range, prompting the pursuit of enhancements to elevate model performance to a target range of 0.70-0.80. The acceptance threshold for the minimum score was set at 0.50, considering that a completely random W/L prediction would result in a 0.50 accuracy rating for a binary variable.

Lasso Regression:
The Lasso Regression model yielded a moderate level of predictive accuracy, with an accuracy score of 0.548. This value suggests that approximately 54.7% of the model's forecasts were correct relative to the total number of predictions. The precision score, indicating the proportion of correctly predicted positive instances among all instances predicted as positive, was 0.577. This implies that the model accurately predicted around 57.7% of instances where the target variable was positive, reflecting a reasonable level of precision in identifying relevant outcomes. However, the recall score, measuring the model's ability to capture all positive instances from the dataset, stood at 0.643. This value indicates that the model identified approximately 64.3% of all actual positive instances, suggesting a relatively high level of recall.

Despite the model's moderate performance in accuracy, precision, and recall, its F1 score, which harmonizes precision and recall, was 0.608. This value suggests that the model achieved a balanced trade-off between precision and recall, indicating reasonable overall performance. However, the model's performance in terms of error metrics was less favorable, as evidenced by the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) values, all hovering around 0.452. These error metrics indicate the average magnitude of errors between predicted and actual values, with lower values desired for better model performance. Furthermore, the negative R-squared (R2) value of -0.825 suggests that the Lasso Regression model performed poorly in explaining the variance in the target variable, indicating a lack of fit to the data.

Ridge Regression:
The Ridge Regression model exhibited a moderate level of predictive accuracy, with an accuracy score of 0.553. This score suggests that approximately 55.3% of the model's forecasts were correct relative to the total number of predictions, indicating a reasonable degree of predictive capability. Moreover, the precision score of 0.588 implies that the model accurately predicted around 58.8% of instances where the target variable was positive, demonstrating a satisfactory level of precision in identifying relevant outcomes. However, the recall score, measuring the model's ability to capture all positive instances from the dataset, stood at 0.608. This value indicates that the model identified approximately 60.8% of all actual positive instances, reflecting a moderate level of recall.

Despite the model's moderate performance in accuracy, precision, and recall, its F1 score, which provides a harmonic mean of precision and recall, was 0.598. This value indicates a balanced trade-off between precision and recall, suggesting overall reasonable performance. However, the model's performance in terms of error metrics was less favorable, with Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) values all approximately equal to 0.447. These error metrics represent the average magnitude of errors between predicted and actual values, with lower values desired for improved model performance. Additionally, the negative R-squared (R2) value of -0.803 suggests that the Ridge Regression model struggled to explain the variance in the target variable, indicating a lack of fit to the data.

Decision Tree:
The Decision Tree model displayed a notable divergence in accuracy between training and testing datasets, with a training accuracy of 0.634 and a testing accuracy of 0.608. This discrepancy suggests a potential issue with overfitting, as the model performed better on the training data compared to unseen testing data. Despite achieving a moderate level of accuracy on the testing set, there remains room for improvement to enhance generalization performance. Consequently, further optimization or regularization techniques may be warranted to mitigate overfitting and improve the model's ability to generalize to unseen data, ensuring robust performance in real-world applications.

Random Forest:
The Random Forest model exhibited a moderate level of accuracy, with an overall accuracy score of 0.589. This score reflects the proportion of correct predictions made by the model across the entire dataset, suggesting a reasonable degree of predictive capability. However, while the model demonstrated a satisfactory level of accuracy, further examination of additional performance metrics such as precision, recall, and error metrics would provide a more comprehensive evaluation of its predictive performance. Additionally, exploring techniques for hyperparameter tuning and model optimization could potentially enhance the Random Forest model's predictive accuracy and generalization capability, thereby improving its effectiveness in real-world applications.
C. Comparison with existing literature or benchmarks, if applicable
Similar notebooks found online had a 70% accuracy score from their Logistic Regression model. This indicates that either our data and/or our models could have had more optimization to become better predictive models.
D. Comparing the various methods
Logistic Regression emerged as the best-performing model in terms of accuracy compared to the other techniques, boasting a score of 0.61. Its superior performance can be attributed to its suitability for binary classification tasks, effective capture of nonlinear relationships between features and outcomes, and strategic emphasis on precision, aligning well with the objective of predicting winning teams in games. Logistic Regression's simplicity and interpretability facilitated better model tuning and optimization, while its ability to accurately estimate coefficients even with a substantial amount of training data further bolstered its predictive capabilities. These factors combined to make Logistic Regression the optimal choice for accurately predicting game outcomes in this scenario.


# V. Challenges Encountered
## A. Identification of technical challenges faced during the project
Converting data from NBA API stats took about 90 minutes and we decided to split our notebooks into several parts to handle the data wrangling and modeling separately. In addition, our lasso and ridge regression models were quite inaccurate and when examining the MSE and MAE, we knew that there was a lot more room for improvement. The lasso and ridge models are actually both random forest models that were then trained with features determined by the lasso and ridge regressions. In addition, the handling of large data and choosing which values to compute during data wrangling required discussion amongst ourselves to figure out what really mattered from the NBA API dataset.

## B. Reflection on lessons learned and improvements for future projects
From this project, we learned that converting data from the NBA API stats was time intensive. Inaccuracies in the lasso and ridge regression models, evident from high MSE and MAE, emphasized the need for rigorous model evaluation and refinement, suggesting the exploration of alternative regression techniques. Additionally, the project highlighted the importance of transparent model development processes and collaborative discussion in handling large datasets, informing future projects for more efficient data processing and enhanced collaboration strategies.


