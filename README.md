# NFL-Spread-Prediction

A random forest LGBM model and simple inference app to predict the final spread of NFL games.

Directory breakdown - 

NFL-Score-Prediction/
        2020df_week20v2full.csv - Dataframe used to build the later df for inference
        2021df_pretest.csv - Dataframe of earlier games in the year used to build preseason/early week games. Not used later season
        classmodel-17-5-home.txt - Current classification model
        NFL-PreSeason-EarlyWeek_Delivery.py - Current .py that arranges the data, loads models, and performs inference. Also lays out streamlit app
        NFL2021-PreWeeks_parser.py - Parser to build the "df_pretest" dataframe
        NFLregmodel_rmse_3_27_home.txt - Current regression model
        Procfile - Needed for streamlit app
        README.md
        requirements.txt
        runtime.txt - Needed for streamlit app

Data for training and inference is scrapped from https://www.pro-football-reference.com. The data is parsed on a per team, per week, per year basis. The training data is based on stats from 2018-2020 of play. Going back further in time could help generalization but might also diverge training because of outdated information, non-current players, massive roster changes with the more time included, etc.

GridSearch was used to establish a good baseline for common parametres on the whole of the data. Then after fitting the best performing model, I analyzed feature importance to indetify some features that could be removed to help with generalization. Here are the features remaining (for the regression model) after removing the ['DEF1stD', 'TOOFF'] columns that were barely affecting the decision trees.
![lgbm-reg-feature-importance](https://user-images.githubusercontent.com/85711261/131881827-0979bd9a-3a27-49ba-b964-5347fd6805bb.png)

In the end the "Home" column was also removed. Both a regression model and classification model were trained, with a few different features used for the classification model vs. the regression model. This should also help with generalization at inference time of the ensemble. The regression model is trained to infer the score per team given the matchup, which is then converted into a spread for the UI. The classification model is trained to predict a win or loss per team, which is converted into an integer (-1, 0, or 1) based on the two predictions per matchup.

In sports betting, a win percentage of at least 60% is usually considered as the baseline of profitability given proper bankroll management. At inference the bettor should only consider betting on matches where both the regression model and the classification model agree on the winning team, and where said regression prediction has betting value when compared to the Vegas lines. An in depth betting strategy guide is beyond the scope of this project, and any results from the models should be only one part of a multipart research inquiry for any gambling.

Dataframe before cleaning the features. Stats at inference are based on an average of the last X weeks of performances.
![df-head-regmodel-before-prep](https://user-images.githubusercontent.com/85711261/131880331-da5b5677-8468-48de-a2a4-abf2cedd45af.png)

View of final app page.
![streamlit-preds-new-full-cut](https://user-images.githubusercontent.com/85711261/131880143-d8dd7a41-df83-455c-a5ce-0c8a950768dd.png)

App url - http://nfl-spread-predictor-2021.herokuapp.com

