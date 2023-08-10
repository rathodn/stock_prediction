# stock_prediction
Summary:
This repo illustrates implementation of XGBoost and Deep Neural Net on predicting stock market

Objective: 
(1) Learn from past 4 years of daily stock market (NYSE and NASDAQ) data and train Machine Learning and Deep Learning models that predict what stocks could give 30% or more return in next 8 business days. 
(2) Perform analytics on stocks data and categorize Tickers for potential actionable trading strategy

Notebooks and what it does:
(1) 0GettingData/0get_Indices_data.ipynb : Gets GSPC indice data for last few years on daily basis and 
(2) 0GettingData/1get_EODHistoricaldata.ipynb : Gets OHLCV data for last few years for ~7.5K stocks listed in USA market on daily basis and filters down to ~1400 based on some criteria
(3) 1ProcessingData/0_0_FE_indices.ipynb: prepares some features based on GSPC indice
(4) 0GettingData/2get_Fundamentals_data.ipynb: gets fundamental data for eligible ~1400 stocks
(5) 1ProcessingData/0_1_Feature_Engineering.ipynb: prepares around 300 features based on technical analysis/ volume etc.
(6) 2Model/0_D1_XGB_EP_Swing.ipynb: implements XGBoost on prepared data and classifies stocks that could potentially give 30% or more return in next 8 days
(7) 2Model/0_D1_DNN_EP_Swing.ipynb: implements Deep Neural Net on prepared data and classifies stocks that could potentially give 30% or more return in next 8 days
(8) 1ProcessingData/1_3_Analytics_insights.ipynb: Gives some analytics on recent performance of stocks
(9) 1ProcessingData/1_1_Ana_Action.ipynb: Implements various trading strategies and showcases potiential Tickers for each strategy

Run in this Sequence:
1 and 2 can start in parallel
3 can start after 1 is complete
4 can start after 2 is complete
5 can start after 4 is complete
6,7, 8 can start in parallel after 5 is complete
9 can start after 6 is complete


How to Use:
> First read the Disclaimer below/at the end.
> I am publishing this just for demonstration of XGBoost and DNN implementation on stock market data. 
> I have used data from https://eodhistoricaldata.com/ and my pipeline demonstrates how to get data there. However, I have developed my data data gathering pipeline few years back and there might be better library/SDK available on the source.
> I have published considerable number of features, although there are many I have removed from original pipeline to keep it private. 
> Models are trained on 3-4 years of daily data upto SPLIT_DATE (refer in notebook) and showcases prediction on validation set (size - 1 week, and is in future to SPLIT_DATE.) There will be a gap of 8 days between SPLIT_DATE and start of validation set to prevent data leakage. Models are optimized using aucpr metric and considering imbalanced classes, appropriate weights are provided to the model. Final performance (not model precision/recall) consideration is based on getting top 5-7 TP (daily atleast 1 opportunity) from topN probability provided by model for given criteria.
For example Here is XBB performance on validation set:
(11, 35)
(7, 35)
   win rate: 0.6364
weekly opps: 8.75
Interpretation: weekly opportunity that gives 30% or more return in 8.75, with win rate of ~63%. In the validation set out of 11 top model probability 7 gave 30% or more return in 8 days.
> Some considerations:
  - you would find that train objective and final winrate calculation is based on differently calculated targets. This is to make it more robust. Also, there are other filters provided such as volume, volatility, direction, price etc before calculating final performance on top of model prediction. This is to align with market condition and trading strategy.
> Feel free to implement your own performance criteria/calculation. This pipeline can further do more feature engineering, hyperparameter tuning, feature selection etc. to improve accuracy.

> 1ProcessingData/1_3_Analytics_insights: use this to understand past trends and Tickers that god good returns for Long/Short 
> 1ProcessingData/1_1_Ana_Action.ipynb: I have removed some cells to keep calculations private, but demoed some strategies and how to prepare it
> 0GettingData/3get_Hourly_Data.ipynb : I have included this file in the repo but not used it in the training. This is just to demonstrate some of the features from intraday data. My private ML pipeline uses it for intraday predictions. You can include it to build some intraday strategies.
> Performance.ipynb : This is demonstrating performance on actual test data i.e. date after SPLIT_DATE. Outputs of cells in this notebook doesn't represent SPLIT_DATE from train. This is just to demonstrate hike/drop on various strategies from 1_1_Ana_Action. Some of the strategies are also different from 1_1_Ana_Action.

> Hopefully this give some idea on developing some of the features for stock prediction using Python and implementation of XGBoost and DNN. Enjoy!


Disclaimer: Risk Management and Entertainment Notice
Please be advised that the predictive modeling algorithms provided here are intended for entertainment and educational purposes only. The algorithm's predictions and recommendations should not be considered as professional advice for making critical decisions. While efforts have been made to ensure the accuracy of the model and its results, there are inherent risks and limitations associated with any predictive model.

1. Risk Management:
   - The predictive model's accuracy is based on the data and assumptions used for its development. There might be unforeseen factors that can influence the outcomes.
   - Predictive models are not foolproof and may not accurately predict real-world outcomes all the time. It is important to exercise caution and critical thinking before relying on the model's recommendations.
   - Decision-making should always be supplemented with thorough research, expert advice, and consideration of individual circumstances.

2. Not a Professional Advisor:
   - The predictive model's output is not a substitute for professional advice from qualified experts, including but not limited to financial advisors, legal experts, medical professionals, or any other field requiring specialized knowledge.
   - Users of the algorithm should seek advice from appropriate professionals when making important decisions that may have legal, financial, medical, or other significant consequences.

3. Entertainment Only:
   - This algorithm is offered solely for entertainment and educational purposes. Any decisions or actions taken based on the algorithm's output are the sole responsibility of the user.
   - The algorithm's predictions should not be considered as endorsements, guarantees, or warranties of any kind.

By using this repo / predictive modeling algorithms, you acknowledge and accept the inherent risks associated with predictive modeling, as well as the limitations and entertainment nature of the provided algorithm. The developers and publishers of this algorithm disclaim any liability for decisions made or actions taken based on its predictions. Users are encouraged to exercise due diligence, critical thinking, and seek professional advice when needed.

Please proceed with the understanding that these predictive modeling algorithms are meant for entertainment/educational purposes and is not a substitute for sound judgment and professional advice.

Nilesh Rathod
10 Aug, 2023
