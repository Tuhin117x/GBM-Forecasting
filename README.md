# Digi-Spark Prediction Challenge

The Digi-Spark Prediction Challenge is a data science competition organized by the Unilever team which comprises of two tracks. Track 1 involves a forecasting business problem comprising of predicting sales volume for new product launches based on social media reviews. Track 2 uses the same data-set but requires participants to submit a well prepared storyboard using PowerBI. 

![alt text](https://github.com/Tuhin117x/GBM-Forecasting/blob/main/3.%20Others/Background.jpg)

This repository houses the code used by our team for the Unilever Digi-Spark Prediction Challenge for Track 1. Our team has used the LightGBM library for preparing a forecasting pipeline which utilizes the social media reviews for generating different features which can be used for training a linear predictor model. The underlying data used for feature engineering was analyzed using simple EDA techniques which has been documented in our EDA workbook. The analysis revealed that the data is for three specific markets - US, GB and CA. The review body column encompasses all the text based reviews for each product. Each of these reviews were pre-processed to generate features like the following<br />
<br />
[=] Number of Positve Keywords<br />
[=] Number of Negative Keywords<br />
[=] Similarity Index to Keywords in Spam Texts<br />
[=] Overall Sentiment Score<br />
[=] Review Word Count<br />
<br />
All these features were finally fed to our forecasting pipeline which used the Gradient Boosting Technique to generate the final predictions on our dataset. Since the volume of input data was small in volume, we experienced significantly higher error rates. The average RMSE score was too low and we tried optimizing model parameters to improve the overall RMSE of the model. Finally the pipeline was run on the final test dataset - holdout_features.csv to generate the final set of predictions.
