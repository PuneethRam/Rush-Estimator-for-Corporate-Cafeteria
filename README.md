# Rush-Estimator-for-Corporate-Cafeteria
Rush Estimator for Corporate Cafeteria

Bayesian Loss for Crowd Count Estimation with Point Supervision

For training crowd count estimators, the model only requires point annotations for each training image, i.e., only one pixel of each person is labeled (typically the center of the head).Using the point annotations, Bayesian loss, a novel function constructs a density contribution probability model. This loss function is differentiable and can be readily applied to any Convolutional Neural Network (CNN) model using the standard back propagation training algorithm. The count of the people is entered into excel sheet and this csv file is used as a dataset to predict the approximate crowd at a given time. The dataset is given to RNN algorithm to get the prediction of the crowd for the upcoming days.

bcc.py file is used to get the count of the people when we input an image. And thse counts are used in the dataset and it is given as input to the crowdcount.ipynb file to get the predictions. The model is given in the Releases.
