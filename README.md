Search Ranking Papers
==========

Collection of papers that I have personally found helpful for search ranking / recommendation systems (updating).

## Great talks on search ranking / recommendation system
Personalization at Amazon Music (ICML 2019)
- https://slideslive.com/38917444/personalization-at-amazon-music

Reinforcement Learning for Recommender Systems: A Case Study on Youtube (2019)
- https://www.youtube.com/watch?v=HEqQ2_1XRTs

Applying Deep Learning to Airbnb Search (Qcon 2019)
- https://www.infoq.com/presentations/dl-airbnb/

Artwork Personalization at Netflix (Data Council 2018)
- https://www.youtube.com/watch?v=UjQMEjkrUGo

Measurement and analysis of predictive feed ranking models on Instagram (Scale 2017)
- https://atscaleconference.com/videos/measurement-and-analysis-of-predictive-feed-ranking-models-on-instagram/

Detecting place visits at scale (Scale 2017)
- https://atscaleconference.com/videos/detecting-place-visits-at-scale/


## My personal notes / ideas on papers

Deep Neural Networks for YouTube Recommendations
- "Training examples are generated from all YouTube watches (even those embedded on other sites) rather than just watches on the recommendations we produce." The model is predicting watch time per impression, but it is unclear how YouTube constructs its training data (impression data). For example,
  * If a video is at the bottom of the recommendation list and user may not scroll down, does this count as an impression?
  * If a video is shown as relevant videos of an opened video, does this count as an impression?

Applying Deep Learning To Airbnb Search
- They tried multi-task learning that optimizes for both booking and long view, and found out that long views increased by a lot while bookings remained neutral. This multi-task learning can be useful for advertisement modeling, where we would optimize for both clicks and conversions.


Deep Learning Recommendation Model for Personalization and Recommendation Systems
- In section 5.1, they evaluate the accuracy of the model on Criteo Ad Kaggle dataset. In this Kaggle competition, logloss is used as the evaluation metric. Why don't they also use logloss and compare their model with the winning models? For this dataset, I believe this model is not likely to perform well.


Amazon Search: The Joy of Ranking Products
- "To manage the size of the training set, we sample unseen examples." An idea inspired by this statement:
  * For pairwise formulation, a good idea might be sampling the possibly unseen examples. For example, user makes a click on item at position 5, then items displayed after position 5 may not be seen and can be sampled.


Online Controlled Experiments at Large Scale
- “We recently ran a slowdown experiment where we slowed 10% of users by 100msec (milliseconds) and another 10% by 250msec for two weeks. The results showed that performance absolutely matters a lot today: every 100msec improves revenue by 0.6%.”
  * Speed wins.
