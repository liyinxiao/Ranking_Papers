Search Ranking Papers
==========

Collection of papers that I have personally found helpful for search ranking / recommendation systems (updating).

## My personal notes / ideas

Deep Neural Networks for YouTube Recommendations
- "Training examples are generated from all YouTube watches (even those embedded on other sites) rather than just watches on the recommendations we produce." The model is predicting watch time per impression, but it is unclear how YouTube constructs its training data (impression data). For example,
  * If a video is at the bottom of the recommendation list and user may not scroll down, does this count as an impression?
  * If a video is shown as relevant videos of an opened video, does this count as an impression?


Deep Learning Recommendation Model for Personalization and Recommendation Systems
- In section 5.1, they evaluate the accuracy of the model on Criteo Ad Kaggle dataset. In this Kaggle competition, logloss is used as the evaluation metric. Why don't they also use logloss and compare their model with the winning models?
