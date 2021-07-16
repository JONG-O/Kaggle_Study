# Kaggle_Study

(1) Feature engineering Feature Engineering: 1 of 6
https://www.kaggle.com/ryanholbrook/what-is-feature-engineering

Welcome to Feature Engineering!
In this course you'll learn about one of the most important steps on the way to building a great machine learning model: feature engineering. You'll learn how to:

determine which features are the most important with mutual information
invent new features in several real-world problem domains
encode high-cardinality categoricals with a target encoding
create segmentation features with k-means clustering
decompose a dataset's variation into features with principal component analysis
The hands-on exercises build up to a complete notebook that applies all of these techniques to make a submission to the House Prices Getting Started competition. After completing this course, you'll have several ideas that you can use to further improve your performance.

Are you ready? Let's go!
(2) Mutual Information Feature Engineering: 2 of 6
https://www.kaggle.com/ryanholbrook/mutual-information
ntroduction
First encountering a new dataset can sometimes feel overwhelming. You might be presented with hundreds or thousands of features without even a description to go by. Where do you even begin?

A great first step is to construct a ranking with a feature utility metric, a function measuring associations between a feature and the target. Then you can choose a smaller set of the most useful features to develop initially and have more confidence that your time will be well spent.

The metric we'll use is called "mutual information". Mutual information is a lot like correlation in that it measures a relationship between two quantities. The advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships.

Mutual information is a great general-purpose metric and especially useful at the start of feature development when you might not know what model you'd like to use yet. It is:

easy to use and interpret,
computationally efficient,
theoretically well-founded,
resistant to overfitting, and,
able to detect any kind of relationship



