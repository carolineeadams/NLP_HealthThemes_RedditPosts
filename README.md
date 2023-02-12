# Using Clustering and Text Prediction to Identify Themes in Reddit Health Posts
Natural language processing (NLP) is being increasingly used for public health surveillance
purposes with a focus on online forums and social media sites, as they are rich with information
about how individuals experience disease and treatment, as well as information about how
many people have or are concerned about specific conditions. To effectively use NLP to glean
meaningful insights for public health surveillance purposes, it is critical for researchers to be
able to differentiate between posts that literally or figuratively discuss health. To start on this
effort, Naseem et al. (2022) used the Reddit API to collect Reddit posts that discussed health
and manually classified them into posts that discussed health figuratively and non-figuratively.

This analysis utilized the dataset compiled by Naseem et al. (2022) to see if text clustering and
prediction efforts could easily identify between these categories for future efforts trying to
distinguish between these types of posts on other platforms or with additional Reddit content.
Text clustering was performed using the K-means algorithm, resulting in four clusters focused
on general chronic health conditions and symptoms, Alzheimer’s disease, allergic reactions, and
heart attacks. Each cluster contained a mixture of personal, non-personal, and figurative health
mentions, indicating that the clustering algorithm did not match the categorizations completed
by Naseem et al. (2022). Prediction efforts utilized the K Nearest Neighbors and Naïve Bayes
algorithms to predict whether a post used health terms literally or not. Both algorithms
achieved approximately 80% accuracy, with Naive Bayes performing slightly better than KNN
across all evaluation metrics.

Although the K-means clustering algorithm did not identify clusters that matched the labels
added by Naseem et al. (2022), the results did provide useful insight into top health themes
discussed on the platform. Furthermore, the high accuracy of the prediction efforts indicates
that it may be a useful approach for future efforts aiming to distinguish between literal and
non-literal uses of health terms. however, given that a substantial proportion of posts were
incorrectly classified, future researchers should involve supplemental qualitative assessment
procedures to ensure validity of findings.

Code for this analysis can be found in the file: clustering-prediction-nlp-code.ipynb
The full report can be found in the file: clustering-and-predicting-themes-report.pdf
