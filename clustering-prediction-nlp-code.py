#!/usr/bin/env python
# coding: utf-8

# ## Caroline Adams
# ## Clustering and Predicting Themes in Reddit Health Posts

# The dataset I am using for this analysis was compiled by Naseem et al. (2022) and contains health-related content from 15 subreddits on Reddit focused on health, daily activities, and fun. Using an API, the authors collected 10,015 unique posts from January 1, 2015, through March 19, 2021. The authors labeled each post as one of three categories: figurative health mention (FHM; discussing health terms and topics figuratively or hyperbolically, not literally), nonpersonal health mention (NPHM; discussion of health condition/symptoms generally), and personal health mention (PHM; discussion of health condition/symptoms in relation to a person). The authors have made this dataset publicly available on GitHub, however, the subreddit and time origination of each post was not included in this public version.

# I used the dataset from Naseem et al. (2022) to see if text prediction methods could accurately distinguish between posts that talk about health metaphorically and those that discuss the health of actual people. Text clustering was performed using the K-means algorithm, resulting in four clusters focused on general chronic health conditions and symptoms, Alzheimer’s disease, allergic reactions, and heart attacks. Each cluster contained a mixture of personal, non-personal, and figurative health mentions, indicating that the clustering algorithm did not match the categorizations completed by Naseem et al. (2022). Prediction efforts utilized the K Nearest Neighbors and Naïve Bayes algorithms to predict whether a post used health terms literally or not.

# The code for this analysis is included below.

# The writeup of the findings can be found in the accompanying report, "Clustering and Predicting Themes in Reddit Health Posts."

# ## Setup & Descriptive Information

#importing packages for analysis

#data wrangling
import json
import requests
import csv
import numpy as np
import pandas as pd

#plotting & visualizations
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import altair as alt
from altair_saver import save
from wordcloud import WordCloud

#modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold # Cross validation
from sklearn.model_selection import cross_validate # Cross validation
from sklearn.model_selection import GridSearchCV # Cross validation + param. tuning.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ### Obtaining Data

#reading in dataset from github
hmc=pd.read_csv("https://raw.githubusercontent.com/usmaann/RHMD-Health-Mention-Dataset/main/RHMD_3_Class.csv")

#examining top rows of dataset
hmc.head()

# assessing for missing data
hmc.isnull().sum()


# ### Table 1 in Report

#calculating how many posts there are per label
hmc['Label'].value_counts()


# ### Figure 1 in Report

#calculating a new variable for post length
hmc['length'] = hmc.apply(lambda row: len(row.Text), axis=1)

#groupby label and calculate average post length
hmc_len=hmc.groupby(by='Label').agg('mean').reset_index()

#turn label into categorical variable
hmc_len["Label"] = hmc_len["Label"].astype('category')

#plot average post length by label

#set figure size
plt.figure(figsize=(10,7))
#initiate bar plot
plt.bar(hmc_len['Label'], hmc_len['length'])
#set x axis ticks
plt.xticks(np.arange(3), ("Figurative Mentions","Non-Personal Mentions", "Personal Mentions"))
#add x axis label
plt.xlabel("Health Content Category")
#add y axis label
plt.ylabel("Character Count")
#add title
plt.title("Length of Posts by Content Category")


plt.show()
#plt.savefig("length_bar.png")


# ### Figure 2 in Report

#turn text column into a list
text_list=hmc['Text'].to_list()

#join all items in list with commas in between
text_join=",".join(text_list)

#create word cloud for all words in all posts
wordcloud=WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color="steelblue", scale=4)
wordcloud.generate(text_join)
wordcloud.to_image()
#wordcloud.to_file("wordcloud.png")


# ### Tables 2-4 in Report

#manually list top health terms identified from word cloud
cloud_keywords=["feel", "depression", "ptsd", "heart attack", "headache", "cancer", "asthma", "allergic", "treatment", "sick", "stroke", "fever", "ocd", "help", "diabetes", "symptom", "cough", "died", "diagnosed", "alzheimer", "live", "sleep", "coughing", "brain", "migraine", "die", "anxiety", "drug", "cat", "mental health", "pain", "addiction", "body", "covid", "risk"]

#manually reorganize terms into lists of diseases, symptoms, and other
diseases=['covid', "addiction", "mental health", "anxiety", "alzheimer", "depression", "diabetes", "ocd", "ptsd", "asthma", "cancer", "stroke", "heart attack", "allergic"]
symptoms=['pain', "coughing", "cough", "symptom", "fever", "sick", "headache", "feel", "migraine"]
other=['body', "risk", "cat", "drug", "die", "brain", "sleep", "live", "diagnosed", "died", "help", "treatment"]


# ### Table 2 in Report

#set empty list to capture disease term
count_dis=[]

#for loop to count how often disease terms appeared
for term in diseases:
    count=0
    for text in text_list:
        if term in text:
            count+=1
    count_dis.append(count)
    terms.append(term)

#create data frame for just frequency of disease terms
count_dis_df= pd.DataFrame(list(zip(diseases, count_dis)),
               columns =['Condition Term', 'Frequency'])
#display terms
count_dis_df


# ### Table 3 in Report

#set empty list to capture symptom terms
count_sym=[]

#for loop to count frequency of symptom terms
for term in symptoms:
    count=0
    for text in text_list:
        if term in text:
            count+=1
    count_sym.append(count)
    terms.append(term)

#new df of frequencies of symptom terms only
count_sym_df= pd.DataFrame(list(zip(symptoms, count_sym)),
               columns =['Symptom Term', 'Frequency'])
#display df
count_sym_df


# ### Table 4 in Report

#empty list to capture other terms
count_other=[]

#for loop to count frequency of other terms in all posts
for term in other:
    count=0
    for text in text_list:
        if term in text:
            count+=1
    count_other.append(count)
    terms.append(term)

#create new df for frequency of other terms only
count_other_df= pd.DataFrame(list(zip(other, count_other)),
               columns =['Other Health Term', 'Frequency'])

#display df
count_other_df


# ### Figures 3-5 in Report

#turn label column into a list
label_list=hmc['Label'].to_list()

#create empty lists to capture counts of terms for each text
count_list=[]
terms=[]
texts=[]

#iterate through each post and each health term to calculate frequency
for text in text_list:
    for term in cloud_keywords:
        count=0
        terms.append(term)
        texts.append(text)
        if term in text:
                count+=1
        count_list.append(count)


#create new df with each text, each health term, and its frequency
health_term_freq = pd.DataFrame(list(zip(texts, terms, count_list)),
               columns =['Text',"Term", 'Frequency'])

#merge original dataset onto text-term df to include labels
health_term_freq_merge=health_term_freq.merge(hmc,on='Text',how='left')

#groupby term and label, sum variables, and reset inex
health_term_freq_merge=health_term_freq_merge.groupby( [ "Term", "Label"] ).sum().reset_index()

#set label to categorical variable
health_term_freq_merge["Label"] = health_term_freq_merge["Label"].astype('category')

#rename labels
health_term_freq_merge['Label']=health_term_freq_merge['Label'].replace([0, 1, 2], ["Figurative Mention", "Non-Personal Mention", "Personal Mention"])


# ### Figure 3 in Report

#initiate grouped bar chart
phm_chart = alt.Chart(health_term_freq_merge[health_term_freq_merge['Label']=="Personal Mention"]).mark_bar(color="orange").encode(
    x=alt.X('Term:N', sort="-y"),
    y="Frequency:Q").properties(
    title='Top Health-Related Words in Personal Health Mention Posts'
)

#display chart
phm_chart


# ### Figure 4 in Report

#initiate groupbed bar chart
nphm_chart = alt.Chart(health_term_freq_merge[health_term_freq_merge['Label']=="Non-Personal Mention"]).mark_bar(color="pink").encode(
    x=alt.X('Term:N', sort="-y"),
    y="Frequency:Q").properties(
    title='Top Health-Related Words in Non-Personal Health Mention Posts'
)

#display chart
nphm_chart


# ### Figure 5 in Report

#initiate groupbed bar chart
fhm_chart = alt.Chart(health_term_freq_merge[health_term_freq_merge['Label']=="Figurative Mention"]).mark_bar().encode(
    x=alt.X('Term:N', sort="-y"),
    y="Frequency:Q").properties(
    title='Top Health-Related Words in Figurative Health Mention Posts'
)

#display chart
fhm_chart


# ## TF-IDF Weighting

# ### Apendix Figure 8

#set empty lists to capture number of features and min df
num_feat_list=[]
min_df_list=[]

#iterate through 30 times and calculate the number of features for each min df
#append values to lists above
for i in range(30):
    tfidf_vectorizer = TfidfVectorizer(min_df=i, stop_words='english')  #initialize tf idf vectorizer
    tfidf = tfidf_vectorizer.fit_transform(text_list)  #fit vectorizer to text list
    num_feat=tfidf.shape[1]  #pull part of shape that represents number of features
    num_feat_list.append(num_feat)  #append num features to list
    min_df_list.append(i)  #append min df number


#create dataframe of results of iteration
mindf_df=pd.DataFrame()
mindf_df['num_feat']=num_feat_list
mindf_df['min_df']=min_df_list

#plot number of features by min df values

#set x axis values
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 24, 25, 26, 27, 28, 29, 30]
x_axis = np.arange(len(x))
plt.figure(figsize=(10,7))  #set fig size
plt.scatter(mindf_df['min_df'], mindf_df['num_feat'])  #initiate scatter plot
plt.xticks(x_axis, x)  #plot x ticks
plt.xlabel("Minimum Document Frequency")  #add x axis label
plt.ylabel("Number of Features")  #add y axis label
plt.title("Number of Features by Minimum Document Frequency")  #add title
plt.show()  #show plot
#plt.savefig("FeatureReduction.png", dpi=300)

#create document term matrix using tfidf
#initiate vectorizer instance
#remove stop words
#remove tokens that are numbers only
#set min df to 10
vector = TfidfVectorizer(stop_words='english', min_df=10, token_pattern="(?ui)\\b\\w*[a-z]+\\w*\\b")

#fit vectorizer to text data
dtm = vector.fit_transform(hmc['Text'])

#calculate shape of document term matrix
dtm.shape


# ## Clustering

# ### Appendix Figure 9

#set range of 2 through 10
k_vals = range(2,11)
inert = []  #create empty list to capture inertia values
silh = [] #create empty list to capture silhouette scores

for i in k_vals:
    km = KMeans(init='k-means++', n_clusters=i, max_iter=300,n_init=10)  #initiate instance of kmeans
    inert.append(km.fit(dtm).inertia_)  #fit to dtm and pull inertia value
    silh.append(silhouette_score(dtm, km.fit_predict(dtm), metric='euclidean'))  #fit to dtm and pull silhouette score

#set x axis values and range
x=[0, 1,2,3,4,5,6,7,8,9,10]
x_axis = np.arange(len(x))

plt.figure(figsize=(10,7))  #set fig size
plt.plot(k_vals, inert)  #plot k values versus inertia values
plt.xticks(x_axis, x)  #plot x ticks
plt.xlabel('Number of Clusters (K)')  #add x axis label
plt.ylabel('SSE')  #add y axis label
plt.xlim(1,)  #set x axis limit
plt.title('Sum of Squared Errors (SSE) versus Number of Clusters')  #add title
plt.show()
#plt.savefig("Kmeans-SSE.png", dpi=300)


# ### Figure 6 in Report

#plot values of k against silhouette scores
plt.plot(k_vals, silh)
plt.xticks(x_axis, x)  #add x ticks and labels
plt.xlabel('Number of Clusters')  #add x axis label
plt.ylabel('Silhouette Score')  #add y axis label
plt.title('Silhouette Score Versus Number of Clusters')  #add plot title
plt.xlim(1,11)  #set x axis limit
plt.show()
#plt.savefig("Silhouette_clusteringKMeans.png", dpi=300)


# ### Cluster Analysis When K=3

#initiate instance of k-means algorithm with n clusters = 3 and fit to DTM
clus_labels3 = KMeans(init='k-means++', n_clusters=3, n_init=10, random_state=0).fit_predict(dtm)

#create dataframe with each post and its assigned cluster label
cluster_df3 = pd.DataFrame({'text':hmc['Text'], 'cluster':clus_labels3})


# ### Table 5 in Report

#calculate number of documents in each cluster
cluster_df3['cluster'].value_counts()


# ### Table 6 in Report

clusters3 = [cluster_df3[cluster_df3['cluster']==i] for i in np.arange(3)]

#define function that takes in documents and number of words
def top_words(documents, num_words):
    """Accepts a vector of documents and returns the specified number of words with
    the highest average tfidf score"""
    #apply TFIDF weighting
    vect = TfidfVectorizer(stop_words='english')
    #fit weighting to documents
    dtm = vect.fit_transform(documents)
    #get list of terms
    term_indices = {index: term for term, index in vect.vocabulary_.items()}
    #set terms as column names
    colterms = [term_indices[i] for i in range(dtm.shape[1])]
    #create df
    dtm_df = pd.DataFrame(dtm.toarray(), columns=colterms)
    #aggregate by mean and sort, return head of list based on number of words set
    return dtm_df.agg('mean').sort_values(ascending=False).head(num_words)

#print list of top 20 words in each cluster
for c in clusters3:
    print(top_words(c['text'], 20), '\n')


# ### Cluster Analysis When K=4

#initiate instance of k-means algorithm with n clusters = 4
clus_labels4 = KMeans(init='k-means++', n_clusters=4, n_init=10, random_state=0).fit_predict(dtm)

#create dataframe with each post and its assigned cluster label
cluster_df4 = pd.DataFrame({'text':hmc['Text'], 'cluster':clus_labels4})


# ### Table 5 in Report

#calculate number of documents in each cluster
cluster_df4['cluster'].value_counts()


# ### Table 6 in Report

clusters4 = [cluster_df4[cluster_df4['cluster']==i] for i in np.arange(4)]

#print list of top 20 words in each cluster
for c in clusters4:
    print(top_words(c['text'], 20), '\n')


# ## Prediction

#collapse personal and non personal health mentions into one class
hmc['Label'].replace(2, 1, inplace=True)

#calculate class counts
hmc['Label'].value_counts()

#create dictionary of term indices
vect.vocabulary_
term_indices = {index: term for term, index in vect.vocabulary_.items()}

#create column name list of terms in term_indices
colterms = [term_indices[i] for i in range(dtm.shape[1])]

#create feature matrix of terms
X = pd.DataFrame(dtm.toarray(), columns=colterms)


#set label equal to y as target array
y = hmc['Label']


# ### Figure 7 in Report


#generating validation curve for K

# Setting the range for the parameter k
parameter_range = [5,10,15,20,25]

# Calculate accuracy on training and test set using the
# n_neighbors parameter with 5-fold cross validation
train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
                                       param_name = "n_neighbors",
                                       param_range = parameter_range,
                                        cv = 5, scoring = "accuracy")

# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis = 1)

# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis = 1)


#plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, mean_train_score,
     label = "Training Score", color = 'r')
plt.plot(parameter_range, mean_test_score,
   label = "Testing Score", color = 'b')

# Creating the plot
plt.title("Validation Curve with K Nearest Neighbors Classifier")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend(loc = 'lower right')
plt.xlim(4,)
plt.show()

#plt.savefig('validation_curve.png', dpi=300)


#setting train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# ### K Nearest Neighbors Model

# Initialize knn algorithms
knn_all = KNeighborsClassifier(n_neighbors=15)  #for cross validation

knn_all.fit(Xtrain, ytrain)  #fit to X and y training data

#apply five-fold cross validation using just training data
scores_knn = cross_val_score(knn_all,
                         Xtrain,
                         ytrain,
                         cv=5)


# Print all five scores from each fold
for i, each in enumerate(scores_knn):
    print(f"CV {i+1}, accuracy score: {each}")


# Get mean score across 5 folds
print(f"Mean CV accuracy score: {scores_knn.mean()}")


#creating predictions using test data
y_preds_knn = knn_all.predict(Xtest)


# ### Table 8 in Report

#computing a confusion matrix for the knn model
#putting into a dataframe and displaying the matrix
pd.DataFrame(confusion_matrix(ytest, y_preds_knn),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# ### Naive Bayes Model

#instantiate Naive Bayes model for cross validation
nb = MultinomialNB()
#fit to X and y training data
nb.fit(Xtrain, ytrain)


# Apply five-fold cross validation to NB model
scores_nb = cross_val_score(nb,
                         Xtrain,
                         ytrain,
                         cv=5)


# Print all five scores from each fold
for i, each in enumerate(scores_nb):
    print(f"CV {i+1}, accuracy score: {each}")


# Get mean score across 5 folds
print(f"Mean CV accuracy score: {scores_nb.mean()}")


# ### Table 9 in Report

#creating predictions using test data
y_preds_nb = nb.predict(Xtest)

#computing a confusion matrix for the NB model
#putting into a dataframe and displaying the matrix
pd.DataFrame(confusion_matrix(ytest, y_preds_nb))

pd.DataFrame(confusion_matrix(ytest, y_preds_nb),
            columns=["Predicted Figurative", "Predicted Literal"],
            index=["Actual Figurative","Actual Literal"]).style.background_gradient(cmap="PiYG")
