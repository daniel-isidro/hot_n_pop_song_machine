![logo](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/hnp_logo.jpeg)

Repository for the **Hot'n'Pop Song Machine** project, a Machine Learning song popularity predictor.

The Github repository of the **front-end web app** of the project, that uses the Streamlit app framework and web hosting on Heroku, can be found **[here]( https://github.com/daniel-isidro/heroku_hot_n_pop)**.

You can play with a live demo of the web app **[here](https://hot-n-pop-song-machine.herokuapp.com)**.

# Table of Contents

[Introduction](#introduction) <br>
[Methodology](#methodology) <br>
[Requirements](#requirements) <br>
[Execution Guide](#execution-Guide) <br>
[Data Acquisition](#data-acquisition) <br>
[Data Preparation](#data-preparation) <br>
[Raw Data Description](#raw-data-description) <br>
[Data Exploration](#data-exploration) <br>
[Modeling](#modeling) <br>
[Summary](#summary) <br>
[Front-end](#front-end) <br>
[Conclusions](#conclusions) <br>
[References](#references) <br>
[About Me](#about-me)

# Introduction

My idea for this project started when I found out about the existence since 2010 of the **[Million Song Dataset](http://millionsongdataset.com/)**, a freely-available collection of **audio features and metadata** for a million contemporary popular music tracks. Since music is one of my passions, it seemed appropriate to base one of my first Data Science projects on this subject. The core of the dataset was provided by the company [**The Echo Nest**](http://the.echonest.com). Its creators intended it to perform music identification, recommendation, playlist creation, audio fingerprinting, and analysis for consumers and developers. In 2014 The Echo Nest [was acquired](https://en.wikipedia.org/wiki/The_Echo_Nest) by [**Spotify**](www.spotify.com), which incorporated that song information into their systems. Now those audio features and metadata are available through the free **Spotify Web API.** Finally I chose to **use this API** instead of the Million Song Dataset for the project, thanks to it is flexibility and my will to practice working with APIs.

**Music information retrieval (MIR)** is the interdisciplinary science of retrieving information from music. MIR is a small but growing field of research with **many real-world applications**. Those involved in MIR may have a background in musicology, psychoacoustics, psychology, academic music study, signal processing, informatics, machine learning, optical music recognition, computational intelligence or some combination of these. MIR applications include:

* Recommender systems
* Track separation and instrument recognition
* Automatic music transcription
* Automatic categorization
* Music generation

According to the International Federation of the Phonographic Industry (IFPI), for the full year **2019** total revenues for the **global recorded music market** grew by 8.2% to **US$20.2 billion**. Streaming for the first time accounted for more than half (56.1%) of global recorded music revenue. Growth in streaming more than offset a -5.3% decline in physical revenue, a slower rate than 2018.

Being able to **predict** what songs have the **traits needed** to be popular and stream well is an asset to the music industry, as it can be influential to **music companies** while producing and planning marketing campaigns. It is beneficial even to **artists**, since they may focus on songs that can be promoted later by the music companies, or can become more popular amongst the general public.

**State-of-the-art papers** on MRI verse on audio signal processing, music discovery, music emotion recognition, polyphonic music transcription, using Deep Learning tools. Recent papers (2019) on MRI may be found on the [**International Society of Music Information Retrieval** website](http://www.ismir.net/conferences/ismir2019.html).

# Methodology

### Machine Learning Techniques

**Classification**

On this project we will use supervised learning classification methods, starting with the **logistic regression** method as it is the simplest classification model. As we progress, we will use other non-linear classifiers such as **decision trees** and **support vector machines**.

**Ensemble Learning**

We will apply ensemble learning to combine simple models for creating new models with better results than the simple ones. On this we will try **random forests** and **gradient boosted trees**. We'll also apply **feature preprocessing** (scaling, one-hot encoding) through **pipelines**.

**Dimensionality Reduction**

We will evaluate using dimensionality reduction to remove the least important information (sometime redundant columns) from our data set. We will use **Recursive Feature Elimination** (RFE) and **Principal Component Analysis** (PCA) methods.

### Statistical Methodologies

**Predictive Analytics**

Including a variety of techniques such as data mining, data modeling, etc.

**Exploratory Data Analysis (EDA)**

For taking a first view of the data and trying to make some feeling or sense of it.

# Requirements

We'll use the Anaconda virtual environment with Python 3.7.7 or higher and the following libraries/packages:

### Anaconda Python Packages

* beautifulsoup4
* jsonschema
* matplotlib
* numpy
* pandas
* requests
* scipy
* seaborn
* scikit-learn
* spotipy
* xgboost

For avoiding future compatibility issues, here are the versions of the key libraries used:

```
jsonschema==3.2.0
numpy==1.18.1
pandas==1.0.3
scikit-learn==0.22.1
spotipy==2.12.0
xgboost==0.90
```
### Spotify Account

You'll need a Spotify account (free or paid) to be able to use their web API, and then register your project as an app. For that, follow the instructions found on the ['Spotify for Developers' guide](https://developer.spotify.com/documentation/general/guides/app-settings/):

1. On [your Dashboard](https://developer.spotify.com/dashboard/) click **CREATE A CLIENT ID**.
2. Enter **Application Name** and **Application Description** and then click **CREATE**. Your application is registered, and the app view opens.
3. On the app view, click **Edit Settings** to view and update your app settings.

<img src="https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/spotifydashboard.png" width="300">

**Note:** Find your **Client ID** and **Client Secret**; you need them in the authentication phase.

* **Client ID** is the unique identifier of your application.
* **Client Secret** is the key that you pass in secure calls to the Spotify Accounts and Web API services. Always store the client secret key securely; never reveal it publicly! If you suspect that the secret key has been compromised, regenerate it immediately by clicking the link on the edit settings view.

### *settings.env* file

In order to not uploading your Spotify Client ID and Client Secret tokens to Github, you can create a **.env text file** and place it into your local Github repository. Create a **.gitignore** file at the root folder of your project so the .env file is not uploaded to the remote repository. The content of the .env text file should look like this:

```
  {
    "SPOTIPY_CLIENT_ID": "754b47a409f902c6kfnfk89964bf9f91",
    "SPOTIPY_CLIENT_SECRET": "6v9657a368e14d7vdnff8c647fc5c552"
  }
```


# Execution Guide

![Workflow](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/workflow.png)

For replicating the project, please execute the following **Jupyter notebooks** in the specified order.

1. **[Web scraping](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/web_scraping/ultimate_music_database_web_scraping.ipynb)**

Getting Billboard 100 US weekly hit songs and artist names from 1962 till 2020 from Ultimate Music Database website.

2. **[Get audio features from hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_hit_songs.ipynb)**

Getting audio features from those hit songs, restricted to years 2000-2020, from Spotify web API, whose response contains an audio features object in JSON format.

3. **[Get audio features from random not-hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_not_hit_songs.ipynb)**

Randomly generating 10,000 not-hit songs from years 2000-2020 and getting their audio features from Spotify web API.

4. **[Data preparation](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_prep/data_prep.ipynb)**

Merging both datasets, hit songs and not-hit songs.

5. **[Data exploration](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_exploration/feature_selection_and_data_visualization.ipynb)**

Data visualization and feature selection.

6. **[ML model selection](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/modeling.ipynb)**

Machine learning models analysis and metrics evaluation, using a balanced dataset. Result is a pickled model.

7. **[Prediction](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/model_predict.ipynb)**

Using the pickled model to make predictions on new songs.

### Refining the Model

If you also want to replicate the second part of the project, where we explore using an **unbalanced dataset**, getting more samples of not-hit songs, and **retrain** the model to try **improving the metrics**, please execute the following Jupyter notebooks in the specified order.

8. **[Get more random not-hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_more_not_hit_songs.ipynb)**

Randomly generating 20,000 more not-hit songs from years 2000-2020, to a total of 30,0000, and getting their audio features from Spotify web API.

9. **[Data preparation (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_prep/data_prep_expanded_dataset.ipynb)**

Merging both datasets, hit songs and not-hit songs. Now resulting on an unbalanced dataset, aprox. 3:1 not-hit to hit songs.

10. **[Data exploration (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_exploration/feature_selection_and_data_visualization_expanded_dataset.ipynb)**

Machine learning models analysis and metrics evaluation, now with the expanded unbalanced dataset.

11. **[ML model selection (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/modeling_expanded_dataset.ipynb)**

Machine learning models analysis and metrics evaluation. Result is a second pickled model.

12. **[Prediction (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/model_predict_expanded_dataset.ipynb)**

Using the second pickled model to make predictions on new songs.

# Data Acquisition

### Web Scraping

![Billboard](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/billboard.png)

For getting all the **Billboard 100** weekly hit songs and artist names in the United States, from 1962 till 2020, we perform **web scraping** on the [**Ultimate Music Database**](http://umdmusic.com/default.asp?Lang=English&Chart=D) website. We scrape this web instead of the official [Billboard.com](https://www.billboard.com) as it contains the same data and it is more conveniently formatted (very few ads, no Javascript, no tracking code).

We use [**BeautifulSoup4**](https://www.crummy.com/software/BeautifulSoup/) as our Python library tool for scraping the web.

The result is a data frame with three columns: **year, artist, and title**. Then we save the data frame into a CSV file.

We do **several scraping passes** on the website, covering just one or two decades, to avoid being kicked by the website.

At the end we merge all data frames into one final CSV file, that contains **all hit titles** from **1962** until **late June 2020**.

### Spotify Web API

**Hit Songs**

Now we take the resulting data frame on the previous step, **remove all songs older than 2000** (as older hit songs may not predict future hits since people's preferences change over time), remove duplicates and clean artist and title names with regular expressions (to get better search results).

Then we use [**spotipy**](https://spotipy.readthedocs.io/) Python library to call the **Spotify Web API** and get the audio features of those hit songs.

<img src="https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/spotify_web_api.png" width="300">

Finally we add a column, **success**, with value 1.0 in all rows, that will serve us in the modeling phase of the project.

The resulting data frame has around **8,000 entries**. We store the result into a CSV file.

**Not-hit Songs**

As Machine learning models usually **perform better** with **balanced datasets**, we will need to get other 8,000 not-hit songs that exist in the Spotify catalog.

So we create a **function** that generates around 10,000 **pseudo-random songs** to balance the hit/not-hit songs dataset.

We specify that the **year range** of those random songs as the same one as the selected for hit songs: from **2000** to **2020**.

We put the results on a data frame, then we remove duplicates and nulls, and we add a column, **success**, with value 0.0 in all rows, that will serve us in the modeling phase of the project.

The resulting data frame has around **9,500 entries**. We store the result into a CSV file.

# Data Preparation

In this section we **combine both datasets** (hit songs and not-hit songs), into one data frame, remove duplicates and nulls, and **remove the exceeding not-hit songs** so we get a balanced dataset (same number of rows with *success==1.0* than with *success==0.0*).

The result is a data frame with around **15,700 entries**. We store the result into a CSV file.

# Raw Data Description

### Audio Features

To get a general understanding of the features we are going to work with, let's have a look on the "audio features" JSON object the Spotify Web API returns when searching for a song. From the [Spotify Web API reference guide](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/):


key | type
--- | ---
`acousticness`  <br><small>A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.</small> | Float
`analysis_url` <br><small>An HTTP URL to access the full audio analysis of this track. An access token is required to access this data.</small> | String
`danceability` <br><small>Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.</small> | Float
`duration_ms` <br><small>The duration of the track in milliseconds.</small> | Integer
`energy` <br><small>Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.</small> | Float
`id` <br><small>The Spotify ID for the track.</small> | String
`instrumentalness` <br><small>Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.</small> | Float
`key` <br><small>The key the track is in. Integers map to pitches using standard [Pitch Class notation](https://en.wikipedia.org/wiki/Pitch_class) . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.</small> | Integer
`liveness` <br><small>Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.</small> | Float
`loudness` <br><small>The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.</small> | Float
`mode`  <br><small>Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.</small> | Integer
`speechiness` <br><small>Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.</small> | Float
`tempo` <br><small>The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.</small> | Float
`time_signature` <br><small>An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).</small> | Integer
`track_href` <br><small>A link to the Web API endpoint providing full details of the track.</small> | String
`type` <br><small>The object type: “audio_features”</small> | String
`uri` <br><small>The Spotify URI for the track.</small> | String
`valence` <br><small>A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).</small> | Float


### Statistical Description

**1. First Look**

We have a look at the raw data we got after running steps 1 to 4 on the execution guide above. Here are the first entries of the dataset.

```Python
data.head()
```

![data_head1](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/data_head1.png)

![data_head2](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/data_head2.png)

**2. Dimensions of the Data**

```Python
data.shape
```

```
15714 rows × 19 columns
```

**3. Data Types**

```Python
data.dtypes
```

```
danceability        float64
energy              float64
key                 float64
loudness            float64
mode                float64
speechiness         float64
acousticness        float64
instrumentalness    float64
liveness            float64
valence             float64
tempo               float64
type                 object
id                   object
uri                  object
track_href           object
analysis_url         object
duration_ms         float64
time_signature      float64
success             float64
dtype: object
```

Apparently we have **14 numerical** and **5 categorical** features. But later we'll see that `key`, `mode`, `time_signature` and `success` are also categorical.

**4. Class Distribution**

We are working with a balanced dataset by design.

```python
df[df['success']==1.0].shape
(7857, 19)
```
```python
df[df['success']==0.0].shape
(7857, 19)
```

**5. Data Summary**

```python
df.describe()
```

![data_summary](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/data_summary.png)

**6. Correlations**

```Python
data.corr(method='pearson')
```

![data_correlation](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/data_corr.png)

**7. Skewness**

Skew refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or squashed in one direction or another. The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.

```python
data.skew()
```

```
danceability       -0.757
energy             -0.296
key                 0.019
loudness           -1.215
mode               -0.699
speechiness         1.310
acousticness        0.645
instrumentalness    2.301
liveness            1.978
valence             0.021
tempo               0.134
duration_ms         8.900
time_signature     -2.628
success             0.000
dtype: float64
```

# Data Exploration

### Data Visualization

**Target Countplot**

![bal_classes](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_bal_classes.png)

**Boxplot**

![boxplot](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_boxplot.png)

**Univariate Analysis: Numerical Variables**

![univar_num](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_univar_num.png)

**Univariate Analysis: Categorical Variables**

![univar_cat](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_univar_cat.png)

**Multivariate Analysis: Two Numerical Variables**

![multivar_cat](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_multivar_num.png)

**Multivariate Analysis: Two Categorical Variables**

![multivar_cat](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_multivar_cat.png)

**Correlation Heatmap**

![heatmap](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_heatmap.png)

**Notes of Interest**

* We are working with a **balanced dataset** (by design).
* There is a lot of outliers in the **duration_ms** feature of the not-hit songs.
* Hit songs have **higher danceability, energy** and **loudness** than not-hit songs.
* Hit songs have **lower speechiness, acousticness, instrumentalness** and  **liveness** than not-hit songs.
* Hit songs have **similar levels** of **key, mode, valence, tempo** than not-hit songs.
* Most hit songs have **low variance** in the features **speechiness, instrumentalness, duration_ms** and **time_signature**.
* Songs are more or less **equally distributed among all keys**.
* **Two thirds** of the songs are on the **major mode**.
* Most of the songs are on the **4 beats by bar** (4/4) time signature.
* **Energy** and **loudness** have a **fairly strong correlation** (0.8).
* **Energy** and **acousticness** have a **moderate negative correlation** (-0.7).

### Feature Selection

We will perform an analysis on whether we will need to use all features in the modeling steps or we should drop some features. We will use the Random Trees classifier from scikit-learn as a base model.

**1. Feature Selection and Random Forest Classification**

Using the Random Trees classifier, a 70/30 train/test split, 10 estimators, we get an accuracy of 0.905.

![fs_heatmap1](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_fs_heatmap1.png)

```
Accuracy is:  0.905408271474019
```

**2. Univariate feature selection and random forest classification**

We use the modules  ```SelectKBest``` and ```f_classif``` to find the best 5 scored features.

![feat_scores](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_feat_scores.png)

**3. Recursive feature elimination (RFE) with random forest**

```
Chosen best 5 feature by rfe:
Index(['energy', 'loudness', 'speechiness', 'acousticness', 'duration_ms'], dtype='object')
```
Then we retrain the Random Forest model with only those 5 features.

![fs_heatmap2](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_fs_heatmap2.png)

```
Accuracy is:  0.8835630965005302
```

Accuracy drops to 0.884 with only those 5 selected features.

**4. Recursive feature elimination with cross validation and random forest classification**

Now using the module ```RFECV```from ```sklearn.feature_selection``` we will not only find the best features but we'll also find how many features do we need for best accuracy.

![feat_sel](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_feat_sel.png)

```
Optimal number of features : 13
Best features : Index(['danceability', 'energy', 'key', 'loudness', 'mode',
   'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
   'tempo', 'duration_ms', 'time_signature'], dtype='object')
```

It seems we should use **all our features available**.

 **5. Tree based feature selection and random forest classification**

 If our would purpose would be actually not finding good accuracy, but learning how to make feature selection and understanding data, then we could use another feature selection method.

 In the Random Forest classification method there is a ```feature_importances_``` attribute that is the feature importances (the higher, the more important the feature).

 ![feat_importance](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_feature_imp.png)

 **Feature Extraction**

 We will evaluate using principle component analysis (PCA) for feature extraction. Before PCA, we need to normalize data for better performance of PCA.

 ![feat_extraction](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_feat_ext.png)

 According to variance ratio, 5 components (0 to 4) could be chosen to be the most significant ones. But later we will verify it's **not worthy to drop features on this project**, which does not have so many, and potentially lose predictive power.

# Modeling

We will use several ML Classifiers algorithms, mostly from `scikit-learn`:

* Logistic Regression
* K-nearest Neighbors
* Support Vector Columns
* Decision Tree
* Random Forest

Also:
* XGBoost

We will employ **pipelines** to perform several transformations to the columns and the ML training in one pass. We will transform the columns with **standardization** (for the numerical columns) and **one-hot encoding** (for the categorical columns).

We will use **Logistic Regression** as the **base model**.

For **standardization** we'll use `RobustScaler()` (which is more robust to outliers than other transformations), except when using algorithms involving trees (that are usually immune to outliers), where we'll use `StandardScaler()`.

For **column encoding** we'll mostly use `OneHotEncoder()`, testing if removing the first labeled column (to avoid collinearity) improves the metrics. We'll also test `OrdinalEncoder()` out of curiosity.

In the model analysis, `GridSearchCV` is incorporated to the pipeline and will be very useful to help us find the **optimal algorithm parameters**.

### Metrics

![Metrics_Balanced](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/metrics_bal.png)

* We did **feature importance scoring**, where `loudness` had 40% of the significance, and since `energy` had a fairly strong correlation to `loudness` (0.8), we tried improving the metrics retraining our selected model (XGBoost, outliers removed) leaving `energy` out. But the metrics got worse, the model lost predictive power.

* **Removing 650+ outliers** in the training set did seem to help improving a little the metrics. Most of the outliers came from the random non-hit songs, feature `duration_ms`. Removing the outliers, which were valid measures and not coming from errors, decreased a little the negatives precision but **improved the negatives recall**. It also **improved the positives precision**, and did not change the positives recall.

XGBoost metrics **before** removing the outliers:

```
            precision    recall  f1-score   support

     0.0        0.94      0.87      0.90      1543
     1.0        0.88      0.95      0.91      1600

accuracy                            0.91      3143
macro avg       0.91      0.91      0.91      3143
weighted avg    0.91      0.91      0.91      3143
```

XGBoost metrics **after** removing the outliers:

```
            precision    recall  f1-score   support

     0.0        0.95      0.85      0.90      1561
     1.0        0.87      0.95      0.91      1582

accuracy                            0.90      3143
macro avg       0.91      0.90      0.90      3143
weighted avg    0.91      0.90      0.90      3143
```  

Boxplot after removing the outliers

![boxplot_removed_outliers](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_boxplot_no_outliers.png)

### Overfitting

While checking for **overfitting** we need to obtain the **accuracy difference** between **train and test set** for each fold result. If our model gives us high training accuracy but low test accuracy, our model is overfitting. If our model does not give good training accuracy, we could say our model is underfitting.

![overfitting_curves](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/overfitting_curves.png)

To check whether the model we find by GridSearchCV is overfitted or not, we can use ```cv_results``` attribute of ```GridSearchCV```. ```cv_results``` is a dictionary which contains details (e.g. ```mean_test_score```, ```mean_score_time``` etc. ) for each combination of the parameters, given in parameters' grid. And to get training score related values (e.g. ```mean_train_score```, ```std_train_score``` etc.), we have to pass ```return_train_score = True``` which is by default false.

![overfitting](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/overfitting.png)

Then, comparing training and testing accuracy mean values, we could ensure whether our model is overfitted or not. We can see on the charts that **none of our models presents high overfitting**. The cross-validation techniques employed helped on that.

### Refining the Model

We tried to refine the first model by **expanding the original dataset** with 20,000 more not-hit songs, (**notebooks 8 to 12** on the execution guide). We rerun all steps with this **unbalanced dataset** to get a new second predictive model.

![Metrics](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/metrics.png)

This time, a **Random Forest** model got better metrics, in principle, than the model of the balanced dataset.

1st model - XGBoost metrics with **balanced** dataset:

```
AUC - Test Set: 95.35%
Logloss: 3.37
accuracy score: 0.902

                precision    recall  f1-score   support

         0.0        0.95      0.85      0.90      1561
         1.0        0.87      0.95      0.91      1582

    accuracy                            0.90      3143
   macro avg        0.91      0.90      0.90      3143
weighted avg        0.91      0.90      0.90      3143
```  

2nd model - Random Forest metrics with **unbalanced** dataset:

```
AUC - Test Set: 96.13%
Logloss: 3.23
accuracy score: 0.906

               precision    recall  f1-score   support

         0.0       0.95      0.93      0.94      5151
         1.0       0.78      0.83      0.80      1557

    accuracy                           0.91      6708
   macro avg       0.86      0.88      0.87      6708
weighted avg       0.91      0.91      0.91      6708
```

We found that this new model performed better when predicting negatives than the first model (which used a balanced dataset), meaning **more precision and less recall predicting negatives** (negatives f1-score up from 0.90 to 0.94). But at the same time the new model **lost a lot of predictive power on the positives** (positives f1-score dropped from 0.91 to 0.80).

As Random Forest models are more **robust to outliers**, we didn't remove them in this case.

### Cost and Optimistic/Pessimistic Metrics

If we were working for a **music company** and the **cost of failing to predict a not-hit song** was high, we would use the **second model** (RF). With it the company would may not assign promotion budget to a song with traits of not being popular. It would also be useful to **artists** willing to discard unpopular songs to send to the marketing agencies for promotion.

If we were working for a **music company** competing with others for the rights of potentially successful songs, and the **cost of not predicting a hit song** was high, or worked for an **artist** planning to send tracks with traits of being hits to music companies for publishing, then we would choose the **first model** (XGB).

# Summary

After all the previous analysis, as our final predictive model we chose the **XGBoost** model, with this characteristics:

* Removed outliers
* StandardScaler()
* OneHotEncoder(), dropping the first column
* Using all features

It performed fairly good in all metrics, did not present much overfitting, and it gives **more uniform predicting results** between positives and negatives.

![AUC](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dv_auc.png)

```
AUC - Test Set: 95.91%
Logloss: 3.31
best params:  {'classifier__colsample_bytree': 0.8, 'classifier__gamma': 1,
   'classifier__learning_rate': 0.01, 'classifier__max_depth': 5,
   'classifier__n_estimators': 1000, 'classifier__subsample': 0.8}
best score: 0.901
accuracy score: 0.904

               precision    recall  f1-score   support

         0.0       0.93      0.87      0.90      1550
         1.0       0.88      0.94      0.91      1593

    accuracy                           0.90      3143
   macro avg       0.91      0.90      0.90      3143
weighted avg       0.91      0.90      0.90      3143
```

Finally we **pickled** this XGBoost model and we used it on the Python script of the front-end web app.

# Front-end

The Github repository of the **front-end web app** of the project, that uses the **Streamlit** app framework and web hosting on **Heroku**, can be found **[here](https://github.com/daniel-isidro/heroku_hot_n_pop)**. Please visit this repository for further explanation.

### User Manual

You can play with a live demo of the web app **[here](https://hot-n-pop-song-machine.herokuapp.com)**. You just input in the text box a song name (e.g. **juice**), or an artist name followed by a song name (e.g. **harry styles watermelon sugar**), and press enter.

![web_app](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/web_app.png)

Then you get the **probability** of the song being hot and popular if it was released today, and below you can play an **audio sample** of the song and see the **cover** of the corresponding album (NOTE: some tracks do not include an audio sample due to copyright reasons).

# Conclusions

Although accurately forecasting what songs will appear on future hit lists can be difficult, as it involves many factors not related to the songs themselves (promotion budget, artist discoverability, external economic factors), the fairly good metrics we obtained on the final models in this project show that **predicting** whether a new song has the **audio traits** to potentially become a success is a task that can be accomplished thanks to Machine Learning.

Different **optimistic and pessimistic models** can be applied, depending on business needs.

**Web scraping, APIs and ML models** can be essential tools to perform studies on any industry (music business in this case).

Creating **interactive apps** as a way to deploy  ML models may be a fun and simple way for the final user to get inside knowledge on any Data Science project.

# References

[**IFPI - Annual Global Music Report 2019**](https://www.ifpi.org/ifpi-issues-annual-global-music-report/)

[**Spotify for Developers** - Get Audio Features for a Track](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)

[**Machine Learning Mastery** - Understand Your Machine Learning Data With Descriptive Statistics in Python](https://machinelearningmastery.com/understand-machine-learning-data-descriptive-statistics-python/)

[**scikit-learn** - Machine Learning in Python](https://scikit-learn.org/stable/index.html)

[**Model Evaluation Metrics**](https://www.kdnuggets.com/2020/05/model-evaluation-metrics-machine-learning.html)

# About Me

[Daniel Isidro](https://www.linkedin.com/in/daniel-isidro/)
