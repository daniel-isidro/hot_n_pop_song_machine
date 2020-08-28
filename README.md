![logo](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/hnp_logo.jpeg)

Repository for the **Hot'n'Pop Song Machine** project, a Machine Learning song popularity predictor,

The Github repository of the **front-end web app** of the project, that uses the Streamlit app framework and web hosting on Heroku, can be found **[here]( https://github.com/daniel-isidro/heroku_hot_n_pop)**.

You can play with a live demo of the web app **[here](https://hot-n-pop-song-machine.herokuapp.com)**.

# Contents

[Introduction](#introduction) <br>
[Requirements](#requirements) <br>
[Execution Guide](#execution-Guide) <br>
[Raw Data Description](#raw-data-description) <br>
[Methodology](#methodology) <br>
[Data Acquisition](#data-acquisition) <br>
[Data Preparation](#data-preparation) <br>
[Data Exploration](#data-exploration) <br>
[Analysis](#analysis) <br>
[Summary](#summary) <br>
[Conclusions](#conclusions) <br>
[Front-end](#front-end) <br>
[References](#references) <br>
[About Me](#about-me)

# Introduction

What
Why
Why is it relevant
Any previous related work/state of the art

# Requirements

Anaconda virtual environment with Python 3.7.7 or higher and the following libraries/packages:

### Anaconda Python packages

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
### Spotify account

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

For replicating the project, please execute the following **Jupyter notebooks** in the specified order.

1. **[Web scraping](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/web_scraping/ultimate_music_database_web_scraping.ipynb)**

Getting Billboard 100 US weekly hit songs and artist names from 1962 till 2020 from Ultimate Music Database website.

2. **[Get audio features from hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_hit_songs.ipynb)**

Getting audio features from those hit songs, restricted to years 2000-2020, from Spotify web API.

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

### Refining the model

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

# Raw Data Description



# Methodology

ML techniques, statistical methodologies

# Data Acquisition

### Web Scraping

![Billboard](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/billboard.png)

For getting all the **Billboard 100** weekly hit songs and artist names in the United States, from 1962 till 2020, we perform **web scraping** on the [Ultimate Music Database](http://umdmusic.com/default.asp?Lang=English&Chart=D) website.

We use **BeautifulSoup4** as our Python library tool for scraping the web.

The result is a data frame with three columns: **year, artist, and title**. Then we save the data frame into a CSV file.

We do **several scraping passes** on the website, covering just one or two decades, to avoid being kicked by the website.

At the end we merge all data frames into one final CSV file, that contains **all hit titles** from **1962** until **late June 2020**.

### Spotify Web API

**Hit Songs**

Now we take the resulting data frame on the previous step, **remove all songs older than 2000** (as older hit songs may not predict future hits since people's preferences change over time), remove duplicates and clean artist and title names with regular expressions (to get better search results).

Then we use **spotipy** Python library to call the **Spotify Web API** and get the audio features of those hit songs.

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

The result is a data frame with around **15,700 entries**.

![Data Prep](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/dataprep.png)

We store the result into a CSV file.

# Data Exploration



# Analysis



# Summary

Main results

# Conclusions

Not a summary of the work. The problem was relevant, now with your work, what can you say about how the problem is solved?

# Front-end

The Github repository of the **front-end web app** of the project, that uses the **Streamlit** app framework and web hosting on **Heroku**, can be found **[here](https://github.com/daniel-isidro/heroku_hot_n_pop)**. Please visit this repository for further explanation.

### User Manual

You can play with a live demo of the web app **[here](https://hot-n-pop-song-machine.herokuapp.com)**. You just input in the text box a song name (e.g. **juice**), or an artist name followed by a song name (e.g. **harry styles watermelon sugar**), and press enter.

![web_app](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/web_app.png)

Then you get the **probability** of the song being hot and popular if it was released today, and below you can play an **audio sample** of the song and see the **cover** of the corresponding album (NOTE: some tracks do not include an audio sample due to copyright reasons).

# References



# About Me
