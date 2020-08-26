![logo](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/hnp_logo.jpeg)

Repository for the **Hot'n'Pop Song Machine** project, a Machine Learning song popularity predictor,

The Github repository of the **front-end web app** of the project, that uses the Streamlit app framework and web hosting on Heroku, can be found at https://github.com/daniel-isidro/heroku_hot_n_pop.

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

### Spotify account

You'll need a Spotify account (free or paid) to be able to use their web API, and then register your project as an app. For that, follow the instructions found on the ['Spotify for Developers' guide](https://developer.spotify.com/documentation/general/guides/app-settings/):

1. On [your Dashboard](https://developer.spotify.com/dashboard/) click CREATE A CLIENT ID.
2. Enter Application Name and Application Description and then click CREATE. Your application is registered, and the app view opens.
3. On the app view, click Edit Settings to view and update your app settings. Edit Settings View

<img src="https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/spotifydashboard.png" width="300">

**Note:** Find your **Client ID** and **Client Secret**; you need them in the authentication phase.

* **Client ID** is the unique identifier of your application.
* **Client Secret** is the key that you pass in secure calls to the Spotify Accounts and Web API services. Always store the client secret key securely; never reveal it publicly! If you suspect that the secret key has been compromised, regenerate it immediately by clicking the link on the edit settings view.

### 'settings.env' file

In order to not upload your Spotify Client ID and Client Secret tokens to Github, you can create a .env text file and place it into your local Github folder (e.g. in the spotify_api folder). Create a .gitignore file at the root folder of your Github local repository so the .env file is not uploaded to the remore repository. The content of the .env text file should look like this:

```
  {
    "SPOTIPY_CLIENT_ID": "754b47a409f902c6kfnfk89964bf9f91",
    "SPOTIPY_CLIENT_SECRET": "6v9657a368e14d7vdnff8c647fc5c5523b08"
  }
```


# Execution Guide

For replicating the project, please execute the following Jupyter notebooks in the specified order.

1. [Web scraping](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/web_scraping/ultimate_music_database_web_scraping.ipynb)
2. [Get audio features from hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_hit_songs.ipynb)
3. [Get audio features from random not-hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_not_hit_songs.ipynb)
4. [Data preparation](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_prep/data_prep.ipynb)
5. [Data exploration](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_exploration/feature_selection_and_data_visualization.ipynb)
6. [ML model selection](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/modeling.ipynb)
7. [Prediction](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/model_predict.ipynb)

If you also want to replicate the second part of the project, where we explore using an unbalanced dataset and retrain the model, please execute the following Jupyter notebooks in the specified order.

8. [Get more random not-hit songs](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/spotify_api/get_audio_features_more_not_hit_songs.ipynb)
9. [Data preparation (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_prep/data_prep_expanded_dataset.ipynb)
10. [Data exploration (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/data_exploration/feature_selection_and_data_visualization_expanded_dataset.ipynb)
11. [ML model selection (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/modeling_expanded_dataset.ipynb)
12. [Prediction (unbalanced dataset)](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/modeling/model_predict_expanded_dataset.ipynb)


# Raw Data Description



# Methodology

ML techniques, statistical methodologies

# Data Acquisition



# Data Preparation



# Data Exploration



# Analysis



# Summary

Main results

# Conclusions

Not a summary of the work. The problem was relevant, now with your work, what can you say about how the problem is solved?

# Front-end

The Github repository of the **front-end web app** of the project, that uses the Streamlit app framework and web hosting on Heroku, can be found at https://github.com/daniel-isidro/heroku_hot_n_pop. Please visit this repository for further explanation.

### User Manual

You can play with a live demo of the web app **[here](https://hot-n-pop-song-machine.herokuapp.com)**. You just input in the text box a song name (e.g. **juice**), or an artist name followed by a song name (e.g. **harry styles watermelon sugar**), and press enter.

![web_app](https://github.com/daniel-isidro/hot_n_pop_song_machine/blob/master/media/web_app.png)

Then you get the probability of the song being hot and popular if it was released today, and below you can play an audio sample of the song and see the cover of the corresponding album (NOTE: not all tracks include an audio sample due to copyright).

# References



# About Me
