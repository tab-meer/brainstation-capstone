# MoodSync: A Mood-Based Spotify Song Recommendation System

## Project Overview  
### Problem Area
How can we use machine learning to improve music recommendations for users to quickly create personalized playlists based on their mood? While services like Spotify allow manual playlist creation, there is a lack of intelligent automated features. Spotify has experimented with an AI feature called 'DJ', which learns users’ music preferences and when activated, plays a curated lineup of songs in real-time, mimicking a live DJ experience. There is an opportunity to enhance music recommendation by integrating this feature into playlist creation, refining 'DJ' to generate playlists based on user inputs such as mood, offering a more innovative and personalized music experience.

This project addresses issues faced by Spotify users, specifically those seeking more efficient and personalized playlist creation. Potential beneficiaries encompass music enthusiasts, tech-savvy users, and busy casual listeners seeking variety in their day-to-day.

### Proposed Solution
Envision a system that categorizes songs by moods using audio features, building on Spotify's DJ feature to refine user preferences through mood classification. Users would input a mood, and the feature would instantly create a unique curated playlist. This ensures diverse music experiences even for the same mood, with the option to save playlists for later.

### The Impact
Enhancing user experience on platforms like Spotify meets the widespread demand for efficient and personalized playlist creation, impacting millions of users. This time-saving feature appeals to users seeking instant and tailored music recommendations, boosting user retention and loyalty. It encourages diverse music exploration and positions Spotify as a technological innovator through AI and machine learning, strengthening investor relations. The data-driven insights from user preferences would help inform strategic decisions in content acquisition, marketing, and feature development.

## Walkthrough Demo

...
...
...

## Project Flowchart

...
...
...

## Project Organization

...
...
...

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible Google Drive folder)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - joblib dump of final model / model object

* `notebooks`
    - contains all final notebooks involved in the project

* `reports`
    - contains final report which summarises the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `capstine_env.yml`
    - Conda environment specification

* `Makefile`
    - Automation script for the project

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

## Dataset

The dataset for the project will be scraped from the [Spotify Web API](https://developer.spotify.com/documentation/web-api). Specifically:
1. A subset of the [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) will be used to source tracks
2. Audio features for the tracks will be fetched using the Spotify API's [get-audio-features endpoint](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)

The process above is time consuming and requires scripting (the API is also rate limited), hence it is out of scope currently and will incorporated in a future deliverable if required. For Sprint 1, I will use data that was obtained through the process above for a [similar project](https://github.com/enjuichang/PracticalDataScience-ENCA/blob/main/data/allsong_data.csv) online. Since the structure of the data is the same what would be the result of scraping the data myself, the file can be swapped out and the same notebook can be used to explore the data.

### Data Dictionary

The data dictionary below was built using the Spotify Web API [specification](https://developer.spotify.com/documentation/web-api) as a reference
 * `acousticness` _number [float]_
   * A confidence measure from 0.0 to 1.0 of whether the track is acoustic
   * 1.0 represents high confidence the track is acoustic
   * Range: `0 - 1`
   * Example: `0.00242`
 * `artist_name` _string_
   * The name of the artist
   * Example: `Beyoncé`
 * `artist_pop` _integer_
   * The popularity of the artist
   * The value will be between 0 and 100, with 100 being the most popular
   * The artist's popularity is calculated from the popularity of all the artist's tracks
   * Range: `0 - 100`
   * Example: `21`
 * `danceability` _number [float]_
   * Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity
   * A value of 0.0 is least danceable and 1.0 is most danceable
   * Example: `0.585`
 * `energy` _number [float]_
   * Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity
   * Typically, energetic tracks feel fast, loud, and noisy
   * For example, death metal has high energy, while a Bach prelude scores low on the scale
   * Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy
   * Example: `0.842`
 * `genres` _array of strings_
   * A list of the genres the artist is associated with
   * If not yet classified, the array is empty
   * Example: `["Prog rock", "Grunge"]`
 * `id` _string_
   * The Spotify ID for the track (base-62 identifier)
   * Example: `6rqhFgbbKwnb9MLmUQDhG6`
 * `instrumentalness` _number [float]_
   * Predicts whether a track contains no vocals
   * "Ooh" and "aah" sounds are treated as instrumental in this context
   * Rap or spoken word tracks are clearly "vocal"
   * The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
   * Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0
   * Example: `0.00686`
 * `key` _integer_
   * The key the track is in
   * Integers map to pitches using standard [Pitch Class notation](https://en.wikipedia.org/wiki/Pitch_class)
   * E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on
   * If no key was detected, the value is -1
   * Range: `-1 - 11`
   * Example: `9`
 * `liveness` _number [float]_
   * Detects the presence of an audience in the recording
   * Higher liveness values represent an increased probability that the track was performed live
   * A value above 0.8 provides strong likelihood that the track is live
   * Example: `0.0866`
 * `loudness` _number [float]_
   * The overall loudness of a track in decibels (dB)
   * Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks
   * Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude)
   * Values typically range between -60 and 0 db
   * Example: `-5.883`
 * `mode` _integer_
   * Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived
   * Major is represented by 1 and minor is 0
   * Example: `0`
 * `speechiness` _number [float]_
   * Speechiness detects the presence of spoken words in a track
   * The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value
   * Values above 0.66 describe tracks that are probably made entirely of spoken words
   * Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music
   * Values below 0.33 most likely represent music and other non-speech-like tracks
   * Example: `0.0556`
 * `tempo` _number [float]_
   * The overall estimated tempo of a track in beats per minute (BPM)
   * In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
   * Example: `118.211`
 * `track_name` _string_
   * The name of the track
   * Example: `Crazy In Love`
 * `track_pop` _integer_
   * The popularity of the track
   * The popularity of a track is a value between 0 and 100, with 100 being the most popular
   * The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are
   * Generally speaking, songs that are being played a lot at the time the data was scraped will have a higher popularity than songs that were played a lot in the past
   * Duplicate tracks (e.g. the same track from a single and an album) are rated independently
   * Range: `0 - 100`
   * Example: `32`
 * `valence`_number [float]_
   * A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track
   * Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
   * Range: `0 - 1`
   * Example: `0.428`

## Credits & References

...
...
...

--------
