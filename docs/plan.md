# Overview
The goal of this library is to use quantitative data-driven analysis to predict outcomes on sports matches. We will start with NFL.

## Data
- use nflreadpy for historical game and team granularity data
- use sharpapi for current game and team granularity data
- data from nflreadpy should be saved in a local cache and updated periodically
- data from sharpapi should be updated periodically as well, and stored alongside nflreadypy data
- pull as much data as possible from nflreadypy (all available columns)
- pull the same columns from sharpapi (where overlap is available)
- data should be stored with a timestamp
- data access will include as_of: get_data(start_date, end_date, as_of_date). meaning, this should retrieve the latest data available as of that date.
- if the as_of date is before our earliest available download, just use the earliest available download (you can print a warning)
- data access should automatically stitch and align the nflreadypy history and sharpapi most recent observations. it should also ensure that the latest nflreadypy and sharpapi observations are not overlapping.

## Model
- a model should take in a frame of data as well as some instructions on what to use for train/test/validation.
- models should be "pluggable". meaning, we start with regression, then move onto trees on XGBoost, with maybe neutral networks in the future. let's try to make the interface of every model consistent such that we can swap data, models, efficacy approaches (further down) easily.
- every model should output a training result, which consists of all data necessary to evaluate the model: weights, rediduals, etc etc. 
- models should support ensembling of models

## Efficacy
- given a fitted model perform analysis on quality of predictions made by model
- need to pay close attention to the data the model is trained on vs data used for efficacy testing

## Backtest
- take a fitted model and use the weights it generated to place bets
- should also take in a sizing strategy - start with equal sized bets, then we can allow sizing based on odds, in the future can also size based on confidence of prediction
- runs in time series 

