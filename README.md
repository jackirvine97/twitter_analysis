# twitter_analysis
Playground for interacting with the Twitter API. Please note the branch `topic-modelling` is the WIP `main` branch.

## Context

This repository hosts scripts for querying the Twitter API and text analysis of tweets. At present it is in an exploratory phase of development and as such it is a collection of sample scripts. As use cases are refined this may evolve.

## Strategy

Pull data into a standard data model. Build a series of analytic capabilities to interface with this data model. 

## Installation

These scripts are best run in a conda virtual environment (conda best installed via [miniconda](https://docs.conda.io/en/latest/miniconda.html)):

```
conda create --name env_name
```

To activate your environment run:

```
conda activate env_name
```

Finally install the requirements for querying Twitter:

```
conda install --yes --file --requirements.txt
```

## Useful Guides

Introduction and cloud bot deploy: https://realpython.com/twitter-bot-python-tweepy/  
Sentiment Analysis: https://medium.com/@r.ratan/tweepy-textblob-and-sentiment-analysis-python-47cc613a4e51
