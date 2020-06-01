import re
import string

import nltk
import numpy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from textblob import TextBlob
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

data = pd.read_csv("HSBC_UK_ALL_TWEETS.csv", encoding="ISO-8859-1", engine='python')

print(data.columns)

# set of alphabets
a = ord('a')
alphabetset = [chr(i) for i in range(a, a + 26)]


def FilterCovidTweets(data):
    tweetsList = data['absolute_tidy_tweets']
    # covidTweets={}
    tweetList = []
    dateList = []
    for i in range(len(tweetsList)):
        tweet = tweetsList[i]
        if "coronavirus" in str(tweet).lower() or "covid" in str(tweet).lower() or "covid-19" in str(
                tweet).lower() or "corona" in str(tweet).lower():
            tweetList.append(tweet)
            dateList.append(data['DATE'][i])
            print("found corona covid-19 ... at ", tweet)
    covidDF = pd.DataFrame(list(zip(tweetsList, dateList)), columns=["TWEET", "DATE"])
    return covidDF


def FilterBankTweets(data):
    tweetsList = data['absolute_tidy_tweets']
    tweetList = []
    dateList = []
    bankTerms = ["bank", "recession", "economy", "transaction", "collapse", "fraud", "finance", "financial", "rate",
                 "mortgage", "loan", "card", "tax", "interest", "rate", "credit", "payment", "customer", "retail",
                 "amount", "deposit",
                 "decline", "branch", "hsbc", "hsbc uk", "debit", "credit", "contact", "call","hsbcuk","one","still","app", "helpdesk"]
    for i in range(len(tweetsList)):
        tweet = tweetsList[i]
        if (any(term in str(tweet) for term in bankTerms)):
            tweetList.append(tweet)
            dateList.append(data['DATE'][i])
    bankDF = pd.DataFrame(list(zip(tweetsList, dateList)), columns=["TWEET", "DATE"])
    return bankDF


def ListToString(listName):
    listToStr = ' '.join([str(elem) for elem in listName])
    return listToStr


def StemWords(words):
    ps = PorterStemmer()
    words_after_stemming = []
    for w in words:
        words_after_stemming.append(ps.stem(w))
    return words_after_stemming


def CreateWordCloud(tweetWords):
    stopwords = set(STOPWORDS)

    # Create the wordcloud object
    # converting list of strings to a string

    # wordcloud = WordCloud(width=800, height=800,
    #                       background_color='white',
    #                       stopwords=stopwords,
    #                       min_font_size=10).generate(tweetWords)

    words = nltk.tokenize.word_tokenize(tweetWords)
    word_count_dict = Counter(str(word) for word in words)
    wordcloud = WordCloud(width=580, height=290, random_state=21, max_font_size=100,
                          background_color='white', stopwords=garbage).generate_from_frequencies(word_count_dict)
    plt.figure(figsize=(5.7, 2.7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    # wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

    # Display the generated image:
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.margins(x=0, y=0)
    # plt.show()


def GiveDay(timestamp):
    day_string = timestamp.day
    return day_string


def GetTweetSentiment(tweets, data):
    positiveList = []
    negativeList = []
    neutralList = []
    dateTimeList = pd.to_datetime(data["DATE"].values).tolist()
    # print(dateTimeList)

    for tweet in tweets:
        analysis = TextBlob(str(tweet))
        if analysis.sentiment.polarity > 0:
            positiveList.append(1)
            negativeList.append(0)
            neutralList.append(0)
        elif analysis.sentiment.polarity < 0:
            negativeList.append(1)
            positiveList.append(0)
            neutralList.append(0)
        else:
            neutralList.append(1)
            positiveList.append(0)
            negativeList.append(0)

    # #Polarity_List stores polarity of tweets preserving the order

    df = pd.DataFrame(list(zip(dateTimeList, positiveList, negativeList, neutralList)),
                      columns=["DateTime", "Positive", "Negative", "Neutral"])
    df.set_index('DateTime')

    # print(df)
    df['Date'] = df["DateTime"].apply(lambda df: pd.datetime.datetime(year=df.year, month=df.month, day=df.day))

    newDF = pd.DataFrame(list(zip(df['Date'], df['Positive'], df['Negative'], df['Neutral'])),
                         columns=["Date", "POS", "NEG", "NEU"])
    DailyTweets = newDF.set_index('Date').resample('D').sum()
    MonthlyTweets = newDF.set_index('Date').resample('M')["POS", "NEG", "NEU"].sum()
    YearlyTweets = newDF.set_index('Date').resample('Y')["POS", "NEG", "NEU"].sum()
    print(YearlyTweets)
    print(MonthlyTweets)
    print(DailyTweets)

    # Visualization of Tweet Sentiments:


def CompareSentiments(covidDF, bankDF):
    covidTweets = covidDF['TWEET']
    bankTweets = bankDF['TWEET']
    print("covid tweetss \n", covidTweets)
    print("BANK TWEETS:::::\n", bankTweets)
    CovidPositiveList = []
    CovidNegativeList = []
    CovidNeutralList = []

    BankPositiveList = []
    BankNegativeList = []
    BankNeutralList = []

    dateTimeListCovid = pd.to_datetime(covidDF["DATE"].values).tolist()
    dateTimeListBank = pd.to_datetime(bankDF['DATE'].values).tolist()

    # print(dateTimeListCovid)
    # print(dateTimeListBank)
    # print(dateTimeList)

    for tweet in covidTweets:
        if tweet != "tweetnotfound":
            analysis = TextBlob(str(tweet))
            if analysis.sentiment.polarity > 0:
                CovidPositiveList.append(1)
                CovidNegativeList.append(0)
                CovidNeutralList.append(0)
            elif analysis.sentiment.polarity < 0:
                CovidNegativeList.append(1)
                CovidPositiveList.append(0)
                CovidNeutralList.append(0)
            else:
                CovidNeutralList.append(1)
                CovidPositiveList.append(0)
                CovidNegativeList.append(0)
        else:
            CovidPositiveList.append(0)
            CovidNegativeList.append(0)
            CovidNeutralList.append(0)

    for tweet in bankTweets:
        if tweet != "tweetnotfound":
            analysis = TextBlob(str(tweet))
            if analysis.sentiment.polarity > 0:
                BankPositiveList.append(1)
                BankNegativeList.append(0)
                BankNeutralList.append(0)
            elif analysis.sentiment.polarity < 0:
                BankNegativeList.append(1)
                BankPositiveList.append(0)
                BankNeutralList.append(0)
            else:
                BankNeutralList.append(1)
                BankPositiveList.append(0)
                BankNegativeList.append(0)
        else:
            BankPositiveList.append(0)
            BankNegativeList.append(0)
            BankNeutralList.append(0)

    # #Polarity_List stores polarity of tweets preserving the order
    # forCovid df
    # print("DATA ki datee ------------- \n",data['DATE'])
    # print("Covid ki datee ------------- \n", covidDF['DATE'])

    df1 = pd.DataFrame(list(zip(dateTimeListCovid, CovidPositiveList, CovidNegativeList, CovidNeutralList)),
                       columns=["DateTime", "CPositive", "CNegative", "CNeutral"])

    # df for banking related tweets
    df2 = pd.DataFrame(list(zip(dateTimeListBank, BankPositiveList, BankNegativeList, BankNeutralList)),
                       columns=["DateTime", "BPositive", "BNegative", "BNeutral"])

    df1.set_index('DateTime')
    df2.set_index('DateTime')

    # print(df)
    df1['Date'] = df1["DateTime"].apply(lambda df: pd.datetime(year=df.year, month=df.month, day=df.day))
    df2['Date'] = df2["DateTime"].apply(lambda df: pd.datetime(year=df.year, month=df.month, day=df.day))

    print("DF---------------c o v i d---------------------------------------\n", df1)
    print("DF----------------b a n k--------------------------------------\n", df2)

    newDF1 = pd.DataFrame(list(zip(df1['Date'], df1['CPositive'], df1['CNegative'], df1['CNeutral'])),
                          columns=["Date", "CovidPOS", "CovidNEG", "CovidNEU"])
    newDF2 = pd.DataFrame(list(zip(df2['Date'], df2['BPositive'], df2['BNegative'], df2['BNeutral'])),
                          columns=["Date", "BankPOS", "BankNEG", "BankNEU"])

    print("newDF---------------c o v i d---------------------------------------\n", newDF1)
    print("newDF----------------b a n k--------------------------------------\n", newDF2)

    newDF1['Total'] = newDF1['CovidPOS'] + newDF1['CovidNEG'] + newDF1['CovidNEU']
    newDF2['Total'] = newDF2['BankPOS'] + newDF2['BankNEG'] + newDF2['BankNEU']

    print("newDF---------------c o v i d---------------------------------------\n", newDF1)
    print("newDF----------------b a n k--------------------------------------\n", newDF2)

    # dailyCovid = newDF1.groupby('Date').sum()
    # dailyBank = newDF2.groupby('Date').sum()

    dailyCovidList = newDF1.set_index('Date').resample('D')["CovidPOS", "CovidNEG", "CovidNEU", "Total"].sum()
    dailyBankList = newDF2.set_index('Date').resample('D')["BankPOS", "BankNEG", "BankNEU", "Total"].sum()

    print("DailyCovid---------------c o v i d---------------------------------------\n", dailyCovidList)
    print("DailyBank----------------b a n k--------------------------------------\n", dailyBankList)

    # taking out data for week 1. 1'st March to 7th March

    covidMonthlyList=dailyCovidList.groupby([ pd.Grouper(freq='M')]).sum()
    bankMonthlyList = dailyBankList.groupby([pd.Grouper(freq='M')]).sum()

    # covidMonthlyList = dailyCovidList.set_index('Date').resample('M')["CovidPOS", "CovidNEG", "CovidNEU", "Total"].sum()
    # bankMonthlyList = dailyCovidList.set_index('Date').resample('M')["BankPOS", "BankNEG", "BankNEU", "Total"].sum()
    print("Monthly Covid---------------c o v i d---------------------------------------\n", covidMonthlyList)
    print("Monthly Bank----------------b a n k--------------------------------------\n", bankMonthlyList)

    # totalTweets categorised based on month PLOT

    totalMonthlyList = []
    for i in range(len(covidMonthlyList)):
        print("total covid monthly: ",covidMonthlyList['Total'][i] ,"total bank month;y: ", bankMonthlyList['Total'][i])
        totalMonthlyList.append(covidMonthlyList['Total'][i] + bankMonthlyList['Total'][i])

    colors = ["#00695C", "#00897B"]
    data_2weeks = [["Febraury", "Febraury", "March", "March", "April", "April", "May", "May"],
                   ["Covid-19", "Bank", "Covid-19", "Bank", "Covid-19", "Bank", "Covid-19", "Bank"],
                   [round(covidMonthlyList['Total'][0] / totalMonthlyList[0] * 100),round( bankMonthlyList['Total'][0] / totalMonthlyList[0] * 100),
                    round(covidMonthlyList['Total'][1] / totalMonthlyList[1] * 100), round(bankMonthlyList['Total'][1] / totalMonthlyList[1] * 100),
                    round(covidMonthlyList['Total'][2] / totalMonthlyList[2] * 100), round(bankMonthlyList['Total'][2] / totalMonthlyList[2] * 100),
                    round(covidMonthlyList['Total'][3] / totalMonthlyList[3] * 100), round(bankMonthlyList['Total'][3] / totalMonthlyList[3] * 100),
                    ]
                   ]
    print("data 2 weeks::: ", data_2weeks)
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['Monthly Comparison', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Monthly Comparison', columns='Tweet', values='Value')
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ["Covid-19", "Bank", ]].plot.bar(stacked=True, color=colors, figsize=(5, 7))
    plt.show()

    # PLOT A STACKED BAR CHART for  week1 ------COVID vs BANKING

    # PLOT COVID VS BANKING TWEETS
    # PlotLine-- Negative Tweets (Week 1 vs Week 2)
    X_Base = [1, 2, 3, 4]
    plt.plot(X_Base, covidMonthlyList['Total'], label="Covid tweets")
    plt.plot(X_Base, bankMonthlyList['Total'], label="Banking tweets")
    plt.ylabel("Tweets")
    plt.xlabel("#Week")
    plt.legend()
    plt.show()

    # PLOT A STACKED BAR CHART for  all weeks ------Total COVID vs BANKING

    # endFunction


def VisualiseTweets(tweetList):
    tweetWords = ""
    allTokens = ""
    stopwords = STOPWORDS
    garbageTerms2 = ['', "http", "want", "need", "' ", '’ ', "us", "hi", "hey", "find", "due", "look", "set", "thats",
                     "sure", "hsbc", 'good', 'dont',
                     "https", "open", "hello", "404", "nt", "able", '.', 'hi', 'i', 'isa', "get", "know", '.', 'dm',
                     'via', 'iâ\x80\x99ve', 'got',
                     "http", "want", "need", "' ", '’ ', "us", "please", "nt", "able", 'nick', "covid", "covid-19",
                     'the', 'if', 'yuriy', 'tijianne', 'name',
                     "get", "know", "coronavirus", "virus", 'ank', 'im', '22', 'see', 'alison', 'give', 'mill', 'see',
                     'via', 'sam', 'full', '03457', "thank", "hsbcuk", "one", "still", "app", "day", "go", "hear",
                     "week", "take", "make", "use", "click", "even", "new", "cent"]
    for val in tweetList:

        # typecaste each val to string
        val = str(val)
        if val != 'tweetnotfound':
            # split the value
            tokens = val.split()
            tweetText = ""
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
                if tokens[i] not in garbageTerms2 and tokens[i] not in stopwords:
                    allTokens += tokens[i].lower() + " "
                else:
                    tokens[i] = ''

            tweetWords += "".join(str(tokens)) + ""
    text_list = allTokens.split(" ")

    print("\n\nTWEETWORDS - - --  \n", tweetWords)
    freq = nltk.FreqDist(text_list)
    print(freq)
    top_freq = freq.most_common(30)
    print(top_freq)
    freq.plot(30, cumulative=False)

    plt.show()

    CreateWordCloud(allTokens)


# print(df)

# print(word_count_dict)

# word_count_head=Counter(data for data in clean_data.head())

#######################################START ###################

import re


# removing @names
def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, str(text))
    for i in r:
        text = re.sub(i, '', str(text))

    return str(text)


import numpy as np

# We are keeping cleaned tweets in a new column called 'tidy_tweets' for removing names who tweeted
data['tidy_tweets'] = np.vectorize(remove_pattern)(data['TWEET'], "@[\w]*: | *RT*")
# data['tidy_tweets']=data['TWEET']
# remove links
cleaned_tweets = []

for index, row in data.iterrows():
    # Here we are filtering out all the words that contains link
    words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
    cleaned_tweets.append(' '.join(words_without_links))

data['tidy_tweets'] = cleaned_tweets

# deleting duplicate rows
data.drop_duplicates(subset=['tidy_tweets'], keep=False)

# removing punctuations
punctuations = string.punctuation
garbage = ["'", '[', ']', ".", ",", '<', '!', '&', '(', ')']
# data['absolute_tidy_tweets'] = data['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
data['absolute_tidy_tweets'] = data['tidy_tweets'].apply(
    lambda x: ''.join([i for i in x if i not in punctuations and i not in garbage]))

# # #removing punctuation
#     punctuations=string.punctuation
#     # #not removing hyphen; incase considering time period
#     punctuations.replace("-",'')
#     pattern= r"[{}]".format(punctuations)
#     words_punc_removed=[]
#     for w in words_in_tweets:
#         words_punc_removed.append(re.sub(pattern,"",w))

# removing stopwords is,am,are
stopwordsSet = set(stopwords.words("english"))
garbageTerms2 = ['', "http", "want", "need", "' ", '’ ', "us", "hi", "hey", "find", "need", "due", "look",
                 "including", "https", "open", "times", "hello", "404", "nt", "able", '.', 'hi', "get", "know", '.',
                 "http", "want", "need", "' ", '’ ', "us", "please", "hi", "hey", "find", "need", "due", "look",
                 "including", "https", "open", "apply", "times", "hours", "time", "hello", "404", "nt", "able", "covid",
                 "covid-19", 'thats', '0800',
                 "get", "know", "coronavirus", "virus", "oh", 'yuriy', 'tijianne', 'name', 'well', "set", "thats"]

cleaned_tweets = []

for index, row in data.iterrows():
    # filerting out all the stopwords
    words_without_stopword1 = [word for word in row.absolute_tidy_tweets.split() if not word in stopwordsSet]
    words_without_stopwords = [word for word in words_without_stopword1 if not word in garbageTerms2]

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType
    cleaned_tweets.append(' '.join(words_without_stopwords))

data['absolute_tidy_tweets'] = cleaned_tweets
# tokenization-diving sentences into tokens and lemmatization- stopping to stop

from nltk.stem import WordNetLemmatizer

# Tokenization
tokenized_tweet = data['absolute_tidy_tweets'].apply(lambda x: x.split())
# Finding Lemma for each word
word_lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
# joining words into sentences (from where they came from)
for i, tokens in enumerate(tokenized_tweet):
    tokenized_tweet[i] = ' '.join(tokens)

# data.to_csv("Fulltweets.csv")
data['absolute_tidy_tweets'] = tokenized_tweet
print(data)
tweetsList = data['absolute_tidy_tweets']
print("/n - - - - printing absolute tidy tweets: /n /n  ", tweetsList)

###########################################

covidDF = FilterCovidTweets(data)
print(covidDF)
bankDF = FilterBankTweets(data)
GetTweetSentiment(tweetsList, data)
index = 0
counter = 0
covidTweetsFull = []
sizeCovidDF = len(covidDF['DATE'])
for i in range(len(data['DATE'])):
    if index < sizeCovidDF and data['DATE'][i] == covidDF['DATE'][index]:
        # print("date is: ", data['DATE'][i], "  and matched: ", covidDF['DATE'][index], " and index is : ", index)
        covidTweetsFull.append(covidDF['TWEET'][index])
        index += 1
    else:
        covidTweetsFull.append("tweetnotfound")
    counter += 1

index = 0
counter = 0
bankTweetsFull = []
sizeBankDF = len(bankDF['DATE'])
for i in range(len(data['DATE'])):
    if index < sizeBankDF and data['DATE'][i] == bankDF['DATE'][index]:
        # print("date is: ", data['DATE'][i], "  and matched: ", bankDF['DATE'][index], " and index is : ", index)
        bankTweetsFull.append(bankDF['TWEET'][index])
        index += 1
    else:
        bankTweetsFull.append("tweetnotfound")
    counter += 1

newcovidDF = pd.DataFrame(list(zip(covidTweetsFull, data['DATE'])), columns=['TWEET', 'DATE'])
newbankDF = pd.DataFrame(list(zip(bankTweetsFull, data['DATE'])), columns=['TWEET', 'DATE'])

CompareSentiments(newcovidDF, newbankDF)
# count_bank_tweets=len(bankTweets)
# count_covid_tweets=len(covidTweets)
# print("#bankTweets: ",count_bank_tweets,"   #covidTweets: ",count_covid_tweets)
# VisualiseTweets(newcovidDF['TWEET'])
# VisualiseTweets(newbankDF['TWEET'])
# CreateWordCloud(newcovidDF['TWEET'])
# CreateWordCloud(newbankDF['TWEET'])
