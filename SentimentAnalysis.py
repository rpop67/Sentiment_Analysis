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
from wordcloud import WordCloud,STOPWORDS



data=pd.read_csv("HSBC_UK_March_to_MidApril.csv")

print(data.columns)

#set of alphabets
a=ord('a')
alphabetset=[chr(i) for i in range(a,a+26)]




def FilterCovidTweets(data):
    tweetsList=data['absolute_tidy_tweets']
    # covidTweets={}
    tweetList=[]
    dateList=[]
    for i in range(len(tweetsList)):
        tweet=tweetsList[i]
        if "coronavirus" in str(tweet).lower() or "covid" in str(tweet).lower() or "covid-19" in str(tweet).lower() or "corona" in str(tweet).lower():
            tweetList.append(tweet)
            dateList.append(data['DATE'][i])
            print("found corona")
    covidDF=pd.DataFrame(list(zip(tweetsList,dateList)),columns=["TWEET","DATE"])
    return covidDF

def FilterBankTweets(data):
    tweetsList=data['absolute_tidy_tweets']
    tweetList = []
    dateList = []
    bankTerms=["bank","recession","economy","transaction","collapse","fraud","finance","financial","rate",
               "mortgage","loan","card","tax","interest","rate","credit","payment","customer","retail","amount","deposit",
               "decline","branch","hsbc","hsbc uk","debit","credit","contact","call","helpdesk"]
    for i in range(len(tweetsList)):
        tweet = tweetsList[i]
        if(any(term in str(tweet) for term in bankTerms)):
            tweetList.append(tweet)
            dateList.append(data['DATE'][i])
    bankDF = pd.DataFrame(list(zip(tweetsList, dateList)), columns=["TWEET", "DATE"])
    return bankDF


def ListToString(listName):
    listToStr = ' '.join([str(elem) for elem in listName])
    return listToStr

def StemWords(words):
    ps=PorterStemmer()
    words_after_stemming=[]
    for w in words:
        words_after_stemming.append(ps.stem(w))
    return words_after_stemming



# def CleanData(tweetsList,garbageTerms):
#
#     # appending all tweets to a string as word_tokenize accepts string as param
#     Tweets = ListToString(tweetsList)
#     words_in_tweets = word_tokenize(Tweets)
#     #removing numbers, punctautions and characters
#
#
#

#
#     # print("after removing punctuation : ",words_punc_removed)
#
#     #removing stopwords
#     stop_words=set(stopwords.words('english'))
#
#     cleanWords=[]
#
#     for w in words_punc_removed:
#         if w.lower() not in stop_words and w.lower() not in alphabetset and w.lower() not in garbageTerms and len(w)>2 :
#             cleanWords.append(w)
#
#     #return S(cleanWords)
#


    #



def CreateWordCloud(tweetWords):
    stopwords = set(STOPWORDS)

    # Create the wordcloud object
    #converting list of strings to a string

    # wordcloud = WordCloud(width=800, height=800,
    #                       background_color='white',
    #                       stopwords=stopwords,
    #                       min_font_size=10).generate(tweetWords)

    words = nltk.tokenize.word_tokenize(tweetWords)
    word_count_dict=Counter(str(word) for word in words)
    wordcloud = WordCloud(width=580, height=290, random_state=21, max_font_size=100,
                          background_color='white',stopwords=garbage).generate_from_frequencies(word_count_dict)
    plt.figure(figsize=(5.7,2.7))
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
    day_string=timestamp.day
    return day_string

def GetTweetSentiment(tweets,data):
    positiveList=[]
    negativeList=[]
    neutralList=[]
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

    df = pd.DataFrame(list(zip(dateTimeList, positiveList,negativeList,neutralList)), columns=["DateTime", "Positive","Negative","Neutral"])
    df.set_index('DateTime')


    # print(df)
    df['Date'] = df["DateTime"].apply(lambda df: pd.datetime.datetime(year=df.year, month=df.month, day=df.day))
    # df.set_index(df["Date"], inplace=True)
    # df['day']=df['TimeStamp'].apply(GiveDay)
    # days = df.groupby('day')
    # print(df)

    newDF=pd.DataFrame(list(zip(df['Date'],df['Positive'],df['Negative'],df['Neutral'])),columns=["Date","POS","NEG","NEU"])
    dailyTweets=newDF.groupby('Date').sum()
    dailyPositiveList=dailyTweets['POS']
    dailyNegativeList=dailyTweets['NEG']
    dailyNeutralList=dailyTweets['NEU']

    #taking out data for week 1. 1'st March to 7th March
    week1=newDF.groupby('Date').sum().head(7)
    negativeWeek1=week1['NEG'].sum()
    positiveWeek1=week1['POS'].sum()
    neutralWeek1=week1['NEU'].sum()


    week2=newDF.groupby('Date').sum().head(14)
    negativeWeek2 = week2['NEG'].sum()-negativeWeek1
    positiveWeek2 = week2['POS'].sum()-positiveWeek1
    neutralWeek2 = week2['NEU'].sum()-neutralWeek1

    #taking out week3 data. Cumulative(week3)-cumulative(week2)
    week3 = newDF.groupby('Date').sum().head(21)

    cumulativePositiveWeek2=positiveWeek1+positiveWeek2
    cumulativeNegativeWeek2=negativeWeek1+negativeWeek2
    cumulativeNeutralWeek2=neutralWeek1+neutralWeek2
    negativeWeek3 = week3['NEG'].sum()-negativeWeek2-negativeWeek1
    positiveWeek3 = week3['POS'].sum()-cumulativePositiveWeek2
    neutralWeek3 = week3['NEU'].sum()-cumulativeNeutralWeek2

    #taking out data for week 4

    week4 = newDF.groupby('Date').sum().head(28)

    cumulativePositiveWeek3 = cumulativePositiveWeek2 + positiveWeek3
    cumulativeNegativeWeek3=cumulativeNegativeWeek2+negativeWeek3
    cumulativeNeutralWeek3=cumulativeNeutralWeek2+neutralWeek3
    negativeWeek4 = week4['NEG'].sum()- cumulativeNegativeWeek3
    positiveWeek4 = week4['POS'].sum() - cumulativePositiveWeek3
    neutralWeek4 = week3['NEU'].sum() - cumulativeNeutralWeek3


    positiveWeekList=[positiveWeek1,positiveWeek2,positiveWeek3]
    negativeWeekList=[negativeWeek1,negativeWeek2,negativeWeek3]
    neutralWeekList=[neutralWeek1,neutralWeek2,neutralWeek3]

    # taking out data for week 5

    week5 = newDF.groupby('Date').sum().head(35)

    cumulativePositiveWeek4 = cumulativePositiveWeek3 + positiveWeek4
    cumulativeNegativeWeek4 = cumulativeNegativeWeek3 + negativeWeek4
    cumulativeNeutralWeek4 = cumulativeNeutralWeek3 + neutralWeek4
    negativeWeek5 = week5['NEG'].sum() - cumulativeNegativeWeek4
    positiveWeek5 = week5['POS'].sum() - cumulativePositiveWeek4
    neutralWeek5 = week5['NEU'].sum() - cumulativeNeutralWeek4

    # taking out data for week 6

    week6 = newDF.groupby('Date').sum().head(46)

    cumulativePositiveWeek5 = cumulativePositiveWeek4 + positiveWeek5
    cumulativeNegativeWeek5 = cumulativeNegativeWeek4 + negativeWeek5
    cumulativeNeutralWeek5 = cumulativeNeutralWeek4 + neutralWeek5
    negativeWeek6 = week6['NEG'].sum() - cumulativeNegativeWeek5
    positiveWeek6 = week6['POS'].sum() - cumulativePositiveWeek5
    neutralWeek6 = week6['NEU'].sum() - cumulativeNeutralWeek5





    #PLOT LINE -- Positive tweets (WEEK 1 vs WEEK2)
    X_Base=[1,2,3,4,5,6,7]
    plt.plot(X_Base, week1['POS'], label="week1")
    print(week2['POS'].tail(7))
    plt.plot(X_Base, week2['POS'].tail(7), label="week 2")
    plt.plot(X_Base, week3['POS'].tail(7),label="week 3")
    # plt.plot(X_Base, neutralWeekList, label="neutral")
    plt.ylabel('Positive Tweets')
    plt.xlabel("day of week")
    # show a legend on the plot
    plt.legend()
    plt.show()

    #PlotLine-- Negative Tweets (Week 1 vs Week 2)
    X_Base=[1,2,3,4,5,6,7]
    plt.plot(X_Base,week1['NEG'], label="week1")
    plt.plot(X_Base,week2['NEG'].tail(7),label="week2")
    plt.plot(X_Base,week3['NEG'].tail(7),label="week3")
    plt.ylabel("Negative Tweets")
    plt.xlabel("Day of Week")
    plt.legend()
    plt.show()

    #PlotLine --- Neutral Tweets (week 1 vs week 2)
    X_Base=[1,2,3,4,5,6,7]
    plt.plot(X_Base,week1['NEU'],label="week1")
    plt.plot(X_Base, week2['NEU'].tail(7),label="week2")
    plt.plot(X_Base,week3['NEU'].tail(7),label="week3")
    plt.ylabel("Neutral Tweets")
    plt.xlabel("Day of Week")
    plt.legend()
    plt.show()

    #DAILY PLOT for March
    X_Base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              30, 31]
    plt.plot(X_Base, dailyPositiveList[0:31], label="Positive tweets")
    plt.plot(X_Base, dailyNegativeList[0:31], label="Negative tweets")
    plt.plot(X_Base, dailyNeutralList[0:31], label="Neutral tweets")
    plt.ylabel("Tweets")
    plt.xlabel("#Day of March,2020")
    plt.legend()
    plt.show()

    # DAILY PLOT for April
    X_Base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plt.plot(X_Base, dailyPositiveList[31:46], label="Positive tweets")
    plt.plot(X_Base, dailyNegativeList[31:46], label="Negative tweets")
    plt.plot(X_Base, dailyNeutralList[31:46], label="Neutral tweets")
    plt.ylabel("Tweets")
    plt.xlabel("#Day of April,2020")
    plt.legend()
    plt.show()


    # PLOT A STACKED BAR CHART for 2 weeks
    #grey_colors = ["#263238", "#455A64","#607D8B"]
    #aqua_palatte

    totalTweets_Week1=positiveWeek1+negativeWeek1+neutralWeek1
    totalTweets_Week2=positiveWeek2+negativeWeek2+neutralWeek2
    totalTweets_Week3 = positiveWeek3 + negativeWeek3 + neutralWeek3
    totalTweets_Week4 = positiveWeek4 + negativeWeek4 + neutralWeek4
    totalTweets_Week5 = positiveWeek5 + negativeWeek5 + neutralWeek5
    totalTweets_Week6 = positiveWeek6 + negativeWeek6 + neutralWeek6

    '''
    colors = ["#00695C", "#00897B", "#26A69A"]
    data_2weeks = [["Week1", "Week1", "Week1", "Week2", "Week2", "Week2"],
            ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"],
            [positiveWeek1/totalTweets_Week1*100, negativeWeek1/totalTweets_Week1*100, neutralWeek1/totalTweets_Week1*100,
             positiveWeek2/totalTweets_Week2*100, negativeWeek2/totalTweets_Week2*100, neutralWeek2/totalTweets_Week2*100]
            ]
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['Week', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Week', columns='Tweet', values='Value')
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ['Positive', 'Negative', 'Neutral']].plot.bar(stacked=True, color=colors, figsize=(7, 7))
    plt.show()
    '''




    #PLOT A STACKED BAR CHART for 3 weeks
    colors = ["#00695C", "#00897B", "#26A69A"]
    data=[["Week1","Week1","Week1",
           "Week2","Week2","Week2",
           "Week3","Week3","Week3",
           "Week4", "Week4", "Week4",
           "Week5", "Week5", "Week5",
           "Week6", "Week6", "Week6"
           ],
          ["Positive","Negative","Neutral",
           "Positive","Negative","Neutral",
           "Positive","Negative","Neutral",
           "Positive", "Negative", "Neutral",
           "Positive", "Negative", "Neutral",
           "Positive", "Negative", "Neutral"
           ],
          [positiveWeek1/totalTweets_Week1*100,negativeWeek1/totalTweets_Week1*100,neutralWeek1/totalTweets_Week1*100,
           positiveWeek2/totalTweets_Week2*100,negativeWeek2/totalTweets_Week2*100,neutralWeek2/totalTweets_Week2*100,
           positiveWeek3/totalTweets_Week3*100,negativeWeek3/totalTweets_Week3*100,neutralWeek3/totalTweets_Week3*100,
           positiveWeek4/totalTweets_Week4*100,negativeWeek4/totalTweets_Week4*100,neutralWeek4 / totalTweets_Week4 * 100,
           positiveWeek5/totalTweets_Week5*100,negativeWeek5/totalTweets_Week5*100,neutralWeek5/totalTweets_Week5*100,
          positiveWeek6/totalTweets_Week6*100,negativeWeek6/totalTweets_Week6*100,neutralWeek6/totalTweets_Week6*100]
          ]
    rows = zip(data[0], data[1], data[2])
    headers = ['Week', 'Tweet', 'Value']
    WeekDF = pd.DataFrame(rows, columns=headers)
    pivot_df = WeekDF.pivot(index='Week', columns='Tweet', values='Value')
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df.loc[:, ['Positive', 'Negative', 'Neutral']].plot.bar(stacked=True, color=colors, figsize=(7, 7))
    plt.show()

    ##############################################
    #comparing week 1 and week 6 data
    # PLOT A STACKED BAR CHART for 2 weeks
    totalTweets_Week6 = positiveWeek6 + negativeWeek6 + neutralWeek6

    colors = ["#00695C", "#00897B", "#26A69A"]
    data_2weeks = [["Week1", "Week1", "Week1", "Week6", "Week6", "Week6"],
                   ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"],
                   [positiveWeek1 / totalTweets_Week1 * 100, negativeWeek1 / totalTweets_Week1 * 100,
                    neutralWeek1 / totalTweets_Week1 * 100,
                    positiveWeek6 / totalTweets_Week6 * 100, negativeWeek6 / totalTweets_Week6 * 100,
                    neutralWeek6 / totalTweets_Week6 * 100]
                   ]
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['Week', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Week', columns='Tweet', values='Value')
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ['Positive', 'Negative', 'Neutral']].plot.bar(stacked=True, color=colors, figsize=(7, 7))
    plt.show()







    # print(week1)
    # negativeWeek1 = week1.head(7)['Negative']
    # positiveWeek1=week1.head(7)['Positive']
    # neutralWeek1=week1.head(7)['Neutral']
    # print(negativeWeek1)
    # print(positiveWeek1)
    # print(neutralWeek1)

    #PLOT POSITIVE, NEGTAIVE AND NEUTRAL FOR DAY TO DAY ANALYSIS
    # negativeCount=days['Negative'].sum()
    # positiveCount=days['Positive'].sum()
    # neutralCount=days['Neutral'].sum()
    # Negativedict=negativeCount.to_dict()
    # Positivedict=positiveCount.to_dict()
    # Neutraldict=neutralCount.to_dict()
    # print(Negativedict)
    # print(Positivedict)
    # print(Neutraldict)
    #
    # fig=plt.figure(figsize=(15,15))
    # plt.plot(list(Positivedict.keys()), list(Positivedict.values()), label='Positive Tweets')
    # plt.plot(list(Negativedict.keys()), list(Negativedict.values()), label="Negative Tweets")
    # plt.plot(list(Neutraldict.keys()),list(Neutraldict.values()), label="Neutral Tweets")
    # plt.xlabel=('Day of the month')
    # plt.ylabel('No. of Tweets')
    # plt.xticks(list(Neutraldict.keys()),fontsize=15,rotation=90)
    # plt.title('Number of tweets on each day',fontsize=20)
    # plt.legend()
    # plt.show()

    #end of function


def CompareSentiments(covidDF,bankDF):
    covidTweets=covidDF['TWEET']
    bankTweets=bankDF['TWEET']
    print("covid tweetss \n",covidTweets)
    print("BANK TWEETS:::::\n",bankTweets)
    CovidPositiveList = []
    CovidNegativeList = []
    CovidNeutralList = []

    BankPositiveList=[]
    BankNegativeList=[]
    BankNeutralList=[]

    dateTimeListCovid = pd.to_datetime(covidDF["DATE"].values).tolist()
    dateTimeListBank=pd.to_datetime(bankDF['DATE'].values).tolist()

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
    #forCovid df
    # print("DATA ki datee ------------- \n",data['DATE'])
    # print("Covid ki datee ------------- \n", covidDF['DATE'])


    df1 = pd.DataFrame(list(zip(dateTimeListCovid, CovidPositiveList, CovidNegativeList, CovidNeutralList)),
                      columns=["DateTime", "CPositive", "CNegative", "CNeutral"])

    #df for banking related tweets
    df2 = pd.DataFrame(list(zip(dateTimeListBank, BankPositiveList, BankNegativeList, BankNeutralList)),
                       columns=["DateTime", "BPositive", "BNegative", "BNeutral"])



    df1.set_index('DateTime')
    df2.set_index('DateTime')

    # print(df)
    df1['Date'] = df1["DateTime"].apply(lambda df: pd.datetime(year=df.year, month=df.month, day=df.day))
    df2['Date'] = df2["DateTime"].apply(lambda df: pd.datetime(year=df.year, month=df.month, day=df.day))

    newDF1 = pd.DataFrame(list(zip(df1['Date'], df1['CPositive'], df1['CNegative'], df1['CNeutral'])),columns=["Date", "CovidPOS", "CovidNEG", "CovidNEU"])
    newDF2 = pd.DataFrame(list(zip(df2['Date'], df2['BPositive'], df2['BNegative'], df2['BNeutral'])),
                          columns=["Date", "BankPOS", "BankNEG", "BankNEU"])

    covidWeeklyList=[]
    bankWeeklyList=[]
    newDF1['Total']=newDF1['CovidPOS']+ newDF1['CovidNEG']+newDF1['CovidNEU']
    newDF2['Total'] = newDF2['BankPOS'] + newDF2['BankNEG'] + newDF2['BankNEU']
    print("newDF---------------c o v i d---------------------------------------\n",newDF1)
    print("newDF----------------b a n k--------------------------------------\n", newDF2)
    dailyCovid=newDF1.groupby('Date').sum()
    dailyBank=newDF2.groupby('Date').sum()
    dailyCovidList=dailyCovid['Total']
    dailyBankList=dailyBank['Total']
    print("DailyCovid---------------c o v i d---------------------------------------\n", dailyCovidList)
    print("DailyBank----------------b a n k--------------------------------------\n", dailyBankList)

    #taking out data for week 1. 1'st March to 7th March
    week1covid = newDF1.groupby('Date').sum().head(7)
    week1bank=newDF2.groupby('Date').sum().head(7)
    CovidNegativeWeek1 = week1covid['CovidNEG'].sum()
    CovidPositiveWeek1 = week1covid['CovidPOS'].sum()
    CovidNeutralWeek1 = week1covid['CovidNEU'].sum()
    BankNegativeWeek1 = week1bank['BankNEG'].sum()
    BankPositiveWeek1 = week1bank['BankPOS'].sum()
    BankNeutralWeek1 = week1bank['BankNEU'].sum()
    covidWeeklyList.append(CovidNegativeWeek1+CovidPositiveWeek1+CovidNeutralWeek1)
    bankWeeklyList.append(BankNegativeWeek1 + BankPositiveWeek1 + BankNeutralWeek1)

    print(week1covid,"\n",week1bank)

    #for Week2
    week2covid = newDF1.groupby('Date').sum().head(14)
    week2bank=newDF2.groupby('Date').sum().head(14)
    print(week2covid,"\n",week2bank)
    CovidNegativeWeek2 = week2covid['CovidNEG'].sum()-CovidNegativeWeek1
    CovidPositiveWeek2 = week2covid['CovidPOS'].sum()-CovidPositiveWeek1
    CovidNeutralWeek2 = week2covid['CovidNEU'].sum()-CovidNeutralWeek1
    BankNegativeWeek2 = week2bank['BankNEG'].sum()-BankNegativeWeek1
    BankPositiveWeek2 = week2bank['BankPOS'].sum()-BankPositiveWeek1
    BankNeutralWeek2 = week2bank['BankNEU'].sum()-BankNeutralWeek1
    covidWeeklyList.append(CovidNegativeWeek2 + CovidPositiveWeek2 + CovidNeutralWeek2)
    bankWeeklyList.append(BankNegativeWeek2 + BankPositiveWeek2 + BankNeutralWeek2)
    print("week2: ",CovidNegativeWeek2,CovidPositiveWeek2,CovidNeutralWeek2,BankNegativeWeek2,BankPositiveWeek2,BankNeutralWeek2)

    #for Week3
    CovidCumulativeNegativeWeek2=CovidNegativeWeek1+CovidNegativeWeek2
    CovidCumulativePositiveWeek2=CovidPositiveWeek1+CovidPositiveWeek2
    CovidCumulativeNeutralWeek2 = CovidNeutralWeek1 + CovidNeutralWeek2
    BankCumulativeNegativeWeek2 = BankNegativeWeek1 + BankNegativeWeek2
    BankCumulativePositiveWeek2 = BankPositiveWeek1 + BankPositiveWeek2
    BankCumulativeNeutralWeek2 = BankNeutralWeek1 + BankNeutralWeek2


    week3covid = newDF1.groupby('Date').sum().head(21)
    week3bank = newDF2.groupby('Date').sum().head(21)
    CovidNegativeWeek3 = week3covid['CovidNEG'].sum()-CovidCumulativeNegativeWeek2
    CovidPositiveWeek3 = week3covid['CovidPOS'].sum()-CovidCumulativePositiveWeek2
    CovidNeutralWeek3 = week3covid['CovidNEU'].sum()-CovidCumulativeNeutralWeek2
    BankNegativeWeek3 = week3bank['BankNEG'].sum()-BankCumulativeNegativeWeek2
    BankPositiveWeek3 = week3bank['BankPOS'].sum()-BankCumulativePositiveWeek2
    BankNeutralWeek3 = week3bank['BankNEU'].sum()-BankCumulativeNeutralWeek2

    covidWeeklyList.append(CovidNegativeWeek3 + CovidPositiveWeek3 + CovidNeutralWeek3)
    bankWeeklyList.append(BankNegativeWeek3 + BankPositiveWeek3 + BankNeutralWeek3)

    print(week3covid,"\n",week3bank)
    print("week3: ", CovidNegativeWeek3, CovidPositiveWeek3, CovidNeutralWeek3, BankNegativeWeek3, BankPositiveWeek3,
          BankNeutralWeek3)

    #for week4
    week4covid = newDF1.groupby('Date').sum().head(28)
    week4bank = newDF2.groupby('Date').sum().head(28)

    CovidCumulativeNegativeWeek3 = CovidCumulativeNegativeWeek2 + CovidNegativeWeek3
    CovidCumulativePositiveWeek3 = CovidCumulativePositiveWeek2 + CovidPositiveWeek3
    CovidCumulativeNeutralWeek3 = CovidCumulativeNeutralWeek2 + CovidNeutralWeek3
    BankCumulativeNegativeWeek3 = BankCumulativeNegativeWeek2 + BankNegativeWeek3
    BankCumulativePositiveWeek3 = BankCumulativePositiveWeek2 + BankPositiveWeek3
    BankCumulativeNeutralWeek3 = BankCumulativeNeutralWeek2 + BankNeutralWeek3
    CovidNegativeWeek4 = week4covid['CovidNEG'].sum()-CovidCumulativeNegativeWeek3
    CovidPositiveWeek4 = week4covid['CovidPOS'].sum()-CovidCumulativePositiveWeek3
    CovidNeutralWeek4 = week4covid['CovidNEU'].sum()-CovidCumulativePositiveWeek3
    BankNegativeWeek4 = week4bank['BankNEG'].sum()-BankCumulativeNegativeWeek3
    BankPositiveWeek4 = week4bank['BankPOS'].sum()-BankCumulativePositiveWeek3
    BankNeutralWeek4 = week4bank['BankNEU'].sum()-BankCumulativeNeutralWeek3

    covidWeeklyList.append(CovidNegativeWeek4 + CovidPositiveWeek4 + CovidNeutralWeek4)
    bankWeeklyList.append(BankNegativeWeek4 + BankPositiveWeek4 + BankNeutralWeek4)

    print(week4covid,"\n",week4bank)
    print("week4: ", CovidNegativeWeek4, CovidPositiveWeek4, CovidNeutralWeek4, BankNegativeWeek4, BankPositiveWeek2,
          BankNeutralWeek2)

    # for week5
    week5covid = newDF1.groupby('Date').sum().head(35)
    week5bank = newDF2.groupby('Date').sum().head(35)

    CovidCumulativeNegativeWeek4 = CovidCumulativeNegativeWeek3 + CovidNegativeWeek4
    CovidCumulativePositiveWeek4 = CovidCumulativePositiveWeek3 + CovidPositiveWeek4
    CovidCumulativeNeutralWeek4 = CovidCumulativeNeutralWeek3 + CovidNeutralWeek4
    BankCumulativeNegativeWeek4 = BankCumulativeNegativeWeek3 + BankNegativeWeek4
    BankCumulativePositiveWeek4 = BankCumulativePositiveWeek3 + BankPositiveWeek4
    BankCumulativeNeutralWeek4 = BankCumulativeNeutralWeek3 + BankNeutralWeek4
    CovidNegativeWeek5 = week5covid['CovidNEG'].sum() - CovidCumulativeNegativeWeek4
    CovidPositiveWeek5 = week5covid['CovidPOS'].sum() - CovidCumulativePositiveWeek4
    CovidNeutralWeek5 = week5covid['CovidNEU'].sum() - CovidCumulativeNeutralWeek4
    BankNegativeWeek5 = week5bank['BankNEG'].sum() - BankCumulativeNegativeWeek4
    BankPositiveWeek5 = week5bank['BankPOS'].sum() - BankCumulativePositiveWeek4
    BankNeutralWeek5 = week5bank['BankNEU'].sum() - BankCumulativeNeutralWeek4

    covidWeeklyList.append(CovidNegativeWeek5 + CovidPositiveWeek5 + CovidNeutralWeek5)
    bankWeeklyList.append(BankNegativeWeek5 + BankPositiveWeek5 + BankNeutralWeek5)

    print(week5bank,"\n",week5covid)
    print("week5: ", CovidNegativeWeek5, CovidPositiveWeek5, CovidNeutralWeek5, BankNegativeWeek5, BankPositiveWeek5,
          BankNeutralWeek5)

    # for week6
    week6covid = newDF1.groupby('Date').sum().head(46)
    week6bank = newDF2.groupby('Date').sum().head(46)

    CovidCumulativeNegativeWeek5 = CovidCumulativeNegativeWeek4 + CovidNegativeWeek5
    CovidCumulativePositiveWeek5 = CovidCumulativePositiveWeek4 + CovidPositiveWeek5
    CovidCumulativeNeutralWeek5 = CovidCumulativeNeutralWeek4 + CovidNeutralWeek5
    BankCumulativeNegativeWeek5 = BankCumulativeNegativeWeek4 + BankNegativeWeek5
    BankCumulativePositiveWeek5 = BankCumulativePositiveWeek4 + BankPositiveWeek5
    BankCumulativeNeutralWeek5 = BankCumulativeNeutralWeek4 + BankNeutralWeek5
    CovidNegativeWeek6 = week6covid['CovidNEG'].sum() - CovidCumulativeNegativeWeek5
    CovidPositiveWeek6 = week6covid['CovidPOS'].sum() - CovidCumulativePositiveWeek5
    CovidNeutralWeek6 = week6covid['CovidNEU'].sum() - CovidCumulativeNeutralWeek5
    BankNegativeWeek6 = week6bank['BankNEG'].sum() - BankCumulativeNegativeWeek5
    BankPositiveWeek6 = week6bank['BankPOS'].sum() - BankCumulativePositiveWeek5
    BankNeutralWeek6 = week6bank['BankNEU'].sum() - BankCumulativeNeutralWeek5

    covidWeeklyList.append(CovidNegativeWeek6 + CovidPositiveWeek6 + CovidNeutralWeek6)
    bankWeeklyList.append(BankNegativeWeek6 + BankPositiveWeek6 + BankNeutralWeek6)

    print(week6covid,"\n",week6bank)
    print("week6: ", CovidNegativeWeek6, CovidPositiveWeek6, CovidNeutralWeek6, BankNegativeWeek6, BankPositiveWeek6,
          BankNeutralWeek6)






    #2- week comparision of covid and banking tweets
    # PLOT A STACKED BAR CHART for  week1 ------COVID vs BANKING
    # totalCovidTweets_Week1 = CovidPositiveWeek1 + CovidNegativeWeek1 + CovidNeutralWeek1
    # totalBankTweets_Week1 = BankPositiveWeek1 + BankNeutralWeek1 + BankNegativeWeek1
    #
    # colors = ["#00695C", "#00897B", "#26A69A"]
    # data_2weeks = [
    #     ["Covid-19(week1)", "Covid-19(week1)", "Covid-19(week1)", "Banking(week1)", "Banking(week1)", "Banking(week1)"],
    #     ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"],
    #     [CovidPositiveWeek1 / totalCovidTweets_Week1 * 100, CovidNegativeWeek1 / totalCovidTweets_Week1 * 100,
    #      CovidNeutralWeek1 / totalCovidTweets_Week1 * 100,
    #      BankPositiveWeek1 / totalBankTweets_Week1 * 100, BankNegativeWeek1 / totalBankTweets_Week1 * 100,
    #      BankNeutralWeek1 / totalBankTweets_Week1 * 100]
    #     ]
    # print("data 2 weeks::: ", data_2weeks)
    # rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    # headers_2Weeks = ['Covid19 vs Banking Tweets', 'Tweet', 'Value']
    # WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    # pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Covid19 vs Banking Tweets', columns='Tweet', values='Value')
    # # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    # pivot_df_2Weeks.loc[:, ['Positive', 'Negative', 'Neutral']].plot.bar(stacked=True, color=colors, figsize=(5, 7))
    # plt.show()

    #Total Tweets -----------WEEKLY
    totalWeek1 = covidWeeklyList[0] + bankWeeklyList[0]
    totalWeek2 = covidWeeklyList[1] + bankWeeklyList[1]
    totalWeek3 = covidWeeklyList[2] + bankWeeklyList[2]
    totalWeek4 = covidWeeklyList[3] + bankWeeklyList[3]
    totalWeek5 = covidWeeklyList[4] + bankWeeklyList[4]
    totalWeek6 = covidWeeklyList[5] + bankWeeklyList[5]

    #totalTweets categorised


    totalCovidTweets_Week1 = CovidPositiveWeek1 + CovidNegativeWeek1 + CovidNeutralWeek1
    totalBankTweets_Week1 = BankPositiveWeek1 + BankNeutralWeek1 + BankNegativeWeek1

    totalCovidTweets_Week2 = CovidPositiveWeek2 + CovidNegativeWeek2 + CovidNeutralWeek2
    totalBankTweets_Week2 = BankPositiveWeek2 + BankNeutralWeek2 + BankNegativeWeek2

    totalCovidTweets_Week3 = CovidPositiveWeek3 + CovidNegativeWeek3 + CovidNeutralWeek3
    totalBankTweets_Week3 = BankPositiveWeek3 + BankNeutralWeek3 + BankNegativeWeek3

    totalCovidTweets_Week4 = CovidPositiveWeek4 + CovidNegativeWeek4 + CovidNeutralWeek4
    totalBankTweets_Week4 = BankPositiveWeek4 + BankNeutralWeek4 + BankNegativeWeek4

    totalCovidTweets_Week5 = CovidPositiveWeek5 + CovidNegativeWeek5 + CovidNeutralWeek5
    totalBankTweets_Week5 = BankPositiveWeek5 + BankNeutralWeek5 + BankNegativeWeek5

    totalCovidTweets_Week6 = CovidPositiveWeek6 + CovidNegativeWeek6 + CovidNeutralWeek6
    totalBankTweets_Week6 = BankPositiveWeek6 + BankNeutralWeek6 + BankNegativeWeek6

    print("total weekly: \n", totalWeek1,totalWeek2,totalWeek3,totalWeek4,totalWeek5,totalWeek6)
    print("covid weekly: \n",totalCovidTweets_Week1,totalCovidTweets_Week2,totalCovidTweets_Week3,totalCovidTweets_Week4,totalCovidTweets_Week5,totalCovidTweets_Week6)
    print("bank weekly: \n", totalBankTweets_Week1, totalBankTweets_Week2,totalBankTweets_Week3, totalBankTweets_Week4,totalBankTweets_Week5, totalBankTweets_Week6)




    #barplot
    import numpy as np
    import matplotlib.pyplot as plt
    data=[[ 227,195,591,724,463,408],
          [0,5,18,21,14,4],
          [227,190,573,703,449,404]
          ]
    # data = [[totalWeek1,totalWeek2,totalWeek3,totalWeek4,totalWeek5,totalWeek6]
    #         [totalCovidTweets_Week1, totalCovidTweets_Week2, totalCovidTweets_Week3,totalCovidTweets_Week4,totalCovidTweets_Week5,totalCovidTweets_Week6],
    #         [totalBankTweets_Week1, totalBankTweets_Week2,totalBankTweets_Week3,totalBankTweets_Week4,totalBankTweets_Week5,totalBankTweets_Week6]]
    X = np.arange(6)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X + 0.00, data[0], color='b', width=0.25)
    ax.bar(X + 0.25, data[1], color='g', width=0.25)
    ax.bar(X + 0.50, data[2], color='r', width=0.25)
    plt.show()

    # PLOT A STACKED BAR CHART for  week1 ------COVID vs BANKING

    colors = ["#00695C", "#00897B"]
    data_2weeks = [["Week 3","Week 3","Week 5","Week 5"],
                   ["Covid-19", "Bank", "Covid-19", "Bank",],
                   [totalCovidTweets_Week3/ totalWeek3 * 100, totalBankTweets_Week3 / totalWeek3 * 100,
                    totalCovidTweets_Week5/ totalWeek5 * 100, totalBankTweets_Week5 / totalWeek5 * 100,
                    ]
                   ]
    print("data 2 weeks::: ",data_2weeks)
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['Week3 vs Week5', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Week3 vs Week5', columns='Tweet', values='Value')
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ["Covid-19", "Bank",]].plot.bar(stacked=True, color=colors, figsize=(5, 7))
    plt.show()

    # PLOT A STACKED BAR CHART for week 6----------------- COVID vs BANKING


    colors = ["#00695C", "#00897B", "#26A69A"]
    data_2weeks = [
        ["Covid-19(week6)", "Covid-19(week6)", "Covid-19(week6)", "Banking(week6)", "Banking(week6)", "Banking(week6)"],
        ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"],
        [CovidPositiveWeek6 / totalCovidTweets_Week6 * 100, CovidNegativeWeek6 / totalCovidTweets_Week6 * 100,
         CovidNeutralWeek6 / totalCovidTweets_Week6 * 100,
         BankPositiveWeek6 / totalBankTweets_Week6 * 100, BankNegativeWeek6 / totalBankTweets_Week6* 100,
         BankNeutralWeek6 / totalBankTweets_Week6 * 100]
        ]
    print("data 2 weeks::: ", data_2weeks)
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['Covid19 vs Banking Tweets', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Covid19 vs Banking Tweets', columns='Tweet', values='Value')
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ['Positive', 'Negative', 'Neutral']].plot.bar(stacked=True, color=colors, figsize=(5, 7))
    plt.show()


    #PLOT COVID VS BANKING TWEETS
    # PlotLine-- Negative Tweets (Week 1 vs Week 2)
    X_Base = [1, 2, 3, 4, 5, 6]
    plt.plot(X_Base, covidWeeklyList, label="Covid tweets")
    plt.plot(X_Base, bankWeeklyList, label="Banking tweets")
    plt.ylabel("Tweets")
    plt.xlabel("#Week")
    plt.legend()
    plt.show()

    # PLOT COVID VS BANKING TWEETS for each day --March
    X_Base = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    plt.plot(X_Base, dailyCovidList[0:31], label="Covid tweets")
    plt.plot(X_Base, dailyBankList[0:31], label="Banking tweets")
    plt.ylabel("Tweets")
    plt.xlabel("#Day of March,2020")
    plt.legend()
    plt.show()

    # PLOT COVID VS BANKING TWEETS for each day--April
    X_Base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    plt.plot(X_Base, dailyCovidList[31:46], label="Covid tweets")
    plt.plot(X_Base, dailyBankList[31:46], label="Banking tweets")
    plt.ylabel("Tweets")
    plt.xlabel("#Day of April,2020")
    plt.legend()
    plt.show()

    # PLOT A STACKED BAR CHART for  all weeks ------Total COVID vs BANKING


    print(covidWeeklyList)
    print(bankWeeklyList)

    colors = ["#00695C", "#26A69A"]
    data_2weeks = [
        ["Week 1", "Week 1",
         "Week 2", "Week 2",
         "Week 3", "Week 3",
         "Week 4", "Week 4",
         "Week 5", "Week 5",
         "Week 6", "Week 6"],
        ["Covid-19 Tweets", "Banking Tweets","Covid-19 Tweets", "Banking Tweets","Covid-19 Tweets", "Banking Tweets",
        "Covid-19 Tweets", "Banking Tweets","Covid-19 Tweets", "Banking Tweets","Covid-19 Tweets", "Banking Tweets"],
        [covidWeeklyList[0] / totalWeek1 * 100, bankWeeklyList[0] / totalWeek1 * 100,
         covidWeeklyList[1] / totalWeek2 * 100, bankWeeklyList[1] / totalWeek2 * 100,
         covidWeeklyList[2] / totalWeek3 * 100, bankWeeklyList[2] / totalWeek3 * 100,
         covidWeeklyList[3] / totalWeek4 * 100, bankWeeklyList[3] / totalWeek4 * 100,
         covidWeeklyList[4] / totalWeek5 * 100, bankWeeklyList[4] / totalWeek5 * 100,
         covidWeeklyList[5] / totalWeek6 * 100, bankWeeklyList[5] / totalWeek6 * 100]
        ]
    print("data 2 weeks::: ", data_2weeks)
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['covid19 vs banking tweets', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='covid19 vs banking tweets', columns='Tweet', values='Value')
    pivot_df_2Weeks.fillna(0)
    print(pivot_df_2Weeks)
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ['Covid-19 Tweets','Banking Tweets']].plot.bar(stacked=True, color=colors, figsize=(5, 7))
    plt.show()


    #Weekly covid sentiment analysis:
    colors = ["#00695C", "#00897B", "#26A69A"]
    totalCovidTweets_Week2=CovidNegativeWeek2+CovidNeutralWeek2+CovidPositiveWeek2
    totalCovidTweets_Week3 = CovidNegativeWeek3 + CovidNeutralWeek3 + CovidPositiveWeek3
    totalCovidTweets_Week4 = CovidNegativeWeek4 + CovidNeutralWeek4 + CovidPositiveWeek4
    totalCovidTweets_Week5 = CovidNegativeWeek5 + CovidNeutralWeek5 + CovidPositiveWeek5
    data_2weeks = [
        ["Week 1", "Week 1","Week 1",
         "Week 2", "Week 2","Week 2",
         "Week 3", "Week 3","Week 3",
         "Week 4", "Week 4","Week 4",
         "Week 5", "Week 5","Week 5",
         "Week 6", "Week 6","Week 6"],
        ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral",
         "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral",
         "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"],
        [CovidPositiveWeek1/totalCovidTweets_Week1*100,CovidNegativeWeek1/totalCovidTweets_Week1*100,CovidNeutralWeek1/totalCovidTweets_Week1*100,
         CovidPositiveWeek2/totalCovidTweets_Week2*100,CovidNegativeWeek2/totalCovidTweets_Week2*100,CovidNeutralWeek2/totalCovidTweets_Week2*100,
         CovidPositiveWeek3 / totalCovidTweets_Week3 * 100, CovidNegativeWeek3 / totalCovidTweets_Week3 * 100,
         CovidNeutralWeek3 / totalCovidTweets_Week3 * 100,
         CovidPositiveWeek4 / totalCovidTweets_Week4 * 100, CovidNegativeWeek4 / totalCovidTweets_Week4 * 100,
         CovidNeutralWeek4 / totalCovidTweets_Week4 * 100,
         CovidPositiveWeek5 / totalCovidTweets_Week5 * 100, CovidNegativeWeek5 / totalCovidTweets_Week5 * 100,
         CovidNeutralWeek5 / totalCovidTweets_Week5 * 100,
         CovidPositiveWeek6 / totalCovidTweets_Week6 * 100, CovidNegativeWeek6 / totalCovidTweets_Week6 * 100,
         CovidNeutralWeek6 / totalCovidTweets_Week6 * 100,
         ]
    ]
    print("data covid weekly::: ", data_2weeks)
    rows_2Weeks = zip(data_2weeks[0], data_2weeks[1], data_2weeks[2])
    headers_2Weeks = ['Polarity', 'Tweet', 'Value']
    WeekDF_2Weeks = pd.DataFrame(rows_2Weeks, columns=headers_2Weeks)
    pivot_df_2Weeks = WeekDF_2Weeks.pivot(index='Polarity', columns='Tweet', values='Value')
    pivot_df_2Weeks.fillna(0)
    print(pivot_df_2Weeks)
    # Note: .loc[:,['Positive','Negative', 'Neutral']] is used here to rearrange the layer ordering
    pivot_df_2Weeks.loc[:, ['Positive', 'Negative','Neutral']].plot.bar(stacked=True, color=colors, figsize=(5, 7))
    plt.show()




    #endFunction



def VisualiseTweets(tweetList):
    tweetWords = ""
    allTokens=""
    stopwords=STOPWORDS
    garbageTerms2 = ['', "http", "want", "need", "' ", '’ ', "us", "hi", "hey", "find", "due", "look","set","thats","sure","hsbc",
                     "https", "open","hello", "404", "nt", "able", '.', 'hi','i','isa', "get", "know", '.','dm','via',
                     "http", "want", "need", "' ", '’ ', "us", "please","nt", "able",'nick',"covid", "covid-19",'the','if','yuriy','tijianne','name',
                     "get", "know", "coronavirus", "virus",'ank','im','22','see','alison','give','mill','see','via','sam','full','03457',"thank"]
    for val in tweetList:

        # typecaste each val to string
        val = str(val)
        if val != 'tweetnotfound':
            # split the value
            tokens = val.split()
            tweetText=""
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
                if tokens[i] not in garbageTerms2 and tokens[i] not in stopwords:
                    allTokens+=tokens[i].lower()+" "
                else :
                    tokens[i]=''

            tweetWords += "".join(str(tokens)) + ""
    text_list = allTokens.split(" ")


    print("\n\nTWEETWORDS - - --  \n", tweetWords)
    freq = nltk.FreqDist(text_list)
    print(freq)
    top_freq=freq.most_common(15)
    print(top_freq)
    freq.plot(30,cumulative=False)

    plt.show()

    CreateWordCloud(allTokens)



# print(df)

#print(word_count_dict)

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
#remove links
cleaned_tweets = []

for index, row in data.iterrows():
    # Here we are filtering out all the words that contains link
    words_without_links = [word for word in row.tidy_tweets.split()        if 'http' not in word]
    cleaned_tweets.append(' '.join(words_without_links))

data['tidy_tweets'] = cleaned_tweets

#deleting duplicate rows
data.drop_duplicates(subset=['tidy_tweets'], keep=False)

# removing punctuations
punctuations=string.punctuation
garbage=["'",'[',']',".",",",'<','!','&','(',')']
# data['absolute_tidy_tweets'] = data['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
data['absolute_tidy_tweets'] = data['tidy_tweets'].apply(lambda x:''.join([i for i in x if i not in punctuations and i not in garbage]))


# # #removing punctuation
#     punctuations=string.punctuation
#     # #not removing hyphen; incase considering time period
#     punctuations.replace("-",'')
#     pattern= r"[{}]".format(punctuations)
#     words_punc_removed=[]
#     for w in words_in_tweets:
#         words_punc_removed.append(re.sub(pattern,"",w))

#removing stopwords is,am,are
stopwordsSet = set(stopwords.words("english"))
garbageTerms2 = ['', "http", "want", "need", "' ", '’ ', "us", "hi", "hey", "find", "need", "due", "look",
                    "including", "https", "open", "times", "hello", "404", "nt", "able", '.','hi',"get", "know",'.', "http", "want", "need", "' ", '’ ', "us", "please", "hi", "hey", "find", "need", "due", "look",
                    "including", "https", "open", "apply", "times", "hours", "time", "hello", "404", "nt", "able","covid","covid-19",'thats','0800',
                    "get", "know","coronavirus","virus","oh",'yuriy','tijianne','name','well',"set","thats"]

cleaned_tweets = []

for index, row in data.iterrows():
    # filerting out all the stopwords
    words_without_stopword1 = [word for word in row.absolute_tidy_tweets.split() if not word in stopwordsSet]
    words_without_stopwords=[word for word in words_without_stopword1 if not word in garbageTerms2]

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType
    cleaned_tweets.append(' '.join(words_without_stopwords))

data['absolute_tidy_tweets'] = cleaned_tweets
#tokenization-diving sentences into tokens and lemmatization- stopping to stop

from nltk.stem import WordNetLemmatizer

# Tokenization
tokenized_tweet = data['absolute_tidy_tweets'].apply(lambda x: x.split())
# Finding Lemma for each word
word_lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
#joining words into sentences (from where they came from)
for i, tokens in enumerate(tokenized_tweet):
    tokenized_tweet[i] = ' '.join(tokens)

# data.to_csv("Fulltweets.csv")
data['absolute_tidy_tweets'] = tokenized_tweet
print(data)
tweetsList=data['absolute_tidy_tweets']
print("/n - - - - printing absolute tidy tweets: /n /n  ",tweetsList)



###########################################

covidDF=FilterCovidTweets(data)
print(covidDF)
bankDF=FilterBankTweets(data)
# GetTweetSentiment(tweetsList,data)
index = 0
counter = 0
covidTweetsFull=[]
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
bankTweetsFull=[]
sizeBankDF = len(bankDF['DATE'])
for i in range(len(data['DATE'])):
    if index < sizeBankDF and data['DATE'][i] == bankDF['DATE'][index]:
        # print("date is: ", data['DATE'][i], "  and matched: ", bankDF['DATE'][index], " and index is : ", index)
        bankTweetsFull.append(bankDF['TWEET'][index])
        index += 1
    else:
        bankTweetsFull.append("tweetnotfound")
    counter += 1




newcovidDF=pd.DataFrame(list(zip(covidTweetsFull,data['DATE'])),columns=['TWEET','DATE'])
newbankDF=pd.DataFrame(list(zip(bankTweetsFull,data['DATE'])),columns=['TWEET','DATE'])

# CompareSentiments(newcovidDF,newbankDF)
# count_bank_tweets=len(bankTweets)
# count_covid_tweets=len(covidTweets)
# print("#bankTweets: ",count_bank_tweets,"   #covidTweets: ",count_covid_tweets)
VisualiseTweets(newcovidDF['TWEET'])
VisualiseTweets(newbankDF['TWEET'])
CreateWordCloud(newcovidDF['TWEET'])
CreateWordCloud(newbankDF['TWEET'])

          
