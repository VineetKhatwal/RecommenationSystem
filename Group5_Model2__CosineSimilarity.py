'''
=================================================================================================================================================================


                                                                BOOK RECOMMENDER SYSTEM

                                                                                                                                            Aayushi Gupta
                                                                                                                                            Girish Chhabra
                                                                                                                                            Vineet Khatwal
=================================================================================================================================================================

'''


'''
=============================================================================================
                                IMPORTING THE REQUIRED FUNCTIONS
=============================================================================================
'''
import sys
import re
import csv
import smtplib
import ssl
import stdiomask
import os
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pyspark import SparkConf, SparkContext
from math import sqrt
csv.field_size_limit(10000000)
'''
=============================================================================================
                                 CONVERTING TSV FILE TO CSV
=============================================================================================
'''
'''
tsv_file = 'amazon_reviews_us_Books_v1_02.tsv'
csv_table = pd.read_table(tsv_file, sep='\t')
csv_table.to_csv('File_Amazon.csv', index=False)
'''

'''
=============================================================================================
                                    USER DEFINED FUNCTIONS
=============================================================================================
'''

# Function to create a dictionary of Book IDs and Book names


def loadBookNames():
    BookNames = {}
    '''
    with open("File_Amazon.csv", encoding='ascii', errors='ignore') as f:
        next(f)
        for line in f:
            fields = line.strip('"')
            fields = line.strip('"\"').split(',')
            print(fields[3],fields[5])
            BookNames[fields[3]] = fields[5]
    return BookNames

    '''
    # with open("amazon_reviews_us_Books_v1_02.tsv", encoding='ascii', errors='ignore') as fd:
    with open("File_Amazon.tsv", encoding='ascii', errors='ignore') as fd:
        f = csv.reader(fd, delimiter="\t", quotechar='"')
        next(f)
        for line in f:
            # if line[3].isdigit():
            #   line[3] = line[3].lstrip("0")
            # print(line[3],line[5])
            BookNames[line[3]] = line[5]
    print("==================================================================")
    print("                        BOOKNAMES")
    print("==================================================================")
    print(BookNames)
    return BookNames


# Function to filter duplicates that are resulted after self-join operation
def filterDuplicates(userRatings):
    ratings = userRatings[1]
    (book1, rating1) = ratings[0]
    (book2, rating2) = ratings[1]
    return book1 < book2


# Function to explicitly extract the ratings as key value RDD where Key - Book pair and value is rating pair
def makePairs(userRatings):
    ratings = userRatings[1]
    (book1, rating1) = ratings[0]
    (book2, rating2) = ratings[1]
    return ((book1, book2), (rating1, rating2))

# Function to compute cosine similarity metric for item based collaborative filtering


def computeCosineSimilarity(ratingPairs):
    #print("COSINE SIMILARITY")
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        #print(ratingX, ratingY, sum_xx, sum_yy, sum_xy)
        numPairs += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = 0
    #print("NUMERATOR", numerator, denominator)
    if (denominator):
        score = (numerator / (float(denominator)))

    ##print("SCORE", score, numPairs)
    return (score, numPairs)


def sendMail(bookRead, recommendationList):
    sender_email = "cmpe.256.recommendation@gmail.com"
    #receiver_email = "vineet.khatwal@sjsu.edu", "aayushi.gupta@sjsu.edu", "girish.chhabra@sjsu.edu"
    receiver_email = "vineet.khatwal@sjsu.edu"
    print("Type your password and press enter:")
    password = stdiomask.getpass(mask='X')
    message = MIMEMultipart("alternative")
    message["Subject"] = "Hi Vineet ! Recommendation for you from CMPE 256 : Team 5"
    message["From"] = "cmpe.256.recommendation@gmail.com"
    message["To"] = ", ".join(receiver_email)

    print("================== Composing the mail ================== ")
    # Create the plain-text and HTML version of your message
    text = """\
    Hi Reader,
    How are you?
    Add more books to your reading bucket list.
    We are glad you liked the book : """ + bookRead + """
    We think you will also like:
    """ + '\n' + '\n'.join(["- " + item for item in recommendationList]) + "\n\n" + """
    Enjoy your Reading time,
    CMPE 256 Team 5
    """

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )

    print("==================   Mail sent ================== ")


'''
=============================================================================================
                                    CONFIGURING THE CORES
=============================================================================================
'''

# configuation to use all the cores of the computer and run as seperate executor using Spark's built-in cluster manager
conf = SparkConf().setMaster("local[*]").setAppName("BookSimilarities")
sc = SparkContext(conf=conf)

print("\nLoading Book names...")
nameDict = sc.broadcast(loadBookNames())
print("Name Dict ==== ", nameDict)
print("Type Dict ==== ", type(nameDict))


bookID_2 = sys.argv[2]
#bookID_2 = bookID_2.lstrip("0")
print("BOOK ID ==================================", bookID_2)
print("BOOK ID ==================================", nameDict.value[bookID_2])
'''
bookID1 = sys.argv[2]
#temp = bookID1.lstrip("0")
#print("Top 10 similar books for " + nameDict.value[temp])
'''
#print("Name Dict ==== ",nameDict.value['345348036'])
# print(type(bookID))


print("\nLoading Book data...")

#ratingsDatawheader = sc.textFile("amazon_reviews_us_Books_v1_02.tsv")
ratingsDatawheader = sc.textFile("File_Amazon.tsv")

# ---------------------------------------------------------------------------------------------------------------
#                                           DATA PREPROCESSING
# ---------------------------------------------------------------------------------------------------------------
ratingsDatawheader = ratingsDatawheader.map(lambda s: s.replace('"', ""))
print("Ratings Data ===================================",
      ratingsDatawheader.take(1))
ratingsDatawheader = ratingsDatawheader.map(lambda s: s.replace(',', ""))
print("Ratings Data ===================================",
      ratingsDatawheader.take(1))
ratingsDatawheader = ratingsDatawheader.map(lambda s: s.replace('\t', ","))
print("Ratings Data ===================================",
      ratingsDatawheader.take(1))
ratingsDatawheader = ratingsDatawheader.map(lambda s: s.replace('"\\"', ""))
print("Ratings Data ===================================",
      ratingsDatawheader.take(1))

header = ratingsDatawheader.first()  # to exclude header row
# print(type(header))

print("Ratings Data with Header===================================",
      ratingsDatawheader.take(3))

ratingsData = ratingsDatawheader.filter(
    lambda line: line != header).map(lambda line: line.strip('"'))
print("Ratings Data without header ===============================",
      ratingsData.take(3))
print(type(ratingsDatawheader))
print(type(ratingsData))


# Map ratings data to key-value pairs: user ID =>bookID, rating
ratings = ratingsData.map(lambda l: l.split(',')).map(
    lambda l: (l[1].strip('"'), (l[3].strip('"'), float(l[7].strip('"')))))
print("Ratings ===================================", ratings.take(3))
print("")

# find every pair of book rated by the same user using self-join (to find every combination).
joinedRatings = ratings.join(ratings)


# Filter out duplicate  pairs resulted from self join ( from the rdd format: userID => ((bookID, rating), (bookID, rating))
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
print("Unique Ratings ===================================",
      uniqueJoinedRatings.take(3))
print("")


# Now key by book pair and strip out the user information.
bookPairs = uniqueJoinedRatings.map(makePairs)

# We now have (book1, book2) => (rating1, rating2)
# Grouping by the book pair for all the ratings across the data set
bookPairRatings = bookPairs.groupByKey()
print("BookPairRatings First ===============================",
      bookPairRatings.take(3))
print("")
# Compute similarities on the RDD ((book1, book2) = > (rating1, rating2), (rating1, rating2) format)
bookPairSimilarities = bookPairRatings.mapValues(
    computeCosineSimilarity).cache()
print("BookPairRatings ===============================")
print(bookPairSimilarities.take(10))
print("")


# Extract similarities for the book we care about that are "good".

if (len(sys.argv) > 1):
    # if (False):
    recommendationList = []
    scoreThreshold = 0.50
    #coOccurenceThreshold = 1
    bookID = sys.argv[2]
    #bookID = bookID.lstrip("0")

    # Filter for Books with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = bookPairSimilarities.filter(lambda pairSim: (pairSim[0][0] == bookID or pairSim[0][1] == bookID)
                                                  and pairSim[1][0] > scoreThreshold
                                                  )
    print("BOOK ID ===============================", bookID)
    print("BookPairRatings ===============================")
    print(bookPairSimilarities.take(10))

    print("FilteredResults ===============================",
          filteredResults.take(3))

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (
        pairSim[1], pairSim[0])).sortByKey(ascending=False).take(10)
    print("RESULTS ===============================", results)
    print()
    print()
    print("=============================================================================================================")
    #print("Top 10 similar books for " + nameDict.value[bookID[1]])
    print("Top 10 similar books for " + nameDict.value[bookID])
    print("=============================================================================================================")
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the book we're looking at
        similarBookID = pair[0]
        if (similarBookID == bookID):
            similarBookID = pair[1]
        '''
        if similarBookID.isdigit():
           similarBookID = similarBookID.lstrip("0")
        '''
        #print("------------------------------------------------------------------------------------------------------------------")
        #print(nameDict.value[similarBookID[1]] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
        print(similarBookID + "-----" + nameDict.value[similarBookID])
        recommendationList.append(nameDict.value[similarBookID])
    print("============================================================================================================")
    sendMail(nameDict.value[bookID], recommendationList)
