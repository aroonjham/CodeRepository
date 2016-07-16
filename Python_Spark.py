# go to https://spark.apache.org/docs/1.1.1/api/python/pyspark.rdd.RDD-class.html for understanding all Spark transformations and actions
# go to http://spark.apache.org/docs/latest/programming-guide.html for additional data


###############################################################
###															###
###															###
###   					BASICS OF AN RDD					###
###															###
###															###
###############################################################

# creating an RDD:
wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)

# getting an RDD from a txt file
import os.path
fileName = "dbfs:/" + os.path.join('databricks-datasets', 'cs100', 'lab1', 'data-001', 'shakespeare.txt')

shakespeareRDD = sc.textFile(fileName, 8).map(removePunctuation) # the key take away here is that we use the SparkContext.textFile() method. optionally, you may apply map to the data extract
print '\n'.join(shakespeareRDD
                .zipWithIndex()  # to (line, lineNum)
                .map(lambda (l, num): '{0}: {1}'.format(num, l))  # to 'lineNum: line'
                .take(15))


# Applying a function to an RDD
def myfunction:
	'''some function'''

newRDD = myRDD.map(myfunction)

# groupByKey() 
'''
the groupByKey() TRANSFORMATION groups all the elements of the RDD with the same key into a single list in one of the partitions
groupByKey should be avoided because
1 - The operation requires a lot of data movement to move all the values into the appropriate partitions
2 - Large lists can exhaust memory
'''
wordPairs = [('cat', 1), ('elephant', 1), ('rat', 1), ('rat', 1), ('cat', 1)]
wordsGrouped = wordPairs.groupByKey()
for key, value in wordsGrouped.collect():
    print '{0}: {1}'.format(key, list(value))
''' the above for loop returns
rat: [1, 1]
elephant: [1]
cat: [1, 1]
'''

# reduceByKey()
'''
The reduceByKey() TRANSFORMATION gathers together pairs that have the same key and applies the function provided to two values at a time
, iteratively reducing all of the values to a single value
reduceByKey() operates by applying the function first within each partition on a per-key basis and then across the partitions, allowing it to scale efficiently to large datasets.
'''
wordPairs = [('cat', 1), ('elephant', 1), ('rat', 1), ('rat', 1), ('cat', 1)]
wordCounts = wordPairs.reduceByKey(lambda x,y: x+y) # Note that reduceByKey takes in a function that accepts two values and returns a single value

# reduce()
# In the example below, we use a reduce() action to sum the counts in wordCounts
wordCounts = [('rat', 2), ('elephant', 1), ('cat', 2)]
from operator import add
totalCount = (wordCounts
              .map(lambda (x,y): y)
              .reduce(lambda x,y: x+y)) # reduce takes in a list of values and not a key value pair like reduceByKey()
			  
# filter()
# the following example exaplains filter TRANSFORMATION

shakeWordsRDD = shakespeareWordsRDD.filter(lambda x: x<>'') # <> not equal to
shakeWordCount = shakeWordsRDD.count()

#####################################################

import re
import datetime

from pyspark.sql import Row

month_map = {'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12}

def parse_apache_time(s):
    """ Convert Apache time format into a Python datetime object
    Args:
        s (str): date and time in Apache time format
    Returns:
        datetime: datetime object (ignore timezone for now)
    """
    return datetime.datetime(int(s[7:11]),
                             month_map[s[3:6]],
                             int(s[0:2]),
                             int(s[12:14]),
                             int(s[15:17]),
                             int(s[18:20]))

#####

def parseApacheLogLine(logline):
    """ Parse a line in the Apache Common Log format
    Args:
        logline (str): a line of text in the Apache Common Log format
    Returns:
        tuple: either a dictionary containing the parts of the Apache Access Log and 1,
               or the original invalid log line and 0
    """
    match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
    if match is None:
        return (logline, 0)
    size_field = match.group(9)
    if size_field == '-':
        size = long(0)
    else:
        size = long(match.group(9))
    return (Row(
        host          = match.group(1),
        client_identd = match.group(2),
        user_id       = match.group(3),
        date_time     = parse_apache_time(match.group(4)),
        method        = match.group(5),
        endpoint      = match.group(6),
        protocol      = match.group(7),
        response_code = int(match.group(8)),
        content_size  = size
    ), 1)
	
#####

APACHE_ACCESS_LOG_PATTERN = '(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)\s?" (\d{3}) (\S+)'

#####

import sys
import os
from test_helper import Test

baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab2', 'apache.access.log.PROJECT')
logFile = os.path.join(baseDir, inputPath) # when you print logfile it will return data/cs100/lab2/apache.access.log.PROJECT

def parseLogs():
    """ Read and parse log file """
    parsed_logs = (sc
                   .textFile(logFile)
                   .map(parseApacheLogLine)
                   .cache())

    access_logs = (parsed_logs
                   .filter(lambda s: s[1] == 1)
                   .map(lambda s: s[0])
                   .cache())

    failed_logs = (parsed_logs
                   .filter(lambda s: s[1] == 0)
                   .map(lambda s: s[0]))
    failed_logs_count = failed_logs.count()
    if failed_logs_count > 0:
        print 'Number of invalid logline: %d' % failed_logs.count()
        for line in failed_logs.take(20):
            print 'Invalid logline: %s' % line

    print 'Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (parsed_logs.count(), access_logs.count(), failed_logs.count())
    return parsed_logs, access_logs, failed_logs


parsed_logs, access_logs, failed_logs = parseLogs()

#####

# Calculate statistics based on the content size.
content_sizes = access_logs.map(lambda log: log.content_size).cache()
print 'Content Size Avg: %i, Min: %i, Max: %s' % (
    content_sizes.reduce(lambda a, b : a + b) / content_sizes.count(),
    content_sizes.min(),
    content_sizes.max())
	
#####

# Response Code Count
responseCodeToCount = (access_logs
                       .map(lambda log: (log.response_code, 1))
                       .reduceByKey(lambda a, b : a + b)
                       .cache())
responseCodeToCountList = responseCodeToCount.take(100)
print 'Found %d response codes' % len(responseCodeToCountList)
print 'Response Code Counts: %s' % responseCodeToCountList

#####
# percentage of return codes
labels = responseCodeToCount.map(lambda (x, y): x).collect()
print labels
count = access_logs.count()
fracs = responseCodeToCount.map(lambda (x, y): (float(y) / count)).collect()
print fracs

#####

# Any hosts that has accessed the server more than 10 times.
hostCountPairTuple = access_logs.map(lambda log: (log.host, 1))

hostSum = hostCountPairTuple.reduceByKey(lambda a, b : a + b)

hostMoreThan10 = hostSum.filter(lambda s: s[1] > 10)

hostsPick20 = (hostMoreThan10
               .map(lambda s: s[0])
               .take(20))

print 'Any 20 hosts that have accessed more then 10 times: %s' % hostsPick20

#####

# end point and their counts

endpoints = (access_logs
             .map(lambda log: (log.endpoint, 1))
             .reduceByKey(lambda a, b : a + b)
             .cache())
ends = endpoints.map(lambda (x, y): x).collect()
counts = endpoints.map(lambda (x, y): y).collect()

# Top Endpoints
endpointCounts = (access_logs
                  .map(lambda log: (log.endpoint, 1))
                  .reduceByKey(lambda a, b : a + b))

topEndpoints = endpointCounts.takeOrdered(10, lambda s: -1 * s[1])

print 'Top Ten Endpoints: %s' % topEndpoints

#### which endpoints did not have a 200 return code
not200 = access_logs.filter(lambda log: log.response_code != 200)
endpointCountPairTuple = not200.map(lambda log: (log.endpoint, 1))
endpointSum = endpointCountPairTuple.reduceByKey(lambda a, b : a + b)
topTenErrURLs = endpointSum.takeOrdered(10, lambda s: -1 * s[1])

#### Number of Unique Hosts
hosts = access_logs.map(lambda log: (log.host, 1)).reduceByKey(lambda a, b : a + b)
uniqueHosts = hosts.map(lambda (k,v):k)
uniqueHostCount = uniqueHosts.count()
print 'Unique hosts: %d' % uniqueHostCount

#### Number of Unique Daily Hosts
dayToHostPairTuple = access_logs.map(lambda x: (x.date_time.day,x.host)).distinct()
dayGroupedHosts = dayToHostPairTuple.map(lambda (k,v): (k,1))
dayHostCount = dayGroupedHosts.reduceByKey(lambda a,b: a+b)
dailyHosts = (dayHostCount.sortByKey().cache())
dailyHostsList = dailyHosts.take(30)
print 'Unique hosts per day: %s' % dailyHostsList

# OR # 

dayToHostPairTuple = access_logs.map(lambda x: (x.date_time.day,x.host))
dayGroupedHosts = dayToHostPairTuple.groupByKey()
dayHostCount = dayGroupedHosts.map(lambda (k,v): (k, len(set(v)) ) )
dailyHosts = (dayHostCount.sortByKey().cache())
dailyHostsList = dailyHosts.take(30)
print 'Unique hosts per day: %s' % dailyHostsList


##### Average Number of Daily Requests per Hosts
dayAndHostTuple = access_logs.map(lambda x: (x.date_time.day, x.host)) # step 1: create (k,v) pair (date_time, host)
groupedByDay = dayAndHostTuple.map(lambda (k,v): (k,1)).reduceByKey(lambda a,b:a+b) # step 2: map k in (k,v) and then reduce to get the count of day_time
sortedByDay = groupedByDay.sortByKey() # step 3: sort by key
avgDailyReqPerHost = (sortedByDay.join(dailyHosts).map(lambda x: (x[0],int(x[1][0]/x[1][1]))).sortByKey().cache()) 
avgDailyReqPerHostList = avgDailyReqPerHost.take(30)
print 'Average number of daily requests per Hosts is %s' % avgDailyReqPerHostList


##### focusing on 404 code errors

badRecords = (access_logs.filter(lambda log: log.response_code == 404).cache())

# print out a list up to 40 distinct endpoints that generate 404 errors

badEndpoints = badRecords.map(lambda x: (x.endpoint,1))
badUniqueEndpoints = badEndpoints.distinct()
badUniqueEndpointsPick40 = badUniqueEndpoints.take(40)
print '404 URLS: %s' % badUniqueEndpointsPick40

# print out a list of the top twenty endpoints that generate the most 404 errors

badEndpointsCountPairTuple = badRecords.map(lambda x: (x.endpoint,1))
badEndpointsSum = badEndpointsCountPairTuple.reduceByKey(lambda a,b: a+b)
badEndpointsTop20 = badEndpointsSum.takeOrdered(20, lambda s: -1 * s[1])
print 'Top Twenty 404 URLs: %s' % badEndpointsTop20

# print out a list of the top twenty-five hosts that generate the most 404 errors

errHostsCountPairTuple = badRecords.map(lambda x: (x.host,1))
errHostsSum = errHostsCountPairTuple.reduceByKey(lambda a,b: a+b)
errHostsTop25 = errHostsSum.takeOrdered(25, lambda s: -1 * s[1])
print 'Top 25 hosts that generated errors: %s' % errHostsTop25

# 404 requests by day expressed as a list

errDateCountPairTuple = badRecords.map(lambda x: (x.date_time.day, 1))
errDateSum = errDateCountPairTuple.reduceByKey(lambda a,b: a+b)
errDateSorted = (errDateSum.sortByKey()
                 .cache())
errByDate = errDateSorted.collect()
print '404 Errors by day: %s' % errByDate

# top five days for 404 response codes 

topErrDate = errDateSorted.takeOrdered(5, lambda s: -1 * s[1])
print 'Top Five dates for 404 requests: %s' % topErrDate

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

import re
DATAFILE_PATTERN = '^(.+),"(.+)",(.*),(.*),(.*)'

def removeQuotes(s):
    """ Remove quotation marks from an input string
    Args:
        s (str): input string that might have the quote "" characters
    Returns:
        str: a string without the quote characters
    """
    return ''.join(i for i in s if i!='"')


def parseDatafileLine(datafileLine):
    """ Parse a line of the data file using the specified regular expression pattern
    Args:
        datafileLine (str): input string that is a line from the data file
    Returns:
        str: a string parsed using the given regular expression and without the quote characters
    """
    match = re.search(DATAFILE_PATTERN, datafileLine)
    if match is None:
        print 'Invalid datafile line: %s' % datafileLine
        return (datafileLine, -1)
    elif match.group(1) == '"id"':
        print 'Header datafile line: %s' % datafileLine
        return (datafileLine, 0)
    else:
        product = '%s %s %s' % (match.group(2), match.group(3), match.group(4))
        return ((removeQuotes(match.group(1)), product), 1)

import sys
import os
from test_helper import Test

baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab3')

GOOGLE_PATH = 'Google.csv'
GOOGLE_SMALL_PATH = 'Google_small.csv'
AMAZON_PATH = 'Amazon.csv'
AMAZON_SMALL_PATH = 'Amazon_small.csv'
GOLD_STANDARD_PATH = 'Amazon_Google_perfectMapping.csv'
STOPWORDS_PATH = 'stopwords.txt'

def parseData(filename):
    """ Parse a data file
    Args:
        filename (str): input file name of the data file
    Returns:
        RDD: a RDD of parsed lines
    """
    return (sc
            .textFile(filename, 4, 0)
            .map(parseDatafileLine)
            .cache())

def loadData(path):
    """ Load a data file
    Args:
        path (str): input file name of the data file
    Returns:
        RDD: a RDD of parsed valid lines
    """
    filename = os.path.join(baseDir, inputPath, path)
    raw = parseData(filename).cache()
    failed = (raw
              .filter(lambda s: s[1] == -1)
              .map(lambda s: s[0]))
    for line in failed.take(10):
        print '%s - Invalid datafile line: %s' % (path, line)
    valid = (raw
             .filter(lambda s: s[1] == 1)
             .map(lambda s: s[0])
             .cache())
    print '%s - Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (path,
                                                                                        raw.count(),
                                                                                        valid.count(),
                                                                                        failed.count())
    assert failed.count() == 0
    assert raw.count() == (valid.count() + 1)
    return valid

googleSmall = loadData(GOOGLE_SMALL_PATH)
google = loadData(GOOGLE_PATH)
amazonSmall = loadData(AMAZON_SMALL_PATH)
amazon = loadData(AMAZON_PATH)

#####

# function simpleTokenize(string) that takes a string and returns a list of non-empty tokens in the string
split_regex = r'\W+'

def simpleTokenize(string):
    string = string.lower()
    string = re.split(split_regex,string) #splits at non-words \W 
    string = filter(None, string) # filters (removes) empty strings
	string = filter(lambda string: string not in stopwords,string) #assumes stopwords = list of stopwords, and removed stopwords from token
    return string

print simpleTokenize(quickbrownfox)

### How many tokens, total, are there in the two datasets?

amazonRecToToken = amazonSmall.map(lambda (k,v): (k,tokenize(v)))
googleRecToToken = googleSmall.map(lambda (k,v): (k,tokenize(v)))


def countTokens(vendorRDD):
    CountPerID = vendorRDD.map(lambda (k,v): len(v)).collect()
    TotalCount = sum(CountPerID)
    """Count and return the number of tokens
    Args:
        vendorRDD (RDD of (recordId, tokenizedValue)): Pair tuple of record ID to tokenized output
    Returns:
        count: count of all tokens
    """
    return TotalCount

totalTokens = countTokens(amazonRecToToken) + countTokens(googleRecToToken)
print 'There are %s tokens in the combined datasets' % totalTokens

### Amazon record with the most tokens

def findBiggestRecord(vendorRDD):
    CountPerIDPair = vendorRDD.map(lambda (k,v): (k,len(v))).collect()
    CountPerIDPairOrdered = sorted(CountPerIDPair,key=lambda x: x[1], reverse=True) #sorted by value in (key value pair)
    return CountPerIDPairOrdered

biggestRecordAmazon = findBiggestRecord(amazonRecToToken)

print 'The Amazon record with ID "%s" has the most tokens (%s)' % (biggestRecordAmazon[0][0],
                                                                 (biggestRecordAmazon[0][1])) # returns The Amazon record with ID "b000o24l3q" has the most tokens (1547)

## OR ##

def findBiggestRecord(vendorRDD):
    CountPerIDPair = vendorRDD.collect()
    CountPerIDPairOrdered = sorted(CountPerIDPair,key=lambda x: len(x[1]), reverse=True) #sorted by length of value in (key value pair)
    return CountPerIDPairOrdered


biggestRecordAmazon = findBiggestRecord(amazonRecToToken)

print 'The Amazon record with ID "%s" has the most tokens (%s)' % (biggestRecordAmazon[0][0],
                                                                 len(biggestRecordAmazon[0][1]))
																 

## Implementing TF
from collections import Counter
def tf(tokens):
    Init_dict = {} #create an empty dictionary
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens 
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    Uniq_tokens = Counter(tokens) #use Counter function to identify unique tokens and return a dictionary like {'token1': 1 , 'token2': 1, .....}
    Len_tokens = len(tokens)
    for v in Uniq_tokens:
        Init_dict.update(Uniq_tokens) # Update Init_dict to equal Uniq_tokens
    
    Init_dict = dict((k, float(v)/float(Len_tokens)) for (k,v) in Init_dict.iteritems()) # iterate over Init_dict values and divide them by Len_tokens
    return Init_dict # return TF as a dictionary

print tf(tokenize(quickbrownfox)) # Should return a dictionary like { 'quick': 0.1666 ... }

## create a corpus .. i.e. join the documents

corpusRDD = amazonRecToToken.fullOuterJoin(googleRecToToken)
corpuskv = corpusRDD.collect()
print corpuskv[0:2] # print first 2 elements of the list

## Implement an IDFs function

def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """
    N = corpus.count()   #step 1: compute the number of documents
    uniqueTokens = corpus.flatMap(lambda (k,v): set(v)) # step 2: compute "unique" tokens in each document. That's why we use "set" on value to return unique values per document
    tokenCountPairTuple = uniqueTokens.map(lambda x: (x,1.0)) # step 3: each token is now converted into a "token,1.0" key value pair. We use "1.0" instead of "1" to enable float
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a,b: a+b) # step 4: reduce by key to calculate the sum of tokens. This gives us (token, n)
    return (tokenSumPairTuple.map(lambda(k,v): (k,float(N/v)))) # step 5: compute N/n
    


idfsSmall = idfs(amazonRecToToken.union(googleRecToToken))
uniqueTokenCount = idfsSmall.count()

print 'There are %s unique tokens in the small datasets.' % uniqueTokenCount 

## calculating TFIDF value ... bringing it all together

def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens) 
    tfIdfDict = {k : v * tfs[k] for k, v in idfs.items() if k in tfs} # take k,v from idfs --> modify v, by multiplying it by tfs value (tfs[k]) for all values of k where there is a match in k in tfs and idfs data #inline function
    return tfIdfDict

recb000hkgj8k = amazonRecToToken.filter(lambda x: x[0] == 'b000hkgj8k').collect()[0][1]
idfsSmallWeights = idfsSmall.collectAsMap() # collectAsMap() Returns the key-value pairs in this RDD to the master as a dictionary
rec_b000hkgj8k_weights = tfidf(recb000hkgj8k, idfsSmallWeights)

print 'Amazon record "b000hkgj8k" has tokens and weights:\n%s' % rec_b000hkgj8k_weights

## defining cosine similarity

# TODO: Replace <FILL IN> with appropriate code
import math

def dotprod(a, b):
    dprod = {k : v * a[k] for k, v in b.items() if k in a} #calculate product of values between 2 dictionaries with potentially non-similar keys. #inline function
    """ Compute dot product
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    return sum(dprod.values())

def norm(a):
    total = 0
    for value in a:
        v = a[value]*a[value]
        total = v + total
    """ Compute square root of the dot product
    Args:
        a (dictionary): a dictionary of record to value
    Returns:
        norm: a dictionary of tokens to its TF values
    """
    return math.sqrt(float(total))

def cossim(a, b):
    dprod = float(dotprod(a,b))
    norma = norm(a)
    normb = norm(b)
    cosine = dprod/(norma*normb)
    """ Compute cosine similarity
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        cossim: dot product of two dictionaries divided by the norm of the first dictionary and
                then by the norm of the second dictionary
    """
    return cosine

testVec1 = {'foo': 2, 'bar': 3, 'baz': 5 }
testVec2 = {'foo': 1, 'bar': 0, 'baz': 20 }
dp = dotprod(testVec1, testVec2)
nm = norm(testVec1)
print dp, nm

def cosineSimilarity(string1, string2, idfsDictionary):
    w1list = tokenize(string1)
    w2list = tokenize(string2)
    """ Compute cosine similarity between two strings
    Args:
        string1 (str): first string
        string2 (str): second string
        idfsDictionary (dictionary): a dictionary of IDF values
    Returns:
        cossim: cosine similarity value
    """
    w1 = tfidf(w1list,idfsDictionary)
    w2 = tfidf(w2list,idfsDictionary)
    return cossim(w1, w2)

cossimAdobe = cosineSimilarity('Adobe Photoshop','Adobe Illustrator',idfsSmallWeights)

print cossimAdobe


# Computing cross similarity
crossSmall = (googleSmall
              .cartesian(amazonSmall)
              .cache()) # Return the Cartesian product of this RDD and another one, that is, the RDD of all pairs of elements (a, b) where a is in self and b is in other. The result will be an RDD of the form: [ ((Google URL1, Google String1), (Amazon ID1, Amazon String1)), ((Google URL1, Google String1), (Amazon ID2, Amazon String2)), ((Google URL2, Google String2), (Amazon ID1, Amazon String1)), ... ]

def computeSimilarity(record):
    """ Compute similarity on a combination record
    Args:
        record: a pair, (google record, amazon record)
    Returns:
        pair: a pair, (google URL, amazon ID, cosine similarity value)
    """
    googleRec = record[0] # returns (a,a1) from a pair ((a,a1),(b,b1))
    amazonRec = record[1] # returns (b,b1) from a pair ((a,a1),(b,b1))
    googleURL = record[0][0] # returns (a) from a pair ((a,a1),(b,b1))
    amazonID = record[1][0] # returns (b) from a pair ((a,a1),(b,b1))
    googleValue = record[0][1] # returns (a1) from a pair ((a,a1),(b,b1))
    amazonValue = record[1][1] # returns (b1) from a pair ((a,a1),(b,b1))
    try:
        cs = cosineSimilarity(googleValue, amazonValue, idfsSmallWeights)
    except:
        cs = 0
    return (googleURL,amazonID,cs)


similarities = (crossSmall
                .map(lambda x: computeSimilarity(x))
                .cache())


def similar(amazonID, googleURL):
    """ Return similarity value
    Args:
        amazonID: amazon ID
        googleURL: google URL
    Returns:
        similar: cosine similarity value
    """
    return (similarities
            .filter(lambda record: (record[0] == googleURL and record[1] == amazonID))
            .collect()[0][2])

similarityAmazonGoogle = similar('b000o24l3q', 'http://www.google.com/base/feeds/snippets/17242822440574356561')
print 'Requested similarity is %s.' % similarityAmazonGoogle

# Similar code using a broadcast variable

def computeSimilarityBroadcast(record):
    """ Compute similarity on a combination record, using Broadcast variable
    Args:
        record: a pair, (google record, amazon record)
    Returns:
        pair: a pair, (google URL, amazon ID, cosine similarity value)
    """
    googleRec = record[0]
    amazonRec = record[1]
    googleURL = record[0][0]
    amazonID = record[1][0]
    googleValue = record[0][1]
    amazonValue = record[1][1]
    cs = cosineSimilarity(googleValue, amazonValue, idfsSmallBroadcast.value)
    return (googleURL, amazonID, cs)

idfsSmallBroadcast = sc.broadcast(idfsSmallWeights)
similaritiesBroadcast = (crossSmall
                         .map(lambda x: computeSimilarity(x))
                         .cache())

				 
						 
def similarBroadcast(amazonID, googleURL):
    """ Return similarity value, computed using Broadcast variable
    Args:
        amazonID: amazon ID
        googleURL: google URL
    Returns:
        similar: cosine similarity value
    """
    return (similaritiesBroadcast
            .filter(lambda record: (record[0] == googleURL and record[1] == amazonID))
            .collect()[0][2])

similarityAmazonGoogleBroadcast = similarBroadcast('b000o24l3q', 'http://www.google.com/base/feeds/snippets/17242822440574356561')
print 'Requested similarity is %s.' % similarityAmazonGoogleBroadcast


# the following set of code assumes availability of a RDD called goldStandard that has high cosine similarity scores for "AmazonID GoogleURL" pairs

sims = similaritiesBroadcast.map(lambda x: (x[1]+' '+x[0],x[2]))  #step1: each element consists of a pair of the form ("AmazonID GoogleURL", cosineSimilarityScore).

trueDupsRDD = (sims
               .join(goldStandard)) # step 2: RDD that has the just the cosine similarity scores for those "AmazonID GoogleURL" pairs that appear in BOTH the sims RDD and goldStandard RDD. 

trueDupsCount = trueDupsRDD.count() # step 3: Count the number of true duplicate pairs in the trueDupsRDD dataset

avgSimDups = sum(trueDupsRDD.map(lambda x: x[1][0]).collect())/float(trueDupsCount) # step 4: Compute the average similarity score


nonDupsRDD = (sims
              .subtractByKey(goldStandard)) # step 5: RDD that has the just the cosine similarity scores for those "AmazonID GoogleURL" pairs from the similaritiesBroadcast RDD that "DO NOT" appear in both the sims RDD and gold standard RDD. This is accomplished using subtractByKey function.

avgSimNon = sum(nonDupsRDD.map(lambda x: x[1]).collect())/float(nonDupsRDD.count()) # step 5: Compute the average similarity score

print 'There are %s true duplicates.' % trueDupsCount
print 'The average similarity of true duplicates is %s.' % avgSimDups
print 'And for non duplicates, it is %s.' % avgSimNon



# the next lines of code work with a much larger dataset


amazonFullRecToToken = amazon.map(lambda (k,v): (k,tokenize(v)))
googleFullRecToToken = google.map(lambda (k,v): (k,tokenize(v)))
print 'Amazon full dataset is %s products, Google full dataset is %s products' % (amazonFullRecToToken.count(),
                                                                                    googleFullRecToToken.count())

# Create full dataset and compute IDF values for tokens																					
fullCorpusRDD = amazonFullRecToToken.union(googleFullRecToToken) # union returns a single list of comprising all elements from RDD A and RDD B.
idfsFull = idfs(fullCorpusRDD)
idfsFullCount = idfsFull.count()
print 'There are %s unique tokens in the full datasets.' % idfsFullCount

# Recompute IDFs for full dataset as a dictionary and broadcast
idfsFullWeights = idfsFull.collectAsMap()
idfsFullBroadcast = sc.broadcast(idfsFullWeights)

# Pre-compute TF-IDF weights.  Build mappings from record ID weight vector.
amazonWeightsRDD = amazonFullRecToToken.map(lambda x: (x[0],tfidf(x[1], idfsFullWeights)))  # here the lambda function uses a user defined function tfidf
googleWeightsRDD = googleFullRecToToken.map(lambda x: (x[0],tfidf(x[1], idfsFullWeights)))
print 'There are %s Amazon weights and %s Google weights.' % (amazonWeightsRDD.count(),
                                                              googleWeightsRDD.count())

# Compute Norms for the weights from the full datasets

amazonNorms = amazonWeightsRDD.map(lambda x: (x[0],norm(x[1]))).collectAsMap() # another example where lambda function uses a user defined function norm
amazonNormsBroadcast = sc.broadcast(amazonNorms)
googleNorms = googleWeightsRDD.map(lambda x: (x[0],norm(x[1]))).collectAsMap()
googleNormsBroadcast = sc.broadcast(googleNorms)

# Create an invert function that given a pair of (ID/URL, TF-IDF weighted token vector{dictionary}), returns a list of pairs of (token, ID/URL)

def invert(record):
    """ Invert (ID, tokens) to a list of (token, ID)
    Args:
        record: a pair, (ID, token vector) where record is a list containing ID and token vector (token vector is a dictionary)
    Returns:
        pairs: a list of pairs of token to ID
    """
    pairs = []

    for k,v in record[1].iteritems():
        tmp = [k,record[0]]
        pairs.append(tmp)
    return (pairs)

amazonInvPairsRDD = (amazonWeightsRDD
                    .flatMap(lambda x: invert(x))
                    .cache())

googleInvPairsRDD = (googleWeightsRDD
                    .flatMap(lambda x: invert(x))
                    .cache())

print 'There are %s Amazon inverted pairs and %s Google inverted pairs.' % (amazonInvPairsRDD.count(),
                                                                            googleInvPairsRDD.count())

																			
# def swap(record):
    """ Swap (token, (ID, URL)) to ((ID, URL), token)
    Args:
        record: a pair, (token, (ID, URL))
    Returns:
        pair: ((ID, URL), token)
    """
    token = record[0]
    keys = record[1]
    return (keys, token)

commonTokens = (amazonInvPairsRDD
                .join(googleInvPairsRDD).map(lambda x: swap(x)).groupByKey().map(lambda e: (e[0], list(e[1]))) # last map function creates a kev value pair where value is a list
                .cache())

print 'Found %d common tokens' % commonTokens.count()


amazonWeightsBroadcast = sc.broadcast(amazonWeightsRDD.collectAsMap())
googleWeightsBroadcast = sc.broadcast(googleWeightsRDD.collectAsMap())

# Create a fastCosinesSimilarity function

def fastCosineSimilarity(record):
    """ Compute Cosine Similarity using Broadcast variables
    Args:
        record: ((ID, URL), token)
    Returns:
        pair: ((ID, URL), cosine similarity value)
    """
    amazonRec = record[0][0]
    googleRec = record[0][1]
    tokens = record[1]


    s = sum((amazonWeightsBroadcast.value[amazonRec][i]*googleWeightsBroadcast.value[googleRec][i] for i in tokens)) # note the use of .value in a broadcast variable
    
    value = s/(float(amazonNormsBroadcast.value[amazonRec])*float(googleNormsBroadcast.value[googleRec])) # note the use of .value in a broadcast variable
    key = (amazonRec, googleRec)
    return (key, value)
    

similaritiesFullRDD = (commonTokens
                       .map(lambda x: fastCosineSimilarity(x))
                       .cache())

print similaritiesFullRDD.count()

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

import sys
import os


baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab4', 'small')

ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)

def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]


ratingsRDD = rawRatings.map(get_ratings_tuple).cache() ## ratingsRDD are tuples of form (UserID, MovieID, Rating)
moviesRDD = rawMovies.map(get_movie_tuple).cache() ## moviesRDD are tuples of form (MovieID, Title)

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print 'There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount)
print 'Ratings: %s' % ratingsRDD.take(3)
print 'Movies: %s' % moviesRDD.take(3)

assert ratingsCount == 487650
assert moviesCount == 3883
assert moviesRDD.filter(lambda (id, title): title == 'Toy Story (1995)').count() == 1
assert (ratingsRDD.takeOrdered(1, key=lambda (user, movie, rating): movie)
        == [(1, 1, 5.0)])   # very nice way of ordering. "takeorder(1," --> returns 1 result. "key=lambda (user, movie, rating): movie)" --> orders by movie. If instead of : movie we use -1*user, we get descending order by user.

### the following code explains an issue with 'plain vanilla' sortByKey sorting

tmp1 = [(1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'delta')]
tmp2 = [(1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'alpha')]

oneRDD = sc.parallelize(tmp1)
twoRDD = sc.parallelize(tmp2)
oneSorted = oneRDD.sortByKey(True).collect()
twoSorted = twoRDD.sortByKey(True).collect()
# Even though the two lists contain identical tuples, there is a difference in ordering
print oneSorted ## returns [(1, u'alpha'), (1, u'epsilon'), (1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]
print twoSorted ## returns [(1, u'delta'), (1, u'epsilon'), (1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]

# to fix that ..... we can use a sort function like the one below

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = unicode('%.3f' % tuple[0]) ## we use unicode since the data is unicode. '%.3f' is used for 3 decimal point number
    value = tuple[1]
    return (key + ' ' + value) ## returns concatenated unicode label 

print oneRDD.map(lambda x: sortFunction(x)).collect() # returns [u'1.000 alpha', u'2.000 alpha', u'2.000 beta', u'3.000 alpha', u'1.000 epsilon', u'1.000 delta']
# the use of .sortBy in conjunction with the UDF 'sortFunction' as shown below eliminates the issue of 'plain vanilla' sortByKey sorting
print oneRDD.sortBy(sortFunction, True).collect() # returns [(1, u'alpha'), (1, u'delta'), (1, u'epsilon'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]
print twoRDD.sortBy(sortFunction, True).collect() # returns [(1, u'alpha'), (1, u'delta'), (1, u'epsilon'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]

# If we just want to look at the first few elements of the RDD in sorted order, we can use the takeOrdered method with the sortFunction we defined

oneSorted1 = oneRDD.takeOrdered(oneRDD.count(),key=sortFunction)
twoSorted1 = twoRDD.takeOrdered(twoRDD.count(),key=sortFunction)

# create a helper function getCountsAndAverages() 

def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    numratings = len(IDandRatingsTuple[1])
    sumratings = float(sum(IDandRatingsTuple[1]))
    avratings = float(sumratings/float(numratings))
    movID = IDandRatingsTuple[0]
    return (movID,(numratings,avratings))
	
tuple1 = (1, (1, 2, 3, 4))

print getCountsAndAverages(tuple1) # returns (1, (4, 2.5))

# From ratingsRDD with tuples of (UserID, MovieID, Rating) create an RDD with tuples of
# the (MovieID, iterable of Ratings for that MovieID)
movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda(a,b,c): (b,c)).groupByKey())
print 'movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3)

# Using `movieIDsWithRatingsRDD`, compute the number of ratings and average rating for each movie to
# yield tuples of the form (MovieID, (number of ratings, average rating))
movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(lambda x: getCountsAndAverages(x))
print 'movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3)

# To `movieIDsWithAvgRatingsRDD`, apply RDD transformations that use `moviesRDD` to get the movie
# names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form (average rating, movie name, number of ratings)
movieNameWithAvgRatingsRDD = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD).map(lambda(k,v): v).map(lambda x: (x[1][1],x[0],x[1][0])))
print 'movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3)

# Apply an RDD transformation to movieNameWithAvgRatingsRDD to limit the results to movies with
# ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by the
# average rating to get the movies in order of their rating (highest rating first)
movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda (a,b,c): c>500)
                                    .sortBy(sortFunction, False))
print 'Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20)

# randomly split the dataset into the multiple groups (training, validation and test)
trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L) # use of randomsplit transformation

# Calculate RMSE

import math

def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda (UserID, MovieID, Rating): ((UserID, MovieID),Rating))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda (UserID, MovieID, Rating): ((UserID, MovieID),Rating))

    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD).map(lambda ((u,m),(p,a)): float(math.pow((p-a),2))))

    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.sum()

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(float(totalError/numRatings))


# sc.parallelize turns a Python list into a Spark RDD.
testPredicted = sc.parallelize([
    (1, 1, 5),
    (1, 2, 3),
    (1, 3, 4),
    (2, 1, 3),
    (2, 2, 2),
    (2, 3, 4)])
testActual = sc.parallelize([
     (1, 2, 3),
     (1, 3, 5),
     (2, 1, 5),
     (2, 2, 1)])
testPredicted2 = sc.parallelize([
     (2, 2, 5),
     (1, 2, 5)])
testError = computeError(testPredicted, testActual)
print 'Error for test dataset (should be 1.22474487139): %s' % testError

testError2 = computeError(testPredicted2, testActual)
print 'Error for test dataset2 (should be 3.16227766017): %s' % testError2

testError3 = computeError(testActual, testActual)
print 'Error for testActual dataset (should be 0.0): %s' % testError3

### using ALS algorithm

### Step 1: using validation data and training data to identify best parameters for ALS model
from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda (a,b,c): (a,b))

seed = 5L
iterations = 5
regularizationParameter = 0.1  #regularization parameter (lambda) like in lasso and ridge penalizes overly complex models. Zero lambda leads to over fitting.
ranks = [4, 8, 12] #this is another parameter for model simplification. Very high rank leads to over fitting, and low ranks leads to under fitting. 
errors = [0, 0, 0]
err = 0
tolerance = 0.03

minError = float('inf') #set minError to infinity
bestRank = -1 # set bestRank to -1
bestIteration = -1 # set bestIteration to -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter) # build ALS model using training dataset
    predictedRatingsRDD = model.predictAll(validationForPredictRDD) # use model on validation dataset using predictAll() method
    error = computeError(predictedRatingsRDD, validationRDD) #compute RMSE between predicted values and actual values
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank

### Step 2: using test data on the best model. The best model had rank = 8

myModel = ALS.train(trainingRDD, rank = 8, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (a,b,c): (a,b))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

### Step 3: Compare RMSE for the results predicted by the model versus the average rating for the training set (additional optional analysis to determine goodness of model)

trainingAvgRating = float(trainingRDD.map(lambda (UserID, MovieID, Rating):(Rating)).sum()/trainingRDD.map(lambda (UserID, MovieID, Rating):(Rating)).count()) # compute average of training data
print 'The average rating for movies in the training set is %s' % trainingAvgRating

testForAvgRDD = testRDD.map(lambda (UserID, MovieID, Rating):(UserID, MovieID,trainingAvgRating)) # Use the average rating and the testRDD to create an RDD with entries of the form (userID, movieID, average rating)
testAvgRMSE = computeError(testRDD, testForAvgRDD) 
print 'The RMSE on the average set is %s' % testAvgRMSE

## print most rated movies
print 'Most rated movies:'
print '(average rating, movie name, number of reviews)'
for ratingsTuple in movieLimitedAndSortedByRatingRDD.take(50):
    print ratingsTuple

## Step 4: Do predictive analysis for a user.

myUserID = 0

# myRatedMovies tuples are (UserID, MovieID, Rating)
myRatedMovies = [
     (0,1214,5), 
    (0,480,5),
    (0,1079,3.5),
    (0,3623,5),
    (0,1580,4),
    (0,1221,5),
    (0,527,5),
    (0,50,5),
    (0,1196,5),
    (0,70,5)
    ]
myRatingsRDD = sc.parallelize(myRatedMovies)
print 'My movie ratings: %s' % myRatingsRDD.take(10) 	

trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)  # add user Ratings to training set

print ('The training dataset now has %s more entries than the original training dataset' %
       (trainingWithMyRatingsRDD.count() - trainingRDD.count()))
	   
myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter) # build ALS model with new training set

predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD) ## apply model to test dataset to create predicted RDD
testRMSEMyRatings = computeError(testRDD,predictedTestMyRatingsRDD) ## compute RMSE between test data set and predicted values
print 'The model had a RMSE on the test set of %s' % testRMSEMyRatings

## create an RDD for all movies that user has NOT rated

# Use the Python list myRatedMovies to transform the moviesRDD into an RDD with entries that are pairs of 
# the form (myUserID, Movie ID) and that does not contain any movies that you have rated.
myRatedMoviesRDD = myRatingsRDD.map(lambda(UID,MID,Rating): (MID))
myUnratedMoviesRDD = (moviesRDD
                      .map(lambda (MID,Title):(MID)).subtract(myRatedMoviesRDD).map(lambda(MID):(0,MID))) ### This transformation will yield an RDD of the form: [(0, 1), (0, 2), (0, 3), (0, 4)]

# Use the input RDD, myUnratedMoviesRDD, with myRatingsModel.predictAll() to predict your ratings for the movies
predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)

