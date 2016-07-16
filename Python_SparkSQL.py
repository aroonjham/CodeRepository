# basic spark code FOR JOINs

df.join(df2, 'col_name') # df inner join with df2 on col_name. Rteurns all columns from df and df2

df.join(df2, 'col_name').select(df.df_col_name, df2.df2_col_name) # df inner join with df2 on col_name. Returns specificed columns

df.join(df2, 'col_name', outer) # outer join between df and df2

df.join(df2, 'col_name', left_outer) # left outer join between df and df2
## https://spark.apache.org/docs/latest/

# Display the type of the Spark sqlContext
type(sqlContext)

# List sqlContext's attributes
dir(sqlContext) # You can use Python's dir() function to get a list of all the attributes (including methods) accessible through the sqlContext object

# Use help to obtain more detailed information
help(sqlContext)

# Use sc.version to see what version of Spark we are running
sc.version

###############################################################
###															###
###															###
###   		Using PySpark to generate fake data				###
###															###
###															###
###############################################################

from faker import Factory
fake = Factory.create()
fake.seed(4321)

# Each entry consists of last_name, first_name, ssn, job, and age (at least 1)
from pyspark.sql import Row
def fake_entry():
  name = fake.name().split()
  return Row(name[1], name[0], fake.ssn(), fake.job(), abs(2016 - fake.date_time().year) + 1)
 
# Create a helper function to call a function repeatedly
def repeat(times, func, *args, **kwargs):
    for _ in xrange(times):
        yield func(*args, **kwargs)

# create a normal Python list, containing Spark SQL Row objects		
data = list(repeat(10000, fake_entry))

len(data) # confirm 10000 rows using length function LEN

# to inspect a row of data which is a list here, we can use
data[0][0], data[0][1], data[0][2], data[0][3], data[0][4]

###############################################################
###															###
###															###
###   				SPARK SQL DATAFRAME						###
###															###
###															###
###############################################################

# to create a dataframe specify row data object (data in the example below) is, and its schema

#example 1: basic dataframe created on the fly
tempDF = sqlContext.createDataFrame([("Joe", 1), ("Joe", 1), ("Anna", 15), ("Anna", 12), ("Ravi", 5)], ('name', 'score'))

#example 2: created using an existing python list called 'data'
dataDF = sqlContext.createDataFrame(data, ('last_name', 'first_name', 'ssn', 'occupation', 'age'))

# lets confirm by examing the type of object dataDF

print 'type of dataDF: {0}'.format(type(dataDF)) # will return type of dataDF: <class 'pyspark.sql.dataframe.DataFrame'>

# Let's take a look at the DataFrame's schema and some of its rows
dataDF.printSchema()

###############################################################
###															###
###															###
###   				LOADING A TEXT FILE						###
###															###
###															###
###############################################################

fileName = "dbfs:/databricks-datasets/cs100/lab1/data-001/shakespeare.txt"

#sqlContext.read.text loads a text file and returns a DataFrame with a single string column named “value”
shakespeareDF = sqlContext.read.text(fileName).select(removePunctuation(col('value')))
shakespeareDF.show(15, truncate=False) #trucate =false ensure that row elements text is NOT truncated

# Importing .csv into databricks
#  click on the Tables icon, and choose 'Create Table'.  Now you can browse to a file on your local hard drive and upload it
# After you upload databricks returns the location of the csv 
# for example, you will see Uploaded to DBFS /FileStore/tables/r1n72xzm1467330705343/WHO.csv
#Now you can actually load the data as a Spark DataFrame like
myData = sqlContext.read.format("com.databricks.spark.csv").load('dbfs:/FileStore/tables/r1n72xzm1467330705343/WHO.csv', header=True, inferSchema=True)



###############################################################
###															###
###															###
###   				RUNNING SQL COMMANDS					###
###															###
###															###
###############################################################

# register the newly created DataFrame as a named table, using the registerDataFrameAsTable() method. You can use SQL statements with a named table.
sqlContext.registerDataFrameAsTable(dataDF, 'dataframe')

# Now you can run a sql statement against 'dataframe'
%sql
SELECT * FROM dataframe
%sql
SELECT count(*) FROM dataframe where occupation = 'Web designer'

# remember the temporary table we created ....
webdesigners = sqlContext.sql("SELECT * FROM dataframe where occupation = 'Web designer'")

###############################################################
###															###
###															###
###   Basic Transformations and actions(select and show)	###
###															###
###															###
###############################################################

# To understand what methods can we call on this DataFrame
help(dataDF)

# How many partitions will the DataFrame be split into?
dataDF.rdd.getNumPartitions()

# Transform dataDF through a select transformation and rename the newly created '(age -1)' column to 'age' using alias()
# Because select is a transformation and Spark uses lazy evaluation, no jobs, stages,
# or tasks will be launched when we run this code.
subDF = dataDF.select('last_name', 'first_name', 'ssn', 'occupation', (dataDF.age - 1).alias('age'))

# lets use the length function to find the number of characters in each word
sub2DF = dataDF.select('last_name', 'first_name', 'ssn', 'occupation', length(dataDF.occupation).alias('length'))
sub2DF = dataDF.select('last_name', 'first_name', 'ssn', 'occupation', length(col('occupation')).alias('length')) # used col(). Returns a Column based on the given column name

#to  take a look at the query plan
subDF.explain(True)

# Let's collect the data. Collect is an action v/s select which is a transformation
results = subDF.collect()
print results

# A better way to visualize the data is to use the show() method. If you don't tell show() how many rows to display, it displays 20 rows
subDF.show()

# If you'd prefer that show() not truncate the data, you can tell it not to:
subDF.show(n=30, truncate=False)

# The display() helper function in databricks is even better to visualize data
display(subDF)

###############################################################
###															###
###															###
###   				A LITTLE EXTRA ON COLLECT()				###
###															###
###															###
###############################################################

#when you do a collect, you dont get a dataframe. Instead you get an LIST of rows
collectedLIST = originalDF.collect() # once you do a collect you get 'LIST' of rows
# if you dont want a list of rows use select and map
collectedTUPLES = originalDF.select('column1', 'column2', 'column3').map(lambda r: (r[0],r[1],r[2])).collect() # returns tuples and removes row
# lets assume there are 3 rows. To split the list into individual elements, we do an unzip of the RDD
x,y,z = zip(*myRDD) # here the '*' within zip(*xxxx) unzips the list
# lets get our data frame back
myList = [list(a) for a in zip(x,y,z)]
myDataFrame = sqlContext.createDataFrame(myList,('x','y','z'))
myDataFrame.show()

###############################################################
###															###
###															###
###   	MORE SPARK SQL TRANSFORMATIONS AND ACTIONS			###
###															###
###															###
###############################################################

# count function
print dataDF.count()
print subDF.count()

# Now we will transform using filter, and then do a show and count
filteredDF = subDF.filter(subDF.age < 10)
filteredDF.show(truncate=False)
filteredDF.count()

# We can do a filter like above using a udf
from pyspark.sql.types import BooleanType
less_ten = udf(lambda s: s < 10, BooleanType()) # A UDF is a special wrapper around a function, allowing the function to be used in a DataFrame query
lambdaDF = subDF.filter(less_ten(subDF.age))
lambdaDF.show()
lambdaDF.count()

# Let's further filter so that we only select even ages
even = udf(lambda s: s % 2 == 0, BooleanType())
evenDF = lambdaDF.filter(even(lambdaDF.age))
evenDF.show()
evenDF.count()

# neat way of writing code
from pyspark.sql.functions import *
(dataDF
 .filter(dataDF.age > 20)
 .select(concat(dataDF.first_name, lit(' '), dataDF.last_name), dataDF.occupation)
 .show(truncate=False)
 )

# Want to see the first 4 rows of a dataframe? Use the take() function
display(filteredDF.take(4)) # take is an action
# if you just want to see the first row you can use the first() function
display(filteredDF.take(1)) # or filteredDF.first()

# distinct() filters out duplicate rows, and it considers all columns
print dataDF.distinct().count()

# distinct values in a column
unique_columnData_count = logsDF.select('column').distinct().count()

# dropDuplicates() is like distinct(), except that it allows us to specify the columns to compare.
AJdataDF = dataDF.dropDuplicates(['first_name','last_name'])
display(AJdataDF.take(4))
AJdataDF.count()

# deleting a column from a dataframe using drop()
# drop() is like the opposite of select(): Instead of selecting specific columns from a DataFrame, it drops a specifed column from a DataFrame
dataDF.drop('occupation').drop('age').show()

# the sample() transformation returns a new DataFrame with a random sample
sampledDF = dataDF.sample(withReplacement=False, fraction=0.10)
print sampledDF.count()
sampledDF.show()

# split() and explode() transformations
from pyspark.sql.functions import split, explode

shakeWordsSplit = (shakespeareDF
                .select(split(shakespeareDF.word,' ').alias('word'))) # here split(DF,' ') splits the sentence at a space and returns each word in a single row
				
shakeWordsExplode = (shakeWordsSplit
                    .select(explode(shakeWordsSplit.word).alias('word'))) # explode() Returns a new row for each element in the given array
					
shakeWordsDF = shakeWordsExplode.filter(shakeWordsExplode.word != '') # removes all the blanks

shakeWordsDF.show()
shakeWordsDFCount = shakeWordsDF.count()
print shakeWordsDFCount

###############################################################
###															###
###															###
###   						GROUP BY						###
###															###
###															###
###############################################################

# groupBy allows you to perform aggregations on a DataFrame
# the most commonly used aggregation function is count(), but there are others (like sum(), max(), and avg()

dataDF.groupBy('occupation').count().show(truncate=False)

# the following groupBy returns only one row ... average of all rows
dataDF.groupBy().avg('age').show(truncate=False)

print "Maximum age: {0}".format(dataDF.groupBy().max('age').first()[0]) # remember, groupBy returns a row. Thats why you have [0] appended at the end
print "Minimum age: {0}".format(dataDF.groupBy().min('age').first()[0])

# whereas the following groupBy returns several row as specified by the level inside groupBy
avgOccDF = dataDF.groupBy('occupation').avg('age') # avg age by occupation
avgOccDF.show() # will return 2 columns 'occupation' and 'avg'

###############################################################
###															###
###															###
###   						SORT and ORDER					###
###															###
###															###
############################################################### 

# Get the five oldest people in the list. To do that, sort by age in descending order using orderBy transformation
orderdataDF = dataDF.orderBy(dataDF.age.desc())
display(orderdataDF.take(5))

# desc() order correct/alternate format
from pyspark.sql.functions import desc
WordsAndCountsDF = wordCount(shakeWordsDF)
topWordsAndCountsDF = WordsAndCountsDF.orderBy(desc("count"))
topWordsAndCountsDF.show()

# for ascending order
orderdataDF = dataDF.orderBy(dataDF.age)
display(orderdataDF.take(5))

# SORT operation
new_sorted_df = (original_df.groupBy('somecolumn').count().sort('somecolumn',ascending=False).cache())
Sorted_df = OriginalDF.select('A_Column').groupBy('A_Column').count().sort('count', ascending=False) # Sorting by 'A_Column'

###############################################################
###															###
###															###
###   				CACHING AND STORAGE						###
###															###
###															###
###############################################################

# if you plan to use a DataFrame more than once, then you should tell Spark to cache it
filteredDF.cache()

# Trigger an action
print filteredDF.count()

# Check if it is cached
print filteredDF.is_cached

# If we are done with the DataFrame we can unpersist it so that its memory can be reclaimed
filteredDF.unpersist()

# Check if it is cached
print filteredDF.is_cached

###############################################################
###															###
###															###
###   	subset dataframe based on filter NOT matching 		###
###															###
###															###
###############################################################

bad_content_size_df = base_df.filter(~ base_df['value'].rlike(r'\d+$')) # ~ means NOT matching a condition
from pyspark.sql.functions import lit, concat
bad_content_size_df.select(concat(bad_content_size_df['value'], lit('*'))).show(truncate=False)

not200DF = logsDF.filter(~(logsDF.target_column == '200'))

###############################################################
###															###
###															###
###   					HANDLING NAs 						###
###															###
###															###
###############################################################

# Replace all null content_size values with 0.
cleaned_df = original_df.na.fill({'columns_having_NAs': 0}) # for safety, it's better to pass a Python dictionary containing (column_name, value) mappings

# Alternatively we can drop rows containing missing values
cleaned_df = original_df.na.drop()
cleaned_df = original_df.na.drop(['columns_having_NAs']) # not tested. I am guessing we need to pass a list if we need to focus on a single column containing NAs

###############################################################
###															###
###															###
###   					PERFORMING DIVISION					###
###															###
###															###
###############################################################

FIRST_df = logsDF.select(dayofmonth('time').alias('day')).groupBy('day').count()

interim_df = (
  FIRST_df.join(SECOND_df,'day').select(FIRST_df.day, SECOND_df['count'].alias('numHosts'), FIRST_df['count'].alias('numRequest'))
)

DIVISION_CONTAINING_df = interim_df.select('day',sqlFunctions.expr('numRequest / numHosts').alias('avg_reqs_per_host_per_day')).cache()
# we used sqlFunctions.expr('numRequest / numHosts') for performing division

print 'Average number of daily requests per Hosts is:\n'
avg_daily_req_per_host_df.show()

###############################################################
###															###
###															###
###   				STRING CLEAN UP FUNCTION				###
###															###
###															###
###############################################################

from pyspark.sql.functions import regexp_replace, trim, col, lower
def removePunctuation(column):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.

    Note:
        Only spaces, letters, and numbers should be retained.  Other characters should should be
        eliminated (e.g. it's becomes its).  Leading and trailing spaces should be removed after
        punctuation is removed.

    Args:
        column (Column): A Column containing a sentence.

    Returns:
        Column: A Column named 'sentence' with clean-up operations applied.
    """
    nonletters = regexp_replace(column,'[^a-zA-Z0-9 ]+','')
    trimmed = trim(nonletters)
    final = lower(trimmed)
    return final
# to use this function
(sentenceDF
 .select(removePunctuation(col('sentence')))
 .show(truncate=False))
 
###############################################################
###															###
###															###
###   					A REGEX FUNCTION					###
###															###
###															###
###############################################################

''' sample data as follows
in24.inetnebr.com - - [01/Aug/1995:00:00:01 -0400] "GET /shuttle/missions/sts-68/news/sts-68-mcc-05.txt HTTP/1.0" 200 1839     
uplherc.upl.com - - [01/Aug/1995:00:00:07 -0400] "GET / HTTP/1.0" 304 0                                                        
uplherc.upl.com - - [01/Aug/1995:00:00:08 -0400] "GET /images/ksclogo-medium.gif HTTP/1.0" 304 0                               
uplherc.upl.com - - [01/Aug/1995:00:00:08 -0400] "GET /images/MOSAIC-logosmall.gif HTTP/1.0" 304 0 
'''

from pyspark.sql.functions import split, regexp_extract
split_df = base_df.select(regexp_extract('value', r'^([^\s]+\s)', 1).alias('host'),
                          regexp_extract('value', r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]', 1).alias('timestamp'),
                          regexp_extract('value', r'^.*"\w+\s+([^\s]+)\s+HTTP.*"', 1).alias('path'),
                          regexp_extract('value', r'^.*"\s+([^\s]+)', 1).cast('integer').alias('status'),
                          regexp_extract('value', r'^.*\s+(\d+)$', 1).cast('integer').alias('content_size'))
split_df.show(truncate=False)

# go to https://regex101.com/#python for help

###############################################################
###															###
###															###
###   		FINDING THE NUMBER OF NANs PER COLUMN			###
###															###
###															###
###############################################################


from pyspark.sql.functions import col, sum

def count_null(col_name):
  return sum(col(col_name).isNull().cast('integer')).alias(col_name)

# Build up a list of column expressions, one per column.
#
# This could be done in one line with a Python list comprehension, but we're keeping
# it simple for those who don't know Python very well.
exprs = [] #exprs
for col_name in split_df.columns:
  exprs.append(count_null(col_name))

# Run the aggregation. The *exprs converts the list of expressions into
# variable function arguments.
split_df.agg(*exprs).show()

###############################################################
###															###
###															###
###   				DATE CONVERSION FUNCTION				###
###															###
###															###
###############################################################

# Parsing time from logs example
month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(s):
    """ Convert Common Log time format into a Python datetime object
    Args:
        s (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring time zone here. In a production application, you'd want to handle that.
	# for more information on format go to https://pyformat.info/ '{:04d}'.format(42) returns 0042
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(s[7:11]),
      month_map[s[3:6]],
      int(s[0:2]),
      int(s[12:14]),
      int(s[15:17]),
      int(s[18:20])
    )

u_parse_time = udf(parse_clf_time)

logs_df = cleaned_df.select('*', u_parse_time(split_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
total_log_entries = logs_df.count()

# Extracting day of month from time. Additional information at https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.current_date
day_to_host_pair_df = logs_df.select('host',dayofmonth('time').alias('day'))

###############################################################
###															###
###															###
###   				BASIC STATS on DATAFRAMES				###
###															###
###															###
###############################################################

summary_stats_df = Original_df.describe(['column_u_care_for']) #using describe() function

###############################################################
###															###
###															###
###   					PARKING LOT							###
###															###
###															###
###############################################################

myDF = OriginalDf.select(col('Any_Column').isNotNull().cast('integer')) #isNotNull() checks every element in the column for Null and returns True if not null.
#.cast('integer') converts that boolean into 1 (for true) or 0


from spark_notebook_helpers import printDataFrames

#This function returns all the DataFrames in the notebook and their corresponding column names.
printDataFrames(True)