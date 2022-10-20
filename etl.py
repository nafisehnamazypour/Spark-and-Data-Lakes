import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format

from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date


config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

os.environ['AWS_ACCESS_KEY_ID']=config.get ('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS','AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    In this function a spark session is created.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads data from song_data dataset and creates a data base with Star Schema with extracting column data from it. 
    The star schema consists songs and artists tables.
    Finally, the data is written in parquete files to load in S3 bucket.
    """
    
    ###########################################################
    # get filepath to song data file
    song_data = os.path.join(input_data,"song_data/*/*/*/*.json")

    #song_data = get_files('data/song_data')     
    # read song data file
    df = spark.read.json(song_data)
    #df.printSchema()

    # extract columns to create songs table  columns = ['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = df.select('song_id', 'title', 'artist_id',
                            'year', 'duration') \
                    .dropDuplicates()
    songs_table.createOrReplaceTempView('songs')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").partitionBy("year", "artist_id").parquet(output_data + "songs")
      
    ###########################################################
    # extract columns to create artists table   columns = artist_id, name, location, lattitude, longitude
    artists_fields = ["artist_id", "artist_name as name","artist_location as location","artist_latitude as latitude",
                      "artist_longitude as longitude"]
    artists_table = df.selectExpr(artists_fields).dropDuplicates()
 
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data,'artists/artists.parquet'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    """
    Loads data from log_data dataset and creates a data base with Star Schema with extracting column data from it. Also, the song_data is read and data is extracted to creat songplays table.
    The star schema consists of users, time and songplays tables.
    Finally, the parquete files are generated to load into S3 bucket.
    """
    
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')
    
    # read log data file
    log_df =  spark.read.json(log_data)
    #df.printSchema()
    
    # filter by actions for song plays
    log_df = log_df.filter(log_df.page == 'NextSong') 
    
    ###########################################################
    # extract columns for users table   columns = user_id, first_name, last_name, gender, level 
      
    users_fields = ["userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"]
    users_table = log_df.selectExpr(users_fields).dropDuplicates()
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users.parquet'), 'overwrite')
 
    ###########################################################
    # create datetime column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    log_df = log_df.withColumn('timestamp', get_timestamp(log_df.ts))
    
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    log_df= log_df.withColumn('start_time', get_datetime(log_df.ts))
    
    # extract columns to create time table   columns = start_time, hour, day, week, month, year, weekday
    log_df = log_df.withColumn("hour", hour("start_time")) \
        .withColumn("day", dayofmonth("start_time")) \
        .withColumn("week", weekofyear("start_time")) \
        .withColumn("weekday", dayofweek("start_time"))\
        .withColumn("month", month("start_time")) \
        .withColumn("year", year("start_time")) 
       
    time_table = log_df.select("start_time", "hour", "day", "week", "weekday", "month", "year")

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month') \
                    .parquet(os.path.join(output_data,
                                          'time/time.parquet'), 'overwrite')

    
    ###########################################################   
    # read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data, "songs"))

    # extract columns from joined song and log datasets to create songplays table columns = songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    log_df = log_df.alias('log_df')
    song_df = song_df.alias('song_df')
    joined_df = log_df.join(song_df, song_df.title == log_df.song)
    songplays_table = joined_df.select(
        col('log_df.start_time').alias('start_time'),
        col('log_df.userId').alias('user_id'),
        col('log_df.level').alias('level'),
        col('song_df.song_id').alias('song_id'),
        col('song_df.artist_id').alias('artist_id'),
        col('log_df.sessionId').alias('session_id'),
        col('log_df.location').alias('location'), 
        col('log_df.userAgent').alias('user_agent'))\
        .withColumn("year", date_format(col("start_time"), "yyyy")) \
        .withColumn("month", date_format(col("start_time"), "MM")) \
        .withColumn('songplay_id', monotonically_increasing_id())
    
    
    time_table = time_table.alias('timetable')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data,
                                 'songplays/songplays.parquet'),
                                 'overwrite')


def main():
    """
    This is the main function which executes create_spark_session(), process_song_data() and process_log_data() functions.
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = ""
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
