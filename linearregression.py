%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
spark = SparkSession.builder.appName("TU-1").getOrCreate()

d_F = spark.read.csv('FremontBridge.csv',inferSchema=True,
                     header=True,timestampFormat='MM/dd/yyyy hh:mm:ss aa')
d_F = d_F.orderBy('date',ascending=True)

d_B = spark.read.csv('BicycleWeather.csv',
                     header=True,inferSchema=True,dateFormat = 'yyyyMMdd')
d_B = d_B.withColumn('DATE',to_date(d_B.DATE.cast('string'),format='yyyyMMdd')).orderBy('DATE',ascending=True)

daily = d_F.groupBy(to_date(d_F.Date).alias('Date')).sum().orderBy('Date')
daily = daily.withColumnRenamed('sum(Fremont Bridge East Sidewalk)','East').withColumnRenamed('sum(Fremont Bridge West Sidewalk)','West')
daily = daily.select('Date',(daily.East + daily.West).alias('total'))
daily = daily.select('Date','total',date_format(daily.Date,'EEEE').alias('day'))

