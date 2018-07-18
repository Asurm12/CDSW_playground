from pyspark.sql import SparkSession
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,HashingTF,IDF
from pyspark.sql.functions import udf,explode,size
from pyspark.sql.types import FloatType,IntegerType
spark = SparkSession.builder.appName("TU-1").getOrCreate()

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

p_train=pd.DataFrame({'data':train.data,'target':train.target,'filenames':train.filenames})
p_test=pd.DataFrame({'data':test.data,'target':test.target,'filenames':test.filenames})

s_train = spark.createDataFrame(p_train)
s_test = spark.createDataFrame(p_test)

tokenizer = RegexTokenizer(inputCol='data',outputCol='words',pattern='\\W')
termFreq = HashingTF(inputCol='words',outputCol='freq')

pipeline = Pipeline(stages=[tokenizer,termFreq])
model = pipeline.fit(s_train)
data = model.transform(s_train)

def v_max(vector):
  return max(vector.toArray())
    
udf_v_max = udf(v_max,FloatType())
slen = udf(lambda s: s[0], IntegerType())
data.select(data.freq).rdd.map(lambda x: x.freq.toArray().argmax()).first()
data.first()
def show():
  plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
  lim = plt.axis()
  plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
  plt.axis(lim);                                   
                                   