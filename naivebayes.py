from sklearn.metrics import confusion_matrix
import seaborn as sns
from pyspark.sql import SparkSession
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,HashingTF,IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import udf,explode,size
sns.set()
spark = SparkSession.builder.appName("TU-1").getOrCreate()

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

p_train=pd.DataFrame({'data':train.data,'target':train.target,'filenames':train.filenames})
p_test=pd.DataFrame({'data':test.data,'target':test.target,'filenames':test.filenames})

s_train = spark.createDataFrame(p_train)
s_test = spark.createDataFrame(p_test)

tokenizer = RegexTokenizer(inputCol='data',outputCol='words',pattern='\\W')
termFreq = HashingTF(inputCol='words',outputCol='freq')
idf = IDF(inputCol='freq',outputCol='tfidf')
nb = NaiveBayes(featuresCol="tfidf", labelCol="target")
pipeline = Pipeline(stages=[tokenizer,termFreq,idf,nb])
model = pipeline.fit(s_train)
data = model.transform(s_test)
p_data = data.sample(False,0.5).limit(500).toPandas()

mat = confusion_matrix(p_data.target,p_data.prediction)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
