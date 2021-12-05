from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
import re
import f2,model


sc=SparkContext.getOrCreate()
ssc=StreamingContext(sc,1)
spark=SparkSession(sc)
sc.setLogLevel("OFF")

try:
	record=ssc.socketTextStream('localhost',6100)
except Exception as e:
	print(e)

def readstream(rdd):
	try:
		df=spark.read.json(rdd)
	except Exception as e:
		print(e)
	newcols = [col(column).alias(re.sub('\s*', '', column)) \
	for column in df.columns]
	
	
	try:
		df1=f2.f1(df)
		df2=f2.f2(df1)
		df3=f2.lemma(df2)
		df4=f2.text_preprocess(df3)
		df4.select("combined_F","label").show(truncate=False)
	except Exception as e:
		print(e)
		
	try:
		#model.f1(df4)
	except Exception as e:
		print(e)
		
		
		
record.foreachRDD(lambda x:readstream(x))


		
	

ssc.start()             
ssc.awaitTermination()  