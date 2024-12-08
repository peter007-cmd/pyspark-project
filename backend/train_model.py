from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName('Customer Segmentation Training').getOrCreate()

sdf = spark.read.csv('./SouthGermanCredit.csv', header=True, inferSchema=True)
sdf = sdf.drop('_c0') 
numerical_cols = ["duration", "age", "amount"]

for col in numerical_cols:
    sdf = sdf.withColumn("log_" + col, F.log(F.col(col)))

features = [f"log_{col}" for col in numerical_cols]
assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled_data = assembler.transform(sdf)

scaler = StandardScaler(inputCol="features", outputCol="standardized")
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

scaler_model.write().overwrite().save('./scaler_model')

kmeans = KMeans(featuresCol="standardized", k=3)  # Set the number of clusters
kmeans_model = kmeans.fit(scaled_data)

kmeans_model.write().overwrite().save('./kmeans_model')

# evaluator = ClusteringEvaluator(featuresCol="standardized", metricName="silhouette", distanceMeasure="squaredEuclidean")
# silhouette = evaluator.evaluate(kmeans_model.transform(scaled_data))
# print(f"Silhouette Score: {silhouette}")
