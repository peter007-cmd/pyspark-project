from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.clustering import KMeansModel
import pyspark.sql.functions as F
from kafka import KafkaProducer
import json

spark = SparkSession.builder \
    .appName('Real-Time Customer Segmentation') \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1") \
    .getOrCreate()

kmeans_model = KMeansModel.load("./kmeans_model")
scaler_model = StandardScalerModel.load("./scaler_model")

schema = StructType([
    StructField("duration", DoubleType(), True),
    StructField("age", IntegerType(), True),
    StructField("amount", DoubleType(), True)
])

df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "customer_data") \
    .load()

df_values = df_stream.selectExpr("CAST(value AS STRING)")
df_parsed = df_values.select(
    F.split(F.col("value"), ",").alias("fields")
).select(
    F.col("fields")[0].cast(DoubleType()).alias("duration"),
    F.col("fields")[1].cast(IntegerType()).alias("age"),
    F.col("fields")[2].cast(DoubleType()).alias("amount")
)

df_parsed = df_parsed.withColumn("log_duration", F.log(F.col("duration")))
df_parsed = df_parsed.withColumn("log_age", F.log(F.col("age")))
df_parsed = df_parsed.withColumn("log_amount", F.log(F.col("amount")))

features = ["log_duration", "log_age", "log_amount"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled_data = assembler.transform(df_parsed)
scaled_data = scaler_model.transform(assembled_data)
predictions = kmeans_model.transform(scaled_data)

prediction_output = predictions.select("duration", "age", "amount", "prediction")

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def send_predictions(batch_df, epoch_id):
    data = batch_df.collect()
    for row in data:
        message = {
            "duration": row["duration"],
            "age": row["age"],
            "amount": row["amount"],
            "prediction": row["prediction"]
        }
        producer.send("predicted_data", value=message)

query = prediction_output.writeStream \
    .outputMode("update") \
    .foreachBatch(send_predictions) \
    .start()

query.awaitTermination()
