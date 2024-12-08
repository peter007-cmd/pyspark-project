from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pyspark.sql.functions as F

spark = SparkSession.builder.appName('South German Credit - Custom Predictions').getOrCreate()

kmeans_model = KMeansModel.load("./kmeans_model")

custom_data = [
    {"duration": 24, "age": 70, "amount": 5000},
    {"duration": 36, "age": 45, "amount": 12000},
    {"duration": 12, "age": 28, "amount": 10000},
]
custom_sdf = spark.createDataFrame(custom_data)

numerical_cols = ["duration", "age", "amount"]  
for col in numerical_cols:
    custom_sdf = custom_sdf.withColumn("log_" + col, F.log(F.col(col)))

features_custom = [f"log_{col}" for col in numerical_cols]
assembler = VectorAssembler(inputCols=features_custom, outputCol="features")
custom_sdf_assembled = assembler.transform(custom_sdf)

training_data = spark.read.csv('./SouthGermanCredit.csv', header=True, inferSchema=True)
training_data = training_data.select(numerical_cols)
for col in numerical_cols:
    training_data = training_data.withColumn("log_" + col, F.log(F.col(col)))

features_training = [f"log_{col}" for col in numerical_cols]
assembler_training = VectorAssembler(inputCols=features_training, outputCol="features")
assembled_training_data = assembler_training.transform(training_data)

scaler = StandardScaler(inputCol="features", outputCol="standardized")
fitted_scaler = scaler.fit(assembled_training_data)

custom_sdf_scaled = fitted_scaler.transform(custom_sdf_assembled)

custom_predictions = kmeans_model.transform(custom_sdf_scaled)

custom_predictions.select("duration", "age", "amount", "prediction").show()
