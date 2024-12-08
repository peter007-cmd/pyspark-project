from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Initialize Spark session
spark = SparkSession.builder.appName('Scaler Preparation').getOrCreate()

# Load and preprocess the training data
sdf = spark.read.csv('./SouthGermanCredit.csv', header=True, inferSchema=True)
sdf = sdf.drop('_c0')  # Drop unnecessary column
numerical_cols = ['age', 'duration', 'amount']

# Vectorize numerical features
assembler = VectorAssembler(inputCols=numerical_cols, outputCol='features')
assembled_data = assembler.transform(sdf)

# Standardize the features
scaler = StandardScaler(inputCol='features', outputCol='standardized', withStd=True, withMean=True)
scaler_model = scaler.fit(assembled_data)

# Save the scaler model
scaler_model.write().overwrite().save('./scaler_model')

print("Scaler model saved successfully.")
