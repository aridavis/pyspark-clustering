from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt

spark = SparkSession.builder.getOrCreate()

#Load Train
df_train = spark.read.option("inferSchema", "true").csv("Training.csv", header=True)
df_train = df_train.select("Algae Concentration", "Oil Concentration", "Trash Pollution")
df_train = df_train.na.drop()
df_train = df_train.withColumn("Trash Pollution", when(df_train["Trash Pollution"] == "Low", 0).
                                                  when(df_train["Trash Pollution"] == "Medium", 1).
                                                  when(df_train["Trash Pollution"] == "High", 2))
cols = df_train.columns
df_train = VectorAssembler(inputCols = cols, outputCol = "Vector").transform(df_train)

scaler = StandardScaler(inputCol = "Vector", outputCol = "features")
df_train = scaler.fit(df_train).transform(df_train)

#Load Test
df_test = spark.read.option("inferSchema", "true").csv("Testing.csv", header=True)
df_test = df_test.select("Algae Concentration", "Oil Concentration", "Trash Pollution", "Polluted")
df_test = df_test.na.drop()
df_test = df_test.withColumn("Trash Pollution", when(df_test["Trash Pollution"] == "Low", 0).
                                                when(df_test["Trash Pollution"] == "Medium", 1).
                                                when(df_test["Trash Pollution"] == "High", 2))
df_test = df_test.withColumn("Polluted", when(df_test["Polluted"] == "No", 0).
                                         when(df_test["Polluted"] == "Yes", 1))
cols = df_test.columns
cols.remove("Polluted")
df_test = VectorAssembler(inputCols = cols, outputCol = "Vector").transform(df_test)

scaler = StandardScaler(inputCol = "Vector", outputCol = "features")
df_test = scaler.fit(df_test).transform(df_test)

kmeans = KMeans().setK(2)
model = kmeans.fit(df_train)
# print(model.clusterCenters())

#Testing
predictions = model.transform(df_test)

#Visualization
predictions = predictions.toPandas()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(predictions["Algae Concentration"], predictions["Oil Concentration"], c=predictions["prediction"])
ax.set_title('Relationship Between Algae Concentration and Oil Concentration in Cluster Prediction')
ax.set_xlabel('Algae Concentration')
ax.set_ylabel('Oil Concentration')
plt.show()

#Print Accuracy
c = 0
for index, row in predictions.iterrows():
    if row["Polluted"] == row["prediction"]:
        c = c + 1

print("Accuracy: {}%".format(c / len(predictions) * 100))