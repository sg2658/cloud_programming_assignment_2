import sys

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def clean_data(data_frame):
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

if __name__ == "__main__":
    print("Starting Spark Application in EMR")


    spark_session = SparkSession.builder.appName("Wine-Quality-Prediction-SPARK-ML").getOrCreate()

    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    spark_session._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")


    training_data_path = "TrainingDataset.csv"
    model_output_path = "trainedmodel"

    print(f"Reading training CSV file from {training_data_path}")
    raw_data_frame = (spark_session.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(training_data_path))
    
    training_data_frame = clean_data(raw_data_frame)

    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol', 'quality']

    print("Creating VectorAssembler")
    features_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    
    print("Creating StringIndexer")
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    print("Caching data for faster access")
    training_data_frame.cache()
    
    print("Creating RandomForestClassifier")
    random_forest_classifier = RandomForestClassifier(labelCol='label', 
                                                      featuresCol='features',
                                                      numTrees=150,
                                                      maxDepth=15,
                                                      seed=150,
                                                      impurity='gini')
    
    print("Creating Pipeline for training")
    training_pipeline = Pipeline(stages=[features_assembler, label_indexer, random_forest_classifier])
    fitted_model = training_pipeline.fit(training_data_frame)

    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                                           predictionCol='prediction', 
                                                           metricName='accuracy')

    print("Retraining model on multiple parameters using CrossValidator")
    cv_model = None
    parameter_grid = ParamGridBuilder() \
        .addGrid(random_forest_classifier.maxDepth, [6, 9]) \
        .addGrid(random_forest_classifier.numTrees, [50, 150]) \
        .addGrid(random_forest_classifier.minInstancesPerNode, [6]) \
        .addGrid(random_forest_classifier.seed, [100, 200]) \
        .addGrid(random_forest_classifier.impurity, ["entropy", "gini"]) \
        .build()
    
    cv_pipeline = CrossValidator(estimator=training_pipeline,
                                 estimatorParamMaps=parameter_grid,
                                 evaluator=accuracy_evaluator,
                                 numFolds=2)

    print("Fitting CrossValidator to the training data")
    best_model = cv_pipeline.fit(training_data_frame)
    
    print("Saving the best model to new param `model`")
    final_model = best_model.bestModel

    print("Saving the best model to S3")
    final_model_path = model_output_path
    final_model.write().overwrite().save(final_model_path)
    spark_session.stop()