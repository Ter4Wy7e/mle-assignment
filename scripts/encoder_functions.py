from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

def cap_outliers_enc(df, caps):
    for key in caps.keys():
        df = df.withColumn(key, F.when(F.col(key) > caps[key], caps[key]).otherwise(F.col(key)))
    return df

def fill_nulls_enc(df, fill_nulls_values):
    for key in fill_nulls_values.keys():
        df = df.withColumn(key, F.when(F.col(key).isNull(), fill_nulls_values[key]).otherwise(F.col(key)))
    return df

