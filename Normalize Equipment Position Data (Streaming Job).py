# Databricks notebook source
# MAGIC %md ### Config Variables

# COMMAND ----------

input_table = "hive_metastore.suncor_geospatial_poc.mlleqmtposjan15_streaming"
output_table = "hive_metastore.suncor_geospatial_poc_silver.equip_pos_jan_15" 
h3_resolution ="15"

# COMMAND ----------

# MAGIC %md Define Projections

# COMMAND ----------

import pyproj

fh_mod = pyproj.Proj(
    "+proj=tmerc +lat_0=0 +lon_0=-111 +k=0.9996 +x_0=500001.412 +y_0=-0.608 +datum=NAD83 +units=m +no_defs",
    preserve_units=False,
)
osb_fr = pyproj.Proj(
    "+proj=tmerc +lat_0=57.0046641 +lon_0=-111.3466946 +k=1.000558715 +x_0=5058.909318 +y_0=3354.476738 +datum=NAD83 +units=mm +no_defs",
    preserve_units=False,
)
wgs84 = pyproj.Proj(proj="latlong", datum="WGS84", ellps="WGS84")

# COMMAND ----------

# MAGIC %md Enable Async Checkpointing

# COMMAND ----------

spark.conf.set(
  "spark.databricks.streaming.statefulOperator.asyncCheckpoint.enabled",
  "true"
)

spark.conf.set(
  "spark.sql.streaming.stateStore.providerClass",
  "com.databricks.sql.streaming.state.RocksDBStateStoreProvider"
)

# COMMAND ----------

# MAGIC %md ### Utility Functions

# COMMAND ----------

#Enable Arrow for better UDF performance
spark.conf.set("spark.databricks.execution.pythonUDF.arrow.enabled", "true")

# COMMAND ----------

from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StructType, StructField, DoubleType
import pyproj
import mosaic as mos

# Define the UDF that returns a struct with two double fields
@udf(
    StructType(
        [StructField("lat", DoubleType(), True), StructField("lon", DoubleType(), True)]
    )
)
def utm2wgs84(easting, northing):
    lon, lat = pyproj.transform(osb_fr, wgs84, easting, northing)
    # Return the struct with the two fields
    return (lat, lon)


# Convert UTM to WGS 64
def convert_to_wgs84(df, eastings, northings):
    return (
        df.withColumn("wgs84_latlon", utm2wgs84(eastings, northings))
        .withColumn("lat", col("wgs84_latlon.lat"))
        .withColumn("lon", col("wgs84_latlon.lon"))
        .drop("wgs84_latlon")
    )


# Add H3 Index
def add_h3_index(df, lon_col="lon", lat_col="lat", res=h3_resolution):
    spark.conf.set("spark.databricks.labs.mosaic.geometry.api", "JTS")
    spark.conf.set("spark.databricks.labs.mosaic.index.system", "H3")

    mos.enable_mosaic(spark, dbutils)

    return df.withColumn("point_geom", mos.st_point(lon_col, lat_col)).withColumn(
        "index", mos.grid_pointascellid("point_geom", resolution=lit(res))
    )

# COMMAND ----------

# MAGIC %md ### Streaming Code

# COMMAND ----------

streaming_df = (
    spark.readStream.format("delta")
    .option("maxFilesPerTrigger", 4)
    .table(input_table)
)

# Batch Testing
# df = spark.read \
#   .format("delta") \
#   .option("versionAsOf", "0") \
#   .table("hive_metastore.suncor_geospatial_poc.mlleqmtposjan15_streaming")
# display(df)

# COMMAND ----------

normalized_streaming_df = streaming_df.transform(
    convert_to_wgs84, "PositionX", "PositionY"
).transform(add_h3_index)

(
    normalized_streaming_df.writeStream.format("delta")
    .outputMode("append")
    .option(
        "checkpointLocation", "dbfs:/FileStore/checkpointDir/suncor_silver_equip_pos"
    )
    .toTable(output_table)
)
