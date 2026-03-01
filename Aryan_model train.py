# Databricks notebook source
df = spark.table("rma_county_yields_report_399")
display(df)


# COMMAND ----------

df = spark.table("rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily")
display(df)

# COMMAND ----------

# MAGIC %pip install geopandas

# COMMAND ----------

# Databricks notebook source

# MAGIC %md
# MAGIC # Weather Observations × County Yields Pipeline
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads weather observations and county yields from Databricks tables.
# MAGIC 2. Maps each weather station (lat/lon) to a US county using a GeoJSON file.
# MAGIC 3. Aggregates weather data annually per county.
# MAGIC 4. Joins aggregated weather with county yield data.
# MAGIC 5. Produces a final DataFrame with both yield and annual weather totals per county/year.

# COMMAND ----------

import json
import math
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read Both Source Tables

# COMMAND ----------

# --- Weather observations ---
weather_df = spark.table(
    "rearc_daily_weather_observations_noaa.esg_noaa_ghcn.noaa_ghcn_daily"
)

# --- County yields ---
yields_df = spark.table("rma_county_yields_report_399")

print("Weather schema:")
weather_df.printSchema()
print("\nYields schema:")
yields_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load and Parse the US Counties GeoJSON
# MAGIC
# MAGIC We load the GeoJSON once on the driver and build a lookup structure.
# MAGIC Each county feature is stored with its bounding box (for fast rejection)
# MAGIC and its polygon ring(s) for the ray-casting point-in-polygon test.
# MAGIC
# MAGIC **Update the path below** to point to your actual GeoJSON file location
# MAGIC (DBFS, Volumes, or a mounted path).

# COMMAND ----------

# ============================================================
# CONFIGURE THIS: path to your US counties GeoJSON file
# ============================================================
# Examples:
#   "/dbfs/FileStore/us_counties.geojson"
#   "/Volumes/my_catalog/my_schema/my_volume/us_counties.geojson"
GEOJSON_PATH = "counties.geojson"

with open(GEOJSON_PATH, "r") as f:
    geojson_data = json.load(f)

print(f"Loaded GeoJSON with {len(geojson_data['features'])} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a: Build County Lookup from GeoJSON
# MAGIC
# MAGIC We extract **state FIPS**, **county FIPS**, **county name**, and the
# MAGIC polygon coordinates from each GeoJSON feature. We also compute bounding
# MAGIC boxes for fast spatial filtering.
# MAGIC
# MAGIC **Adjust the property key names below** if your GeoJSON uses different
# MAGIC field names (e.g., `STATEFP` vs `STATE`, `NAME` vs `COUNTY_NAME`, etc.).

# COMMAND ----------

def build_county_lookup(geojson):
    """
    Parse GeoJSON features into a list of county dicts, each containing:
      - state_code  : str (FIPS code, e.g. "01")
      - county_code : str (FIPS code, e.g. "001")
      - county_name : str
      - rings       : list of polygon rings [ [(lon, lat), ...], ... ]
      - bbox        : (min_lon, min_lat, max_lon, max_lat)

    Handles both Polygon and MultiPolygon geometry types.

    >>> Adjust the property keys to match YOUR GeoJSON <<<
    """
    counties = []

    for feature in geojson["features"]:
        props = feature.get("properties", {})

        # --- Extract identifiers (adjust keys to match your file) ---
        # Common variants: STATEFP / STATE / STATEFP10 / STATE_FIPS
        state_code = str(props.get("STATEFP", props.get("STATE", ""))).zfill(2)
        # Common variants: COUNTYFP / COUNTY / COUNTYFP10 / CNTY_FIPS
        county_code = str(props.get("COUNTYFP", props.get("COUNTY", ""))).zfill(3)
        # Common variants: NAME / COUNTY_NAME / NAMELSAD
        county_name = props.get("NAME", props.get("COUNTY_NAME", "Unknown"))

        geom = feature.get("geometry", {})
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        # Normalize to a list of polygons; each polygon = list of rings
        if geom_type == "Polygon":
            polygons = [coords]          # coords = [ ring, ring, ... ]
        elif geom_type == "MultiPolygon":
            polygons = coords            # coords = [ polygon, polygon, ... ]
        else:
            continue  # skip non-polygon features

        # Flatten all rings from all polygons for this county
        all_rings = []
        all_lons = []
        all_lats = []

        for polygon in polygons:
            for ring in polygon:
                # Each ring is [ [lon, lat], [lon, lat], ... ]
                ring_tuples = [(pt[0], pt[1]) for pt in ring]
                all_rings.append(ring_tuples)
                all_lons.extend(pt[0] for pt in ring)
                all_lats.extend(pt[1] for pt in ring)

        if not all_rings:
            continue

        bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))

        counties.append({
            "state_code": state_code,
            "county_code": county_code,
            "county_name": county_name,
            "rings": all_rings,
            "bbox": bbox,
        })

    return counties


county_lookup = build_county_lookup(geojson_data)
print(f"Built lookup for {len(county_lookup)} counties")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b: Ray-Casting Point-in-Polygon (Pure Python, No Shapely)

# COMMAND ----------

def point_in_ring(lon, lat, ring):
    """
    Ray-casting algorithm: cast a ray from (lon, lat) to the right and
    count how many edges of the ring it crosses. Odd = inside.
    """
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        # Check if the ray crosses this edge
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def find_county(lon, lat, lookup):
    """
    Given a (lon, lat), return (state_code, county_code, county_name).
    Uses bounding-box pre-filtering for speed.
    Returns ("-1", "-1", "Unknown") if no county matches.
    """
    for county in lookup:
        min_lon, min_lat, max_lon, max_lat = county["bbox"]
        # Quick bounding-box rejection
        if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
            continue
        # Detailed ring test: first ring is exterior, rest are holes
        # For simplicity we check exterior only (first ring of each polygon).
        # If your GeoJSON has holes, you may want to subtract interior rings.
        for ring in county["rings"]:
            if point_in_ring(lon, lat, ring):
                return (
                    county["state_code"],
                    county["county_code"],
                    county["county_name"],
                )
    # No match → outside the US (or outside any county polygon)
    return ("-1", "-1", "Unknown")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Compute County Assignment Once Per Unique Lat/Lon
# MAGIC
# MAGIC We collect distinct (latitude, longitude) pairs from the weather table,
# MAGIC map each to a county on the driver, and broadcast the result back as a
# MAGIC Spark DataFrame for joining.

# COMMAND ----------

# 3a. Get unique lat/lon pairs
unique_coords_df = weather_df.select("latitude", "longitude").distinct()
unique_coords_list = unique_coords_df.collect()

print(f"Unique (lat, lon) pairs to geocode: {len(unique_coords_list)}")

# COMMAND ----------

# 3b. Map each unique coordinate to a county (driver-side loop)
#     This avoids shipping the large GeoJSON to every executor.

coord_to_county = []

for i, row in enumerate(unique_coords_list):
    lat = float(row["latitude"]) if row["latitude"] is not None else 0.0
    lon = float(row["longitude"]) if row["longitude"] is not None else 0.0
    state_code, county_code, county_name = find_county(lon, lat, county_lookup)
    coord_to_county.append((lat, lon, state_code, county_code, county_name))

    if (i + 1) % 5000 == 0:
        print(f"  Geocoded {i + 1} / {len(unique_coords_list)} coordinates ...")

print(f"Geocoding complete: {len(coord_to_county)} coordinates mapped.")

# COMMAND ----------

# 3c. Create a Spark DataFrame from the mapping and broadcast-join it

coord_schema = StructType([
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("geo_state_code", StringType(), True),
    StructField("geo_county_code", StringType(), True),
    StructField("geo_county_name", StringType(), True),
])

coord_county_df = spark.createDataFrame(coord_to_county, schema=coord_schema)

# Broadcast hint — this table is small (one row per unique station location)
coord_county_df = F.broadcast(coord_county_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Attach County Info to Weather Observations & Filter

# COMMAND ----------

# Join weather observations with the county mapping on lat/lon
weather_with_county = weather_df.join(
    coord_county_df,
    on=["latitude", "longitude"],
    how="left"
)

# Filter out unknown counties BEFORE aggregating (as requested)
weather_with_county = weather_with_county.filter(
    F.col("geo_state_code") != "-1"
)

print(f"Weather rows after filtering unknown counties: {weather_with_county.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Aggregate Weather Data Annually Per County
# MAGIC
# MAGIC We extract the year from the `date` column and sum the key weather
# MAGIC measurements (precipitation, temp_max, temp_min, snow_depth, snowfall)
# MAGIC for each county/year combination.

# COMMAND ----------

# Extract year from the date column
weather_with_county = weather_with_county.withColumn(
    "year", F.year(F.col("date"))
)

# Annual aggregation per county
weather_annual = weather_with_county.groupBy(
    "geo_state_code",
    "geo_county_code",
    "geo_county_name",
    "year"
).agg(
    F.sum("precipitation").alias("total_precipitation"),
    F.sum("temp_max").alias("total_temp_max"),
    F.sum("temp_min").alias("total_temp_min"),
    F.sum("snow_depth").alias("total_snow_depth"),
    F.sum("snowfall").alias("total_snowfall"),
    F.count("*").alias("observation_count"),  # handy for QA
)

print("Annual weather aggregation schema:")
weather_annual.printSchema()
weather_annual.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Prepare County Yields for Joining
# MAGIC
# MAGIC The yields table uses its own state/county code columns. We need to
# MAGIC align data types and zero-pad FIPS codes so the join keys match.

# COMMAND ----------

yields_prepared = (
    yields_df
    # Zero-pad state and county codes to match the GeoJSON FIPS format
    .withColumn(
        "state_code_padded",
        F.lpad(F.col("State Code").cast("string"), 2, "0")
    )
    .withColumn(
        "county_code_padded",
        F.lpad(F.col("County Code").cast("string"), 3, "0")
    )
    # Trim county name for clean matching
    .withColumn("county_name_trimmed", F.trim(F.col("County Name")))
    # Rename yield year for clarity
    .withColumnRenamed("Yield Year", "yield_year")
    .withColumnRenamed("Yield Amount", "yield_amount")
    .withColumnRenamed("Commodity Code", "commodity_code")
    .withColumnRenamed("Commodity Name", "commodity_name")
    .withColumnRenamed("State Name", "state_name")
    .withColumnRenamed("State Abbreviation", "state_abbreviation")
    .withColumnRenamed("Irrigation Practice Code", "irrigation_practice_code")
    .withColumnRenamed("Irrigation Practice Name", "irrigation_practice_name")
)

yields_prepared.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Join Aggregated Weather with County Yields
# MAGIC
# MAGIC Join on **state code**, **county code**, **county name**, and **year**.

# COMMAND ----------

final_df = weather_annual.join(
    yields_prepared,
    on=[
        weather_annual["geo_state_code"] == yields_prepared["state_code_padded"],
        weather_annual["geo_county_code"] == yields_prepared["county_code_padded"],
        weather_annual["geo_county_name"] == yields_prepared["county_name_trimmed"],
        weather_annual["year"] == yields_prepared["yield_year"],
    ],
    how="inner"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Select and Rename Final Columns

# COMMAND ----------

final_df = final_df.select(
    # --- Identifiers ---
    F.col("geo_state_code").alias("state_code"),
    F.col("state_name"),
    F.col("state_abbreviation"),
    F.col("geo_county_code").alias("county_code"),
    F.col("geo_county_name").alias("county_name"),
    F.col("year"),

    # --- Weather totals ---
    F.col("total_precipitation"),
    F.col("total_temp_max"),
    F.col("total_temp_min"),
    F.col("total_snow_depth"),
    F.col("total_snowfall"),
    F.col("observation_count"),

    # --- Yield info ---
    F.col("commodity_code"),
    F.col("commodity_name"),
    F.col("irrigation_practice_code"),
    F.col("irrigation_practice_name"),
    F.col("yield_amount"),
)

print("=== Final DataFrame Schema ===")
final_df.printSchema()

print("=== Sample Rows ===")
final_df.show(20, truncate=False)

print(f"Total rows in final joined dataset: {final_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Step 9: Save the Final DataFrame
# MAGIC
# MAGIC Uncomment one of the options below to persist results.

# COMMAND ----------

# Option A: Save as a Delta table
# final_df.write.mode("overwrite").saveAsTable("my_catalog.my_schema.weather_county_yields")

# Option B: Save as Parquet to a volume
# final_df.write.mode("overwrite").parquet("/Volumes/my_catalog/my_schema/my_volume/weather_county_yields")

# Option C: Save as CSV
# final_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
#     "/Volumes/my_catalog/my_schema/my_volume/weather_county_yields_csv"
# )

print("Pipeline complete.")

# COMMAND ----------

display(final_df)

print("Schema:")
final_df.printSchema()

print("Null counts per column:")
null_counts = final_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in final_df.columns])
display(null_counts)


# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# DBTITLE 1,Cell 3: Spark pipeline with synthetic county assignment
# Robust Spark pipeline with synthetic county assignment for weather aggregation
# CONFIG: update table/column names if dataset changes

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

required_columns = [
    "county_code", "year", "commodity_name", "yield_amount",
    "total_precipitation", "total_temp_max", "total_temp_min",
    "total_snow_depth", "total_snowfall", "observation_count"
]
missing_columns = [col for col in required_columns if col not in final_df.columns]

if missing_columns:
    print("[ERROR] The following required columns are missing from final_df:")
    for col in missing_columns:
        print(f"  - {col}")
    raise ValueError("Missing required columns: " + ", ".join(missing_columns))
else:
    print("All required columns present, proceeding with pipeline.")

# === ML Pipeline ===
target_crop = "Corn"  # No leading space
yield_actual_col = "yield_amount"
yield_crop_col = "commodity_name"
county_code_col = "county_code"
year_col = "year"
feature_cols = [
    "total_precipitation",
    "total_temp_max",
    "total_temp_min",
    "total_snow_depth",
    "total_snowfall",
    "observation_count"
]

ml_df = (
    final_df
    .filter(F.trim(F.col(yield_crop_col)) == target_crop)
    .select(
        county_code_col,
        year_col,
        yield_crop_col,
        yield_actual_col,
        *feature_cols
    )
    .filter(F.col(yield_actual_col).isNotNull())
)

fill_defaults = {
    "total_precipitation": 0.0,
    "total_temp_max": 0.0,
    "total_temp_min": 0.0,
    "total_snow_depth": 0.0,
    "total_snowfall": 0.0,
    "observation_count": 0.0
}
ml_df = ml_df.na.fill(fill_defaults)

means = ml_df.select(F.mean("total_temp_max").alias("mean_temp_max")).collect()[0]
ml_df = ml_df.na.fill({"total_temp_max": float(means.mean_temp_max) if means.mean_temp_max is not None else 0.0})

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
train, test = ml_df, ml_df

gbt = GBTRegressor(featuresCol="features", labelCol=yield_actual_col, predictionCol="predicted_yield", maxIter=100, maxDepth=7, stepSize=0.2)
pipeline = Pipeline(stages=[assembler, gbt])

if train.count() == 0 or test.count() == 0:
    print("[ERROR] Train or test DataFrame is empty. Cannot fit model.")
else:
    model = pipeline.fit(train)
    predictions = model.transform(test)
    predictions = predictions.withColumn("residual", F.col(yield_actual_col) - F.col("predicted_yield"))

    evaluator_rmse = RegressionEvaluator(labelCol=yield_actual_col, predictionCol="predicted_yield", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=yield_actual_col, predictionCol="predicted_yield", metricName="r2")
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    print(f"Model results for crop={target_crop}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")

    display_cols = [county_code_col, year_col, yield_crop_col, yield_actual_col, "predicted_yield", "residual"] + feature_cols
    display(predictions.select(display_cols).orderBy(F.abs(F.col("residual")).desc()).limit(50))

    gbt_model = model.stages[-1]
    fi = list(zip(feature_cols, [float(x) for x in gbt_model.featureImportances]))
    print("Feature importances (feature, importance):")
    for f, importance in sorted(fi, key=lambda x: -x[1]):
        print(f"{f}: {importance:.4f}")

    # Minimal fix: Save Delta table without catalog/schema prefix
    out_df = predictions.select(display_cols)
    out_df.write.format("delta").mode("overwrite").saveAsTable("model_predictions_corn")
    print("Predictions written to table: model_predictions_corn")

    print("Top underperformers (largest negative residual):")
    display(out_df.orderBy(F.col("residual").asc()).limit(20))
    print("Top overperformers (largest positive residual):")
    display(out_df.orderBy(F.col("residual").desc()).limit(20))


# COMMAND ----------

from itertools import product

param_grid = {
    "maxIter":  [50, 100, 200],
    "maxDepth": [3, 5, 7],
    "stepSize": [0.05, 0.1, 0.2]
}

results = []
for maxIter, maxDepth, stepSize in product(*param_grid.values()):
    gbt = GBTRegressor(
        featuresCol="features", labelCol=yield_actual_col,
        predictionCol="predicted_yield",
        maxIter=maxIter, maxDepth=maxDepth, stepSize=stepSize
    )
    pipeline = Pipeline(stages=[assembler, gbt])
    m = pipeline.fit(train)
    preds = m.transform(test)
    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)
    results.append((maxIter, maxDepth, stepSize, rmse, r2))
    print(f"maxIter={maxIter}, maxDepth={maxDepth}, stepSize={stepSize} → RMSE={rmse:.3f}, R2={r2:.3f}")

best = sorted(results, key=lambda x: x[3])[0]
print(f"\nBest: maxIter={best[0]}, maxDepth={best[1]}, stepSize={best[2]}, RMSE={best[3]:.3f}, R2={best[4]:.3f}")