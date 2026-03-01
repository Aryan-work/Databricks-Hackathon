# Databricks notebook source
import pandas as pd

# Step 1 - download the FIPS reference table
fips_url = "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
fips_df = pd.read_csv(fips_url)

# Step 2 - clean FIPS table
fips_df = fips_df.dropna(subset=["state"])
fips_df["county"] = fips_df["name"].str.upper()\
    .str.replace(" COUNTY", "", regex=False)\
    .str.replace(" PARISH", "", regex=False)\
    .str.replace(".", "", regex=False)\
    .str.strip()
fips_df["state_abbr"] = fips_df["state"].str.upper().str.strip()

# Step 3 - read flagged anomalies
df = spark.read.table("flagged_anomalies").toPandas()
df["county"] = df["county"].str.upper().str.replace(".", "", regex=False).str.strip()
df["state_abbr"] = df["state_abbr"].str.upper().str.strip()

# Step 4 - join anomalies with FIPS
df_enriched = pd.merge(
    df,
    fips_df[["county", "state_abbr", "fips"]],
    on=["county", "state_abbr"],
    how="left"
)

# Step 5 - check match rate
matched = df_enriched["fips"].notna().sum()
total = len(df_enriched)
print(f"Matched: {matched}/{total} ({matched/total*100:.1f}%)")

unmatched = df_enriched[df_enriched["fips"].isna()][["state_abbr","county"]].drop_duplicates()
print(f"Unmatched counties ({len(unmatched)}):")
print(unmatched.to_string())

# Fix FIPS zero padding
df_enriched["fips"] = df_enriched["fips"].fillna(0).astype(int).astype(str).str.zfill(5)
print("\nSample FIPS:", df_enriched["fips"].head(5).tolist())

# Save flagged_anomalies_geo
spark_df = spark.createDataFrame(df_enriched)
spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("flagged_anomalies_geo")
print("Saved: flagged_anomalies_geo delta table")

# Step 6 - county summary
county_df = spark.read.table("county_summary").toPandas()
county_df["county"] = county_df["county"].str.upper().str.replace(".", "", regex=False).str.strip()

# Build state → state_abbr lookup from enriched data
state_lookup = df_enriched[["state","state_abbr"]].drop_duplicates().set_index("state")["state_abbr"]
county_df["state_abbr"] = county_df["state"].map(state_lookup)

# Join county summary with FIPS
county_df = pd.merge(
    county_df,
    fips_df[["county", "state_abbr", "fips"]],
    on=["county", "state_abbr"],
    how="left"
)

# Check match rate
matched2 = county_df["fips"].notna().sum()
total2 = len(county_df)
print(f"\nCounty summary matched: {matched2}/{total2} ({matched2/total2*100:.1f}%)")

# Fix FIPS zero padding
county_df["fips"] = county_df["fips"].fillna(0).astype(int).astype(str).str.zfill(5)
print("Sample county FIPS:", county_df["fips"].head(5).tolist())

# Save county_summary_geo
spark_df2 = spark.createDataFrame(county_df)
spark_df2.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("county_summary_geo")
print("Saved: county_summary_geo delta table")

# COMMAND ----------

# DBTITLE 1,Year Filter SQL Parameter


# COMMAND ----------

# DBTITLE 1,US County Choropleth Map Data


# COMMAND ----------

# DBTITLE 1,Top 10 Most Stress-Impacted Counties


# COMMAND ----------

