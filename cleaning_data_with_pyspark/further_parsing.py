'''
Further parsing
You've molded this dataset into a significantly different format than it was before, but there are still a few things left to do.
You need to prep the column data for use in later analysis and remove a few intermediary columns.
The spark context is available and pyspark.sql.functions is aliased as F. The types from pyspark.sql.types are already imported. 
The split_df DataFrame is as you last left it.
Remember, you can use .printSchema() on a DataFrame in the console area to view the column names and types.

'''

def retriever(cols, colcount):
  # Return a list of dog data
  return cols[4:colcount]

# Define the method as a UDF
udfRetriever = F.udf(retriever, ArrayType(StringType()))

# Create a new column using your UDF
split_df = split_df.withColumn('dog_list', udfRetriever(split_df.split_cols, split_df.colcount))

# Remove the original column, split_cols, and the colcount
split_df = split_df.drop('_c0').drop('split_cols').drop('colcount')


# Define a UDF to determine the number of pixels per image
def dogPixelCount(doglist):
  totalpixels = []
  for dog in doglist:
    totalpixels += (dog[0] - dog[1]) * (dog[0] - dog[1])
  return totalpixels

# Define a UDF for the pixel count
udfDogPixelCount = F.udf(dogPixelCount, StringType())
joined_df = joined_df.withColumn('dog_pixels', udfDogPixelCount(joined_df.dogs))

# Create a column representing the percentage of pixels
joined_df = joined_df.withColumn('dog_percent', (joined_df.dog_pixels / (joined_df.dog_pixels.count())) * 100)

# Show the first 10 annotations with more than 60% dog
joined_df.select(joined_df['dog_percent']> 60 )