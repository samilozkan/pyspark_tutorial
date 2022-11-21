'''
Validate rows via join
Another example of filtering data is using joins to remove invalid entries. 
You'll need to verify the folder names are as expected based on a given DataFrame named valid_folders_df. 
The DataFrame split_df is as you last left it with a group of split columns.
The spark object is available, and pyspark.sql.functions is imported as F.
'''


# Rename the column in valid_folders_df
valid_folders_df = valid_folders_df.withColumnRenamed('_c0', 'folder')

# Count the number of rows in split_df
split_count = split_df.count()

# Join the DataFrames
joined_df = split_df.join(F.broadcast(valid_folders_df), "folder")

# Compare the number of rows remaining
joined_count = joined_df.count()
print("Before: %d\nAfter: %d" % (split_count, joined_count))