"""Print the row count of a parquet file. Usage: python count_rows.py <parquet>"""
import sys

import pyarrow.parquet as pq

print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
