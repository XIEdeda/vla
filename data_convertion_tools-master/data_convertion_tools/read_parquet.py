path = "/mnt/d/github/data_conversion_tools/lerobot_data_v2.1/0421_cyw_coffee_02.img.224x168.N.dof.16/data/chunk-000/episode_000000.parquet"

import pyarrow.parquet as pq

table = pq.read_table(path)
print("---- schema ----")
print(table.schema)

print("---- shape ----")
pd = table.to_pydict()
for name in table.column_names:
    string = "column name: " + name + ", shape: ("# + str(n)
    col = pd.get(name, [])
    while isinstance(col, list):
        string += str(len(col)) + ","
        col = col[0]
    string += ')'
    print(string)