import numpy as np
from grid_utils.generate_grid import fix_index_array

arr = np.array([1,1,2,3,3,3,7,9,8,7])
arr_fix = fix_index_array(arr)[0]

print(arr)
print(arr_fix)


