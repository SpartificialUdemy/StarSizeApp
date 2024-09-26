import numpy as np
import pandas as pd


# Define a seed value (other than 100 as 100 was used for training)
np.random.seed(5007) 

# Test data
X_test = 3*np.random.rand(500,1)
y_test = 9 + 2*X_test + np.random.rand(500,1)

dict_info = {'Brightness' : X_test.reshape(-1,),
             'True Size' : y_test.reshape(-1,)}

input_df = pd.DataFrame(dict_info)
input_df.to_csv('input_star_data.csv', index=False)