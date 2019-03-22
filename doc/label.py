import numpy as np 
import pandas as pd
import os
from pprint import pprint as print

dir = "/Users/tianchenwang/Git/proj3/train_set"
os.chdir(dir)

label = pd.read_csv("label.csv")
print(label)
print(label.query("Label == 2"))
print(label.query("Label == 2").shape)