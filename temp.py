import os 
import numpy as np

val_base_dir = "/home/sdf/xujiping/tmb_bladder/data/HL_data/val"

val_0 = os.listdir(val_base_dir + '/' + '0')
val_1 = os.listdir(val_base_dir + '/' + '1')

val_name = []

for s in val_0:
    val_name.append(s[:16])

for s in val_1:
    val_name.append(s[:16])

val_name = set(val_name)



