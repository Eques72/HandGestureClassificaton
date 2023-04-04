import pandas as pd
import numpy as np
import torch as t
import DatasetManager as dm
 
#use haGRID database
def main():
    #TEST
    D = dm.DatasetManager("D:\\Politechnika\\BIAI\\ann_subsample\\ann_subsample", "D:\\Politechnika\\BIAI\\subsample", True, None, 42, 0.8)
    D.getSample(666)
    pass