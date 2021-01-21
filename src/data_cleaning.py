#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this file after web_scraping.py to clean the scraped data.

@author: Wei Zhao @ Metis, 01/06/2021
"""
import pickle
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def get_cty_mpg_val(t):
    tt = t.split(" hwy")[0].split("cty / ")
    tt = tt[0]
    return tt
def get_hwy_mpg_val(t):
    tt = t.split(" hwy")[0].split("cty / ")
    tt = tt[-1]
    return tt
def get_engine_size(t):
    tt = t.split(" ")[0].split('l')
    tt = tt[0]
    return tt
def get_engine_turbo(t):
    tt = t.split(" ")[-1]
    if "turbo" in tt:
        return "turbo"
    else:
        return "regular"

def data_cleaning(df):
    
    df["cty_mpg"] = df["mpg"].apply(get_cty_mpg_val)
    df["hwy_mpg"] = df["mpg"].apply(get_hwy_mpg_val)
    # df["cty_mpg"][df["cty_mpg"] != 'n/a '] =  (df["cty_mpg"][df["cty_mpg"] != 'n/a ']
    #                                            .astype(np.int64))
    df["cty_mpg"][df["cty_mpg"] != 'n/a '] =  df["cty_mpg"][df["cty_mpg"] != 'n/a ']
    df["cty_mpg"][df["cty_mpg"] == 'n/a '] = None    
    df["hwy_mpg"][df["hwy_mpg"] != 'n/a'] =  df["hwy_mpg"][df["hwy_mpg"] != 'n/a']                                              
    df["hwy_mpg"][df["hwy_mpg"] == 'n/a'] = None
    
    
    df["mileage"] = df["mileage"].str.split(",").str.join('').astype(np.int64)    
    # inspected all vehicles with n/a fuel type
    # All of them use gas
    df["fuel type"][df["fuel type"]=="n/a"] = "gas"
    
    df["price"] = df["price"].str[1:].str.split(",").str.join('')
    df["drive type"][df["drive type"] == "4wd"] = "awd"
    # The # of vehicles using manual transmission is little 
    # So it is reasonable to assume they use automatic 
    # transmission 
    df["transmission"][df["transmission"] == "n/a"] = "automatic"
    
    df["engine_size"] = df["engine"].apply(get_engine_size)
    df["engine_size"][df["engine_size"]==""] = None
    df["engine_type"] = df["engine"].apply(get_engine_turbo)
    
    df = df.dropna(subset=["price", "mileage",
                      "engine_size",
                      "cty_mpg", "hwy_mpg"])
    
    df["year"] = df["year"].astype(np.int64)
    df["cty_mpg"] = df["cty_mpg"].astype(np.int64)
    df["hwy_mpg"] = df["hwy_mpg"].astype(np.int64)
    df["price"] = df["price"].astype(np.int64)
    df["engine_size"] = df["engine_size"].astype(float)
    # only look at used vehicle < $50k
    df = df.drop(df[(df["price"] > 40000) | (df["price"] < 8000)].index, inplace=False)
    df = df.drop(df[df["cty_mpg"] > 80].index, inplace=False)
    
    df2 = df[["make", "year", "mileage", "fuel type", "drive type", 
             "transmission", "engine_size", "engine_type",
             "cty_mpg", "hwy_mpg", "price"]]
    df2 = df2.rename(columns={"fuel type": "fuel_type", "drive type": "drive_type"})
    
    return df2
#%%
def main():
    fn = ("../data/vehicle_value_data.pickle")
    with open(fn,'rb') as read_file:
        df = pickle.load(read_file)
    
    df = data_cleaning(df)
    return df

if __name__ == "__main__":
    df = main()    
