#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions defined in this file are used for web scraping from TrueCar.com

@author: Wei Zhao @ Metis, 01/06/2021
"""
import sys
import pickle
import requests
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
sys.setrecursionlimit(1000000000)
#%%
def veh_filter(city_loc="boston-ma",
               search_radius=25,
               current_page_num=100
               ):
    """
    Function to define a filter to select vehicles.
    The default searches all vehicles 
    within 50 miles of Worcester MA
    """
    url = (
          "https://www.truecar.com/used-cars-for-sale/listings/"
           + "location-" + city_loc
           + "?page=" + str(current_page_num)
           + "&searchRadius=" + str(search_radius)
           + "&sort[]=distance_asc_script"
           )
    return url

def scrape_data(url):
    """
    Function to scrape data from TrueCar. 
    Scraped features includes:
        price, year, make, model, mileage, transmission,
        drive type (dr_type), mpg, engine, fuel type (fuel_type)
    The function returns a complete pandas data frame
    """
    veh_year = []
    veh_make = []
    veh_model = []
    dict_keys = ["price", "mileage", "transmission",
                 "drive type", "mpg", "engine",
                 "fuel type"]
    dict_spec = defaultdict(list)
    
    failed_url = []
    failed_url_card = []
    try:       
        for c_url, u in enumerate(url):
            c_url += 1
            print("Extracting page {}".format(c_url))
            ua = UserAgent()
            user_agent = {'User-agent': ua.random}
            page = requests.get(u, headers = user_agent).text
            soup = bs(page, "html5lib")
            veh_header = soup.find_all("span",
                              class_="vehicle-header-make-model")
            # get make and model
            for v in veh_header:
                veh_make.append(v.text.split(" ")[0])
                veh_model.append(v.text.split(" ")[-1])
            # get year
            veh_card_year = soup.find_all("span",
                              class_="vehicle-card-year")
            for v in veh_card_year:
                veh_year.append(v.text)
                
            # now need to extract details for each vehicle
            #1) get the link for each card
            t_a = soup.find_all("a", class_="card")
            url_card = ["https://www.truecar.com"
                        + t.get("href") for t in t_a]
            #2) for each url identify mileage card
            
            for c_car, uc in enumerate(url_card):
                c_car += 1
                if c_car%10 == 0:
                    print("Extracted {} cars".format(c_car))

                ua = UserAgent()
                user_agent = {'User-agent': ua.random}
                page = requests.get(uc, headers = user_agent).text
                soup2 = bs(page, "html5lib")

                # get price value
                t_price = soup2.find('div', class_="label-block-text")
                if t_price is not None:
                    dict_spec["price"].append(t_price.text)
                else:
                    # some cars do not have lised price and
                    # need to contact dealer, then fill None
                    dict_spec["price"].append(None)                
                # only need the first spacing-2 class 
                # as that's for vehicle details
                t_txt = soup2.find("div", class_="spacing-2")
                t_col = t_txt.find_all("div", class_="col-12")                
                
                for col in t_col:
                    txt_heading = col.find('h4',class_="heading-5")
                    txt_li = col.find("li")
                    
                    if (txt_heading is not None 
                        and txt_li is not None):
                        
                        if txt_heading.text.lower() in dict_keys:
                            if len(txt_li.text.lower()) > 0:
                                (dict_spec[txt_heading.text.lower()]
                                 .append(txt_li.text.lower()))
                            else:
                                (dict_spec[txt_heading.text.lower()]
                                 .append(None))
                                 
            t_dict_all = dict_spec
            t_dict_all["year"] = veh_year
            t_dict_all["make"] = veh_make
            t_dict_all["model"] = veh_model
            df = pd.DataFrame(t_dict_all)   
            
            # save the data in case there is an outage of power or internet
            if (c_url+1)%50 == 0:
                fn = ("../data/vehicle_value_data.pickle")
                with open(fn, 'wb') as to_write:
                    pickle.dump(df, to_write)
                    
    except:
        failed_url.append(u)
        failed_url_card.append(uc)

    return df, failed_url, failed_url_card

#%%

def main():
    
    max_page_num = 333 # 333 pages are allowed    
    current_page_num = range(1, max_page_num+1)
    
     # put all pages into a list 
    url = [veh_filter(city_loc="boston-ma",
                      search_radius=25,                      
                      current_page_num=str(n))
           for n in current_page_num]
    
    df, failed_url, failed_url_card = scrape_data(url)
      

if __name__ == "__main__":
    main()