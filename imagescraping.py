# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import requests
from bs4 import BeautifulSoup as BS
import os
import glob

#image downloading function

def imagedownloader(url, folder):
    #url = "https://www.airbnb.com/s/Dhaka/homes?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2023-09-01&monthly_length=3&price_filter_input_type=0&price_filter_num_nights=5&channel=EXPLORE&date_picker_type=calendar&checkin=2023-08-30&checkout=2023-08-31&source=structured_search_input_header&search_type=filter_change"
    os.mkdir(os.path.join(os.getcwd(), folder))
    os.chdir(os.path.join(os.getcwd(), folder))

    response = requests.get(url)
    html_text = response.text

    soup = BS(html_text, "html.parser")

    print(soup.title.text)

    images = soup.find_all('img')

    index = 1

    for image in images:
        link = image['src']
        name = 'Image_' + str(index)
        #name = link.split('/')[-1].split('.')[-2]

        with open(name + '.jpg', 'wb') as f:
            im = requests.get(link)
            f.write(im.content)
            print('Writing: ', name)

        index = index + 1

#end of image downloading function


URL = "https://iheartcraftythings.com/rocket-drawing.html"
folder_name = 'test_image_scraper'

imagedownloader(URL, folder_name)