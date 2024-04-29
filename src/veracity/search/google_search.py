import requests
from typing import List
from src.utils.utils import load_json, load_lang_codes
import os
import re
from bs4 import BeautifulSoup
from googlesearch import search


def search_google(query: str, num_web_pages: int = 10, timeout : int = 6, save_url: str = '') -> List[str]:
    """Searches the query using Google. 
    Args:
        query: Search query.
        num_web_pages: the number of web pages to request.
        save_url: path to save returned urls, such as 'urls.txt'
    Returns:
        search_results: A list of the top URLs relevant to the query.
    """
    query = query.replace(" ", "+")

    # set headers: Google returns different web-pages according to agent device
    # desktop user-agent
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    # mobile user-agent
    MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
    headers = {'User-Agent': USER_AGENT}
    
    # set language
    # set the Google interface language, use &hl=XX
    # set the preferred language of the search results, use &lr=lang_XX
    # set language as en, otherwise it will return many translation web pages to Arabic that can't be opened correctly.
    lang = "en" 

    # scrape google results
    urls = []
    for page in range(0, num_web_pages, 1):
        # here page is google search's bottom page meaning, click 2 -> start=10
        # url = "https://www.google.com/search?q={}&start={}".format(query, page)
        url = "https://www.google.com/search?q={}&lr=lang_{}&hl={}&start={}".format(query, lang, lang, page)
        r = requests.get(url, headers=headers, timeout=timeout)
        html_content = r.text
        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())
        # Find all paragraph tags and extract text
        # paragraphs = soup.find_all('p')
        # for p in paragraphs:
        #     print(p.get_text())
        # collect all urls by regular expression
        # how to do if I just want to have the returned top-k pages?
        divs_with_urls = soup.find_all('div', string=re.compile(r'https?://'))
        print(divs_with_urls)
        for div in divs_with_urls:
            span_text = div.find_all("span")
            url = div.get_text(strip=True)
            print(url, span_text)
        urls += re.findall('href="(https?://.*?)"', r.text)

    # set to remove repeated urls
    urls = list(set(urls))

    # save all url into a txt file
    if not save_url == "":
        with open(save_url, 'w') as file:
            for url in urls:
                file.write(url + '\n')
    return urls

for lang in load_lang_codes():
    split = "test"
    data_file = f"data/veracity_prediction/{lang}_{split}.json"
    if not os.path.exists(data_file):
            continue
    for data in load_json(data_file):
        google_search_results = search_google("Earth is flat", num_web_pages=1, timeout=6, save_url=f"data/veracity_prediction/google_search_results_{lang}_{split}.txt")
        break
    break