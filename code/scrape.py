# This file scrapes the rotten tomatoes review api
# eventually this will scape the rotten tomatoes review api
# in a truly randomized and gaussian way.
import requests
import json
import re
import time

from bs4 import BeautifulSoup

# global variables that are only there for testing purposes
forgoed_count = 0
empty_count = 0
apikey ='qpbuunhuhmmp9xsapp3gjcp2'

def grab_review_link(movie):
    link = movie['links']['reviews']
    if link is None:
        return None

    response = requests.get(str(link) + '?apikey=' + apikey)
    json_response = json.loads(response.text)

    reviews = json_response.get('reviews')
    if reviews is None:
        # this is way to high, needs to be looked into and fixed
        import pdb; pdb.set_trace()
        print json_response

    return reviews

def grab_review_response(review):
    accepted_publications = ['Washington Post', 'Seattle Times', 'San Francisco Chronicle', 'Boston Globe']
    review_publication = review['publication']
    global forgoed_count

    if review_publication not in accepted_publications:
        forgoed_count = forgoed_count + 1
        return None

    ext_link = review['links'].get('review')
    if ext_link is None:
        return None

    response = requests.get(ext_link)
    return response

def write_to_xml(soup, textList):
    # write out each review out to a file in a directory
    # this will help us build the proper corpus of rotten tomatoes
    # documents
    filename = soup.title.text.replace(' ', '_')
    filename = re.sub('[^A-Za-z0-9]+', '_', filename)
    f = open('/Users/abhinavkhanna/Documents/Princeton/Independent Work/rotten_tomato/rotten_tomato_crawler/' + filename + '.xml', 'w')
    xmlOpen ='<?xml version="1.0" encoding="utf-8"?><?xml-model  href="../schema/digcor.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>'
    f.write(xmlOpen + '\n')
    f.write('<items>' + '\n')
    f.write('<item>' + '\n')
    f.write('<date></date>\n<year></year>\n<month></month>\n<day></day>\n')
    f.write('<solr-datetime></solr-datetime>\n')
    f.write('<headline>' + soup.title.text.replace('&', ' ').encode('utf-8') + '</headline>\n')
    print soup.title.text.replace('&', ' ').encode('utf-8')
    f.write('<source>Rotten Tomatoes</source>\n')
    f.write('<body>\n')
    for text in textList:
        t = re.sub('[^A-Za-z0-9\s]+', '', text.text)
        f.write(t.encode('utf-8'))
    f.write('</body>\n')
    f.write('</item>\n')
    f.write('</items>\n')
    f.close()

def scrape():
    # uses the api to scrape some reviews
    query_terms = ['action', 'comedy', 'family', 'animation', 'foreign', 'classics', 'documentary', 'drama', 'horror', 'mystery', 'romance', 'fantasy']
    global empty_count

    print 'Grabbing all hte movies'
    for query_term in query_terms:
        for i in range(1,10):
            # get the first 10 pages of rotten tomatoes data set
            url = 'http://api.rottentomatoes.com/api/public/v1.0/movies.json'
            url = url + '?apikey=qpbuunhuhmmp9xsapp3gjcp2&q=' + str(query_term) + '&page=' + str(i)
            response = requests.get(url)
            json_rep = json.loads(response.text)
            movies = json_rep.get('movies')
            if movies is None:
                continue

            for movie in movies:
                # second option, lets do it all streamlined
                time.sleep(2) # delays to ensure we do not go over query limit for API calls
                reviews = grab_review_link(movie)
                if reviews is None:
                    # this is way to high, needs to be looked into and fixed
                    empty_count = empty_count + 1
                    continue
                
                for review in reviews:
                    time.sleep(2)
                    response = grab_review_response(review)
                    if response is None:
                        continue

                    soup = BeautifulSoup(response.text)
                    textList = soup.find_all('p') # find all the paragraphs
                    write_to_xml(soup, textList)


scrape()
