## This is a collection of functions to let the model properly scrape through websites ##

import requests

response = requests.get('https://www.geeksforgeeks.org/python/python-programming-language-tutorial/')

print(response.status_code)

print(response.content)