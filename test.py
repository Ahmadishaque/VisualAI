import requests
from bs4 import BeautifulSoup

# URL of the webpage you want to scrape
url = 'https://asu.campusdish.com/en/diningvenues/tempe-campus/tookerhousedining/'

# Make a GET request to fetch the webpage content
response = requests.get(url)

# Parse the page content with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all span tags with the specified class and data-testid attribute
span_tags = soup.find_all('span', class_='HeaderItemNameLink', data_testid='product-card-header-link')

# Extract the inner text from each span tag and store it in a list
span_texts = [span.get_text() for span in span_tags]

# Print the list of extracted texts
print(span_texts)
