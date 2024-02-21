import requests
from bs4 import BeautifulSoup

# Define the URL of the page containing the stories
url = "https://americanliterature.com/short-stories-for-children/"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all <figure> tags with class "al-figure"
    figure_tags = soup.find_all("figure", class_="al-figure")
    
    # Initialize an empty list to store href values
    hrefs = []
    
    # Loop through each <figure> tag
    for figure_tag in figure_tags:
        # Find the <a> tag within the <figure> tag
        a_tag = figure_tag.find("a")
        # Get the value of the "href" attribute and append it to the list
        href = a_tag.get("href")
        hrefs.append(href)


else:
    print("Failed to retrieve the webpage.")
# URL of the webpage
with open("children_short_stories_paragraphs.txt", "w+", encoding="utf-8") as file:    
    for href in hrefs:
        url = "https://americanliterature.com" + href
    # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find all <p> tags and extract their content
            paragraphs = soup.find_all('p')
            for paragraph in paragraphs:
                file.write(paragraph.get_text().strip())

    else:
        print("Failed to retrieve the webpage.")