import os
import json
import time
from langchain_community.document_transformers import BeautifulSoupTransformer
from playwright.sync_api import sync_playwright
from langchain.schema import Document

# Define the user agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def fetch_html_with_user_agent(url, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=USER_AGENT)
                page = context.new_page()
                page.goto(url, timeout=60000)  # Increase timeout to 60 seconds
                content = page.content()
                browser.close()
            return content
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            attempt += 1
            time.sleep(delay)
    raise Exception(f"Failed to fetch {url} after {retries} attempts")

# URLs to scrape
urls = ["https://www.mentorworks.ca/government-funding/funding-regions/western/agri-food-and-bio-industrial-innovation-program-abip/"]
html_contents = [fetch_html_with_user_agent(url) for url in urls]

# Create Document objects from the HTML content
documents = [Document(page_content=html, metadata={"url": url}) for html, url in zip(html_contents, urls)]

# Transform HTML content using BeautifulSoupTransformer
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(documents, tags_to_extract=["span"])

# Extract content from transformed documents
extracted_content = []
for doc in docs_transformed:
    extracted_content.append({
        "content": doc.page_content,
        "metadata": doc.metadata
    })

# Save extracted content to a JSON file
output_file = "scraped_content1.json"
with open(output_file, "w") as f:
    json.dump(extracted_content, f, indent=4)

print(f"Scraped content saved to {output_file}")
