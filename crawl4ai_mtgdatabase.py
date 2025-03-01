import os
import re
import json
import asyncio
import requests
import backoff 
import openai
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Create rate limiting wrapper
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai_client.chat.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def embeddings_with_backoff(**kwargs):
    return openai_client.embeddings.create(**kwargs)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_json(text: list) -> List[str]:
    """Split text into card name, card type, oracle text (if applicable), and power/toughness (if applicable)."""
    
    card_information = {}

    for item in text:
        if item[0] == 'Card Name': # Check if item is Card Name
            card_information['Card Name'] = item[1]

        elif item[0] == 'Card Type': # Check if item is Card Type
            card_information['Card Type'] = item[1]

        elif item[0] == 'Oracle Text': # Check if item is Oracle Text
            card_information['Oracle Text'] = item[1]

        elif item[0] == 'Power Toughness': # Check if item is Power/Toughness
            card_information['Power Toughness'] = item[1]

    return card_information

async def get_title_and_summary(chunk: dict) -> Dict[str, str]:
    """Extract card name, card type, oracle text (if applicable), and power/toughness (if applicable) using GPT-4"""
    

    system_prompt = """You are an AI that extracts extracts a card's name, a card's type, a card's oracle text (if applicable), and a card's power/toughness (if applicable) from Magic: The Gathering from a dictionary. Note that not all cards have a oracle text or power and toughness. You will be fed a string separated by return key or \n markers.
    The first line is the Card's Name. The second line is the Card Type. If there is are words as opposed to numbers, then that is the Oracle Text of the card. If the third string has either numbers such as 5/5 or something like */*,  then that is the Power/Toughness of that card. If there are four lines present in the string, then the third line is the Oracle Text
    and the fourth line is the Power/Toughness of the card.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: The Card Name will always be title.
    For the summary: Create a summary of the card using the rest of the text.
    Keep both title and summary concise but informative."""
    
    try:
        response = await completions_with_backoff(
            model=os.getenv("AI", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Card Information: \n\n{chunk}"}  # Send first 200 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embeddings_with_backoff(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: dict, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""

    joined_text = "\n".join([str(value) for value in chunk.values()])

    extracted = await get_title_and_summary(joined_text)

    # Get embedding
    embedding = await get_embedding(joined_text)
    
    # Create metadata
    metadata = {
        "source": "mtgcarddatabase",
        "chunk_size": len(joined_text),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=0,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, data: list):
    """Process  JSON document and store its chunks in parallel"""
    # Ensure valid JSON
    try:
        data = list(data[0].items())
    except Exception as e:
        raise ValueError(f"Invalid Dict: {e}")
    
    # Split into chunks
    chunks = chunk_json(data)
    # Process chunks in parallel
    tasks = [
        process_chunk(chunks, url)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-dev-shm-usage", "--no-sandbox"],
    )

    schema = {
		"name": "Card",
		"baseSelector": ".cdb_card_face",
		"fields": [
			{
				"name": "Card Name",
				"selector": "div.cdb_cardname",
				"type": "text"
			},
			{
				"name": "Card Type",
				"selector": "div.cdb_type_row",
				"type": "text"
			},
			{
				"name": "Oracle Text",
				"selector": "div.cdb_oracle_box",
				"type": "text"
			},
			{
				"name": "Power Toughness",
				"selector": "div.cdb_power_toughness_row",
				"type": "text"
			}
		]
	}

    schema = JsonCssExtractionStrategy(schema)

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS,
                                    extraction_strategy = schema
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    data = json.loads(result.extracted_content)
                    await process_and_store_document(url, data)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_sitemaps(sitemap_url):
    """
    Filters a list of sitemaps, returning only sitemaps that contain ONLY the needed sitemaps.
    
    Args:
    	urls: A list of sitemaps (string)
    
    Returns:
    	A new list containing only needed sitemaps
    """
    filtered_sitemap = []
    response = requests.get(sitemap_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content,features='xml')
    for tag in soup.find_all('loc'):
        if re.match(r"https?:\/\/www.mtgnexus.com\/sitemap-page-cards-\d+-sitemap.xml", tag.text):
            filtered_sitemap.append(tag.text)
    return filtered_sitemap

def get_mtgdatabase_urls() -> List[str]:
    """Get URLs from MTGNexus sitemap."""
    sitemap_url = "https://www.mtgnexus.com/sitemap.xml"
    sitemaps  = get_sitemaps(sitemap_url)
    mtgdatabase_urls = []
    try:
        for sitemap in sitemaps:
            response = requests.get(sitemap)
            response.raise_for_status()
            
            # Parse the XML
            soup = BeautifulSoup(response.content,features='xml')
            for tag in soup.find_all('loc'):
                mtgdatabase_urls.append(tag.text)
            
        return mtgdatabase_urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def get_unique_urls(urls) -> List[str]:
    """Remove duplicate URLs"""
    card_name = []
    unique_urls = []
    pattern = r"https?:\/\/www.mtgnexus.com\/cards\/\S+\/\d+-(\S+)"

    for url in urls:
        match = re.match(pattern,url)
        if match:
            group1 = match.group(1)
            if group1 not in card_name:
                card_name.append(match.group(1))
                unique_urls.append(url)
            elif match.group(1) in card_name:
                continue
        else:
            # Handle URLs that don't match the pattern (optional)
            print(f"URL did not match the pattern: {url}") # or you can just continue.

    return unique_urls

async def main():
    # Get URLs from MTGNexus
    urls = get_mtgdatabase_urls()
    unique_urls = get_unique_urls(urls)
    if not unique_urls:
        print("No URLs found to crawl")
        return
    
    len_urls = len(unique_urls)
    print(f"Found {len_urls} URLs to crawl")
    await crawl_parallel(unique_urls)

if __name__ == "__main__":
    asyncio.run(main())