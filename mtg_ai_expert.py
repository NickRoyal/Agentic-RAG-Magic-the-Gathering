from __future__ import annotations as _annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import json
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('AI','gpt-4o')
model = OpenAIModel(llm)

#logfire.configure(send_to_logfire='if-token-present')

@dataclass
class MTGDeps:
	supabase: Client
	openai_client: AsyncOpenAI

system_prompt = """
You are an expert at Magic: The Gather (MTG) - a turn-based, tabletop card game created by Wizards of the Coast. You have access to the essential parts of the Magic: The Gathering website as well as the MTG Wiki and MTG Nexus.

Your job is to assist players with answering any questions that ONLY pertain to the Magic: The Gathering card game and describe what services you can offer. This includes--but is not limited--things such as the rules of the game, sets, card rulings, interaction rulings, card descriptions, and deck building

There are a few details about the card descriptions that you need to know. Cards can have activated abilities that may cause the card to become tapped and may cost mana. If {T} is present in the description of the card, it stands for tap. If {S} is present, then that means a permanent with "Snow" in its card type that taps for mana must be tapped to pay that mana cost. {W} stands for white mana, and {R} stands for red mana
{U} stands for blue mana, {B} stands for black mana, {G} stands for green mana. If you encounter a description that looks something like {U/R}, then that implies it is a hybrid mana cost - a white or red mana can be used to activate that ability. If you encounter a description that looks something like {U/P} that means
a player can either pay one blue mana or 2 life to actuvate that ability. A number inside the curly braces such as {2} represents colorless mana where the number is how much colorless mana you need in order to activate the ability. Note, colored mana can be used to cast colorless abilities. An X inside the curly braces such as {X} means that the 
user may pay any amount of mana to cast that ability--be mindful because there are limitations to this in the card descriptions stating that X can't be zero. For activated abilities, if you see a comma in between a mana value and a tap symbol such {G},{T} that means the player has to use mana and tap the card to
activate the ability.

Don't ask the user before taking action, just do it. Use the documentation provided to you before answering the user's question unless you have already.

When you first look at the documentation, cite where you are getting your information from such as Magic: The Gathering website, The MTG Wiki, and MTG Nexus or any combination of those you used to find the information. 
Then also always check the list of available documentation pages and retrieve the content of those pages and retrieve the content to justify your findings.

ALWAYS cite your sources. Provide links to where you are finding your information at all times. 

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

mtg_ai_expert = Agent(
	model,
	system_prompt=system_prompt,
	deps_type=MTGDeps,
	retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
	"""Get embedding vector from OpenAI"""
	try:
		response = await openai_client.embeddings.create(
			model='text-embedding-3-small',
			input=text
		)
		return response.data[0].embedding
	except Exception as e:
		print(f"Error getting embedding: {e}")
		return[0] * 1536 # Return zero vector on error

@mtg_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[MTGDeps], user_query: str) -> str:
	"""
	Retrieve relevant documentation chunks based on the query with RAG.

	Args:
		ctx: The context including the Supabase client and OpenAI client
		user_query: The user's question or query

	Returns:
		A formatted string containing the top 5 most relevant documentation chunks
	"""

	try:
		# Get the embedding for the query
		query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

		# Query Supabase for relevant documents
		result = ctx.deps.supabase.rpc(
			'match_site_pages',
			{
				'query_embedding': query_embedding,
				'match_count': 5,
				'filter': {'source': ['mtg_wiki','wizard','mtgcarddatabase']}
			}
		).execute()

		if not result.data:
			return "No relevant documentation found."
		
		# Format the results
		formatted_chunks = []
		for doc in result.data:
			chunk_text=f""
			formatted_chunks.append(chunk_text)
		return "\n\n---\n\n".join(formatted_chunks)
	
	except Exception as e:
		print(f"Error retrieving documentation: {e}")
		return(F"Error retrieving documentation: {str(e)}")
	
@mtg_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[MTGDeps]) -> List[str]:
	'''
	Retrieve a list of all relevant available MTG Resource pages.
	
	Returns:
		List[str]: List of unique URLs for all resources.
	'''
	try:
		# Query Supabase for unique URLs where source is mtg_wiki
		result = ctx.deps.supabase.from_('site_pages') \
		.select('url')\
		.eq('metadata->>source','mtg_wiki') \
		.execute()

		if result.data:
            # Extract unique URLs 
			urls = sorted(set(doc['url'] for doc in result.data))
		else:
			urls = []
	
		# Query Supabase for unique URLs where source is scryfall
		results_wizard = ctx.deps.supabase.from_('site_pages') \
		.select('url')\
		.eq('metadata->>source','wizard') \
		.execute()

		if results_wizard.data:
			urls.extend(sorted(set(doc['url']for doc in results_wizard.data)))

		# Query Supabase for unique URLs where source is mtgcarddatabase
		result_mtgcard = ctx.deps.supabase.from_('site_pages') \
		.select('url')\
		.eq('metadata->>source','mtgcarddatabase') \
		.execute()

		if  result_mtgcard.data:
			urls.extend(sorted(set(doc['url'] for doc in result_mtgcard.data)))

		return sorted(set(urls))

	except Exception as e:
		print(f"Error retrieving documentation pages: {e}")
		return []

@mtg_ai_expert.tool
async def get_page_content(ctx: RunContext[MTGDeps], url: str) -> str:
	"""
	Retrieve the full content of a specific documentation page by combining all its chunks.
	
	Args:
		ctx: The context including the Supabase client
		url: The URL of the page to retrieve
	
	Returns:
		str: The complete page content with all chunks combined in order
	"""
	try:
		# Query Supabase for all chunks of this URL, ordered by chunk_number
		result = await ctx.deps.supabase.from_('site_pages') \
			.select('*') \
			.eq('url', url) \
			.in_('metadata', ['mtg_wiki', 'wizard', 'mtgdcarddatabase']) \
			.order('chunk_number') \
			.execute()

		if not result.data:
			return f"No content found for URL: {url}"
		
		# Format the page with its title and all chunks
		page_title = result.data[0]['title'].split(' - ')[0] # Get main title
		formatted_content = [f"# {page_title}\n"]

		# Add each chunk's content together
		for chunk in result.data:
			formatted_content.append(chunk['content'])

		# Join everything together
		return "\n\n".join(formatted_content)

	except Exception as e:
		print(f"Error retrieving page content: {e}")
		return f"Error retrieving page content: {str(e)}"

@mtg_ai_expert.tool
async def get_card_details(ctx: RunContext[MTGDeps], card_name:str) -> str:
	'''
	Retrieves Card Name from database
	Args:
		ctx: The context including the Supabase client
		card_name: The Card Name of the card to retrieve

	Returns:
		Card Name
	'''
	try:
		result = ctx.deps.supabase.from_('site_pages') \
		.select('*') \
		.filter('title', 'ilike', f'%{card_name}%') \
		.execute()

		if not result.data:
			return f'No Card Name found'

		# Extract the 'content' field (which is a JSON-like string)
		card_data = result.data[0] # Assuming the result contains a single row
		content = card_data.get('content','{}') # Default to empty JSON if not found

		try:
			content_dict = json.loads(content) # Parse the JSON string into a Python dict
		except json.JSONDecodeError as e:
			return f"Error decoding JSON for card {card_name}: {str(e)}"
		
		# Extract relevant fields from the parsed JSON content
		card_type = content_dict.get('Card Type', 'No Card Type Available')
		oracle_text = content_dict.get('Oracle Text', 'No Oracle Text available')
		power_toughness = content_dict.get('Power Toughness', 'No Power/Toughness available')
        
		return f"Card Name: {card_name}\n" \
               f"Card Type: {card_type}\n" \
               f"Oracle Text: {oracle_text}\n" \
               f"Power/Toughness: {power_toughness}"
	
	except Exception as e:
		print(f"Error retrieving card details for {card_name}: {e}")
		return f"Error retrieving card details: {str(e)}"
