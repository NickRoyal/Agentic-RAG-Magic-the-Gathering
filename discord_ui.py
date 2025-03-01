import discord
from typing import Literal, TypedDict
import asyncio
import os
from discord.ext import commands
from mtg_ai_expert import MTGDeps, mtg_ai_expert
from supabase import Client
from openai import AsyncOpenAI
import json
import logging
# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger('discord')

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str

# Create bot instance
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!",intents=intents)

@bot.event
async def on_message(message):
    logger.info(f'Received message: "{message.content}" from {message.author}')
    conversation = []
    
    if message.author == bot.user:
        return  # Ignore the bot's own messages
    
    # Prepare dependencies
    deps = MTGDeps(supabase=supabase, openai_client=openai_client)
    
    # Get the user input
    user_input = message.content
    logger.info(f"User input: {user_input}")
    
    try:
        async with mtg_ai_expert.run_stream(user_input, deps=deps) as result:
            # Log the raw messages
            new_msgs = result.new_messages()
            logger.info(f"New messages: {new_msgs}")
            
            async for chunk in result.stream_text(delta=True):
                 partial_text += chunk
            
            # Filter out unwanted messages
            filtered_messages = [
                msg for msg in new_msgs
                if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
            ]
            logger.info(f"Filtered messages: {filtered_messages}")
            
            # Combine the text from filtered messages
            content = ""
            for msg in filtered_messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'text'):
                            content += part.text + ' '
            
            logger.info(f"Final content to send: {content.strip()}")
            
            conversation.append(ModelResponse(parts=[TextPart(content=content.strip())]))
            await message.channel.send(conversation)
    
    except Exception as e:
        logger.error(f"Error while processing the message: {e}")
        await message.channel.send("An error occurred while processing your request.")




bot.run(os.getenv("DISCORD_TOKEN"))