import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from backboard import BackboardClient

async def main():
    c = BackboardClient(api_key=os.getenv("BACKBOARD_API_KEY"))
    assistants = await c.list_assistants()
    for a in assistants:
        prompt_preview = (a.system_prompt or "")[:100].replace("\n", " ")
        print(f"{a.assistant_id} | {a.name} | {prompt_preview}...")

asyncio.run(main())
