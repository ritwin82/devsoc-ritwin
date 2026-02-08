"""Delete ALL existing Backboard assistants, then verify they're gone."""
import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from backboard import BackboardClient

async def main():
    c = BackboardClient(api_key=os.getenv("BACKBOARD_API_KEY"))
    assistants = await c.list_assistants()
    print(f"Found {len(assistants)} assistants to delete...")
    
    for a in assistants:
        try:
            await c.delete_assistant(a.assistant_id)
            print(f"  Deleted: {a.assistant_id} ({a.name})")
        except Exception as e:
            print(f"  Failed to delete {a.assistant_id}: {e}")
    
    # Verify
    remaining = await c.list_assistants()
    print(f"\nRemaining assistants: {len(remaining)}")

asyncio.run(main())
