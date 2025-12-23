# from picnic.ai.llm import AsyncLLMClient
# import asyncio

# async def list_models():
#     client = AsyncLLMClient()
#     models = await client.openai_client.models.list()
#     for m in models.data:
#         print(m.id)

# asyncio.run(list_models())
# from picnic.ai.llm import AsyncLLMClient
# import asyncio
# import re

# async def list_models():
#     client = AsyncLLMClient()
#     models = await client.openai_client.models.list()
    
#     # 1. Parse models into a structured list
#     parsed_models = []
#     for m in models.data:
#         # Split "provider/model_name"
#         if "/" in m.id:
#             provider, name = m.id.split("/", 1)
#         else:
#             provider, name = "unknown", m.id

#         # Try to extract a date (YYYYMMDD or YYYY-MM-DD) for sorting
#         # This regex looks for 2023, 2024, 2025 followed by digits
#         date_match = re.search(r'(202[3-9][0-1][0-9][0-3][0-9])', name.replace("-", ""))
#         date_int = int(date_match.group(1)) if date_match else 0
        
#         parsed_models.append({
#             "id": m.id,
#             "provider": provider,
#             "name": name,
#             "date": date_int
#         })

#     # 2. Sort: First by Provider (A-Z), then by Date (Newest First), then Name
#     # We use a tuple key: (provider, -date, name)
#     # "-date" makes it sort descending (newest first)
#     sorted_models = sorted(parsed_models, key=lambda x: (x["provider"], -x["date"], x["name"]))

#     # 3. Print grouped results
#     current_provider = ""
#     for item in sorted_models:
#         if item["provider"] != current_provider:
#             current_provider = item["provider"]
#             print(f"\n--- {current_provider.upper()} ---")
        
#         print(item["id"])

# if __name__ == "__main__":
#     asyncio.run(list_models())


from picnic.ai.llm import AsyncLLMClient
import asyncio
import re
import logging
import sys

# Configure logging to hide noisy HTTP request logs during the scan
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
# mute litellm logger if it's too chatty
logging.getLogger("litellm").setLevel(logging.CRITICAL)

async def verify_model(client, model_id, semaphore):
    """
    Tries to send a tiny 'ping' to the model. 
    Returns (model_id, True) if it works, (model_id, False) if 404/Auth error.
    """
    async with semaphore:
        try:
            # We send a minimal token request. We don't care about the answer,
            # just that it didn't crash with a 404.
            await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1
            )
            return model_id, True
        except Exception as e:
            # Check for specific 404 or Not Found errors
            error_str = str(e).lower()
            if "404" in error_str or "notfound" in error_str or "publisher model" in error_str:
                return model_id, False
            
            # If we get a 400 (Bad Request) or 429 (Rate Limit), the model EXISTS, 
            # it just didn't like our input or we are too fast. We count that as Available.
            if "429" in error_str or "400" in error_str:
                 return model_id, True
                 
            return model_id, False

async def list_available_models():
    print("Fetching model list (this may take a moment)...")
    client = AsyncLLMClient()
    
    # 1. Get the raw list of candidates
    try:
        models = await client.openai_client.models.list()
    except Exception as e:
        print(f"Failed to fetch model list: {e}")
        return

    # 2. Verify them in parallel
    # Use a Semaphore to limit us to ~10 concurrent checks so we don't get 
    # blocked for rate limiting while checking availability.
    sem = asyncio.Semaphore(10)
    
    print(f"Verifying access to {len(models.data)} models...")
    
    tasks = [verify_model(client.openai_client, m.id, sem) for m in models.data]
    results = await asyncio.gather(*tasks)
    
    # Filter to only verified models
    verified_ids = {r[0] for r in results if r[1] is True}

    # 3. Parse and Sort (Same logic as before, but only for verified models)
    parsed_models = []
    for m in models.data:
        if m.id not in verified_ids:
            continue
            
        if "/" in m.id:
            provider, name = m.id.split("/", 1)
        else:
            provider, name = "unknown", m.id

        date_match = re.search(r'(202[3-9][0-1][0-9][0-3][0-9])', name.replace("-", ""))
        date_int = int(date_match.group(1)) if date_match else 0
        
        parsed_models.append({
            "id": m.id,
            "provider": provider,
            "name": name,
            "date": date_int
        })

    # 4. Sort and Print
    sorted_models = sorted(parsed_models, key=lambda x: (x["provider"], -x["date"], x["name"]))

    current_provider = ""
    for item in sorted_models:
        if item["provider"] != current_provider:
            current_provider = item["provider"]
            print(f"\n--- {current_provider.upper()} ---")
        
        print(item["id"])

if __name__ == "__main__":
    asyncio.run(list_available_models())