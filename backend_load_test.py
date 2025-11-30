# backend_load_test.py
import argparse
import asyncio
import time
import random
from llm_core import generate_text, generate_image

TEST_TEXTS = [
    "Напиши анекдот про 3 котов, русского, немца и француза",
    "Кратко опиши, как работает ГЭС?"
]
TEST_IMAGE = "Создай картинку сражения двух котов в образах рыцаря-джедая и ситха"

async def user_session(i, openai_key, grok_key, gemini_key):
    st = time.time()
    provider = random.choice(["openai", "grok", "gemini"])
    try:
        if random.random() < 0.8:
            # текстовый запрос
            res = await generate_text(provider=provider,
                                      model="gpt-4o-mini" if provider=="openai" else ("grok-2-mini" if provider=="grok" else "gemini-1.5-flash"),
                                      history=[],
                                      user_input=random.choice(TEST_TEXTS),
                                      openai_key=openai_key,
                                      grok_key=grok_key,
                                      gemini_key=gemini_key)
            return {"user": i, "provider": provider, "time": time.time()-st, "type": "text", "ok": True}
        else:
            img = await generate_image(provider=provider, prompt=TEST_IMAGE, openai_key=openai_key, grok_key=grok_key)
            return {"user": i, "provider": provider, "time": time.time()-st, "type": "image", "ok": True}
    except Exception as e:
        return {"user": i, "provider": provider, "time": time.time()-st, "type": "error", "ok": False, "err": str(e)}

async def main(users):
    openai_key = input("OPENAI_API_KEY: ").strip()
    grok_key = input("GROK_API_KEY: ").strip()
    gemini_key = input("GEMINI_API_KEY: ").strip()

    tasks = [user_session(i, openai_key, grok_key, gemini_key) for i in range(users)]
    results = await asyncio.gather(*tasks)
    fname = f"backend_test_{users}.log"
    with open(fname, "w", encoding="utf8") as f:
        for r in results:
            f.write(str(r) + "\n")
    print("Done. Results in", fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.users))
