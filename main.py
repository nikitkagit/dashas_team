import asyncio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client("https://openjiuwen.iir.nsu.ru/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(tools)

            res = await session.call_tool("create_registration_challenge", arguments={})
            data = json.loads(res.content[0].text)
            print(data)

asyncio.run(main())