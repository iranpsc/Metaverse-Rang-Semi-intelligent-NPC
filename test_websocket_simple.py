#!/usr/bin/env python3
"""
Simple WebSocket test to verify the connection works
"""

import asyncio
import websockets
import json


async def test_websocket():
    uri = "ws://localhost:8000/ws/audio-transcription/"

    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")

            # Wait for connection established message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📨 Received: {data}")

            # Send a test command
            test_command = {"command": "get_status"}
            await websocket.send(json.dumps(test_command))
            print("📤 Sent status command")

            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📨 Response: {response_data}")

            print("✅ WebSocket test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())

