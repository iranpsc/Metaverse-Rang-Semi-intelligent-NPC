import json
from channels.generic.websocket import AsyncWebsocketConsumer

class RAGChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("DEBUG: ECHO Consumer connected")
        await self.accept()

    async def disconnect(self, close_code):
        print(f"DEBUG: Disconnected {close_code}")

    async def receive(self, text_data):
        print(f"DEBUG: Received: {text_data}", flush=True) # <--- Force log to appear
        await self.send(text_data=json.dumps({'type': 'token', 'token': 'Echo: ' + text_data}))