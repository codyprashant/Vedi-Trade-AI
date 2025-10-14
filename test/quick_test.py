#!/usr/bin/env python3
import asyncio
import websockets
import json
from datetime import datetime

async def quick_test():
    uri = 'ws://localhost:8001/ws/prices?symbol=XAUUSD'
    try:
        async with websockets.connect(uri) as websocket:
            print('🔗 Connected to WebSocket endpoint')
            print('📡 Waiting for Yahoo Finance data...')
            print('=' * 50)
            
            # Get the first message (should be immediate)
            message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(message)
            
            print('📊 YAHOO FINANCE DATA RECEIVED:')
            print(json.dumps(data, indent=2))
            print('=' * 50)
            
            # Print analysis
            print('📋 DATA ANALYSIS:')
            print(f'Symbol: {data.get("symbol", "N/A")}')
            print(f'Timestamp: {data.get("time", "N/A")}')
            print(f'Bid Price: {data.get("bid", "N/A")}')
            
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    asyncio.run(quick_test())