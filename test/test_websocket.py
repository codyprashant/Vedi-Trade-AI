#!/usr/bin/env python3
"""
WebSocket client to test the /ws/prices endpoint and see what data we receive from Yahoo Finance
Extended version to capture multiple price updates
"""

import asyncio
import websockets
import json
from datetime import datetime

async def test_websocket_extended():
    uri = "ws://localhost:8001/ws/prices?symbol=XAUUSD"
    
    print(f"Connecting to: {uri}")
    print("=" * 60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            print("📡 Listening for live price data from Yahoo Finance...")
            print("⏱️  Note: Price updates come every ~120 seconds from Yahoo Finance")
            print("=" * 60)
            
            # Listen for messages for 5 minutes to catch multiple price updates
            message_count = 0
            start_time = asyncio.get_event_loop().time()
            last_price_time = None
            
            while True:
                try:
                    # Set a timeout to avoid hanging indefinitely
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    message_count += 1
                    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    print(f"📨 Message #{message_count} at {current_time}:")
                    
                    # Try to parse as JSON for pretty printing
                    try:
                        data = json.loads(message)
                        print(json.dumps(data, indent=2))
                        
                        # Track price updates vs heartbeats
                        if "bid" in data and data["bid"] is not None:
                            if last_price_time:
                                time_diff = asyncio.get_event_loop().time() - last_price_time
                                print(f"⏰ Time since last price update: {time_diff:.1f} seconds")
                            last_price_time = asyncio.get_event_loop().time()
                            print(f"💰 XAUUSD Bid Price: ${data['bid']:.2f}")
                        elif data.get("type") == "heartbeat":
                            print("💓 Heartbeat received")
                            
                    except json.JSONDecodeError:
                        print(f"Raw message: {message}")
                    
                    print("-" * 40)
                    
                    # Stop after 5 minutes or 20 messages
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > 300 or message_count >= 20:
                        print(f"🛑 Stopping after {elapsed:.1f} seconds and {message_count} messages")
                        break
                        
                except asyncio.TimeoutError:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"⏰ No message received in 10 seconds (elapsed: {elapsed:.1f}s), continuing to listen...")
                    if elapsed > 300:
                        print("🛑 Stopping after 5 minutes timeout")
                        break
                    continue
                    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("=" * 60)
    print("🏁 WebSocket test completed")

if __name__ == "__main__":
    asyncio.run(test_websocket_extended())