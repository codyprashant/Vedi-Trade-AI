#!/usr/bin/env python3
"""
Comprehensive test script to capture and document Yahoo Finance fast_info response
for XAUUSD (GC=F) and other symbols for documentation purposes.
"""

import yfinance as yf
import json
from datetime import datetime
import pprint

def test_fastinfo_complete():
    """Test fast_info for multiple symbols to document the response structure."""
    
    # Test symbols and their Yahoo Finance equivalents
    test_symbols = {
        "XAUUSD": "GC=F",  # Gold futures
        "USDCAD": "USDCAD=X",  # Forex pair
        "GBPUSD": "GBPUSD=X",  # Forex pair
        "SPX": "^GSPC",  # S&P 500 index
        "AAPL": "AAPL"  # Stock
    }
    
    print("=" * 80)
    print("YAHOO FINANCE FAST_INFO COMPREHENSIVE TEST")
    print("=" * 80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_results = {}
    
    for symbol, yahoo_symbol in test_symbols.items():
        print(f"🔍 Testing {symbol} ({yahoo_symbol})")
        print("-" * 60)
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            fi = getattr(ticker, "fast_info", None)
            
            result = {
                "symbol": symbol,
                "yahoo_symbol": yahoo_symbol,
                "fast_info_available": fi is not None,
                "fast_info_type": str(type(fi)) if fi else None,
                "timestamp": datetime.now().isoformat()
            }
            
            if fi:
                # Get all available keys
                if hasattr(fi, 'keys'):
                    keys = list(fi.keys())
                    result["available_keys"] = keys
                    result["key_count"] = len(keys)
                    
                    # Get all key-value pairs
                    key_values = {}
                    for key in keys:
                        try:
                            value = fi.get(key)
                            key_values[key] = {
                                "value": value,
                                "type": str(type(value)),
                                "is_none": value is None
                            }
                        except Exception as e:
                            key_values[key] = {
                                "error": str(e),
                                "type": "error"
                            }
                    
                    result["key_values"] = key_values
                    
                    print(f"✅ fast_info available with {len(keys)} keys")
                    print(f"📋 Available keys: {keys}")
                    print()
                    print("📊 Key-Value Details:")
                    for key, info in key_values.items():
                        if "error" in info:
                            print(f"  ❌ {key}: ERROR - {info['error']}")
                        else:
                            value_str = str(info['value'])
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                            print(f"  📈 {key}: {value_str} ({info['type']})")
                else:
                    result["error"] = "fast_info is not dict-like"
                    print(f"❌ fast_info is not dict-like: {fi}")
            else:
                result["error"] = "fast_info is None"
                print("❌ fast_info is None")
            
            all_results[symbol] = result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error testing {symbol}: {error_msg}")
            all_results[symbol] = {
                "symbol": symbol,
                "yahoo_symbol": yahoo_symbol,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        
        print()
    
    # Print summary
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    for symbol, result in all_results.items():
        if "error" in result:
            print(f"❌ {symbol}: {result['error']}")
        else:
            key_count = result.get("key_count", 0)
            print(f"✅ {symbol}: {key_count} keys available")
    
    print()
    
    # Save detailed results to JSON file
    output_file = "fastinfo_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"💾 Detailed results saved to: {output_file}")
    
    # Focus on XAUUSD for detailed output
    if "XAUUSD" in all_results and "key_values" in all_results["XAUUSD"]:
        print()
        print("=" * 80)
        print("🥇 XAUUSD (GC=F) DETAILED FAST_INFO RESPONSE")
        print("=" * 80)
        
        xau_data = all_results["XAUUSD"]["key_values"]
        for key, info in xau_data.items():
            if "error" not in info:
                print(f"{key:25} : {info['value']} ({info['type']})")
    
    return all_results

if __name__ == "__main__":
    results = test_fastinfo_complete()