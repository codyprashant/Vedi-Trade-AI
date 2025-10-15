#!/usr/bin/env python3
"""
XAUUSD Signal Generation Analysis Test
=====================================
This script performs a comprehensive analysis of signal generation for XAUUSD,
including all parameters, decision metrics, and final efficiency calculations.
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
import pandas as pd

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.config import (
    INDICATOR_PARAMS, WEIGHTS, SIGNAL_THRESHOLD, 
    PRIMARY_TIMEFRAME, CONFIRMATION_TIMEFRAME, TREND_TIMEFRAME,
    ALIGNMENT_BOOST_H1, ALIGNMENT_BOOST_H4
)
from app.indicators import (
    compute_indicators, evaluate_signals, ema_trend_direction, 
    atr_last_and_mean, compute_strategy_strength, best_signal
)
import yfinance as yf

class XAUUSDSignalAnalyzer:
    def __init__(self):
        self.symbol = "GC=F"  # Yahoo Finance symbol for Gold Futures (XAUUSD equivalent)
        self.analysis_results = {}
        
    def print_separator(self, title):
        """Print a formatted separator with title"""
        print("\n" + "="*80)
        print(f" {title} ".center(80, "="))
        print("="*80)
        
    def print_subsection(self, title):
        """Print a formatted subsection"""
        print(f"\n--- {title} ---")
        
    async def fetch_market_data(self):
        """Fetch market data for all required timeframes"""
        self.print_separator("FETCHING MARKET DATA")
        
        try:
            # Fetch data for different timeframes
            timeframes = {
                "15m": "15m",
                "1h": "1h", 
                "4h": "4h"
            }
            
            self.market_data = {}
            
            for tf_name, tf_code in timeframes.items():
                print(f"Fetching {tf_name} data for {self.symbol}...")
                
                # Calculate period based on timeframe
                if tf_name == "15m":
                    period = "5d"  # 5 days of 15m data
                elif tf_name == "1h":
                    period = "30d"  # 30 days of 1h data
                else:  # 4h
                    period = "60d"  # 60 days of 4h data
                
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period=period, interval=tf_code)
                
                if df.empty:
                    print(f"❌ No data available for {tf_name}")
                    continue
                    
                # Add time column and normalize column names
                df = df.reset_index()
                df['time'] = df['Datetime']
                
                # Normalize column names to lowercase for indicators
                df.columns = [col.lower() if col != 'time' else col for col in df.columns]
                
                self.market_data[tf_name] = df
                print(f"✅ {tf_name}: {len(df)} bars, Latest: {df.iloc[-1]['time']}")
                
        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            return False
            
        return True
        
    def analyze_configuration(self):
        """Analyze current configuration parameters"""
        self.print_separator("CONFIGURATION ANALYSIS")
        
        print("📊 TIMEFRAME CONFIGURATION:")
        print(f"  Primary Timeframe: {PRIMARY_TIMEFRAME}")
        print(f"  Confirmation Timeframe: {CONFIRMATION_TIMEFRAME}")
        print(f"  Trend Timeframe: {TREND_TIMEFRAME}")
        
        print("\n🎯 SIGNAL THRESHOLD:")
        print(f"  Minimum Signal Strength: {SIGNAL_THRESHOLD}%")
        
        print("\n⚡ ALIGNMENT BOOSTS:")
        print(f"  H1 Alignment Boost: {ALIGNMENT_BOOST_H1}%")
        print(f"  H4 Alignment Boost: {ALIGNMENT_BOOST_H4}%")
        
        print("\n📈 INDICATOR PARAMETERS:")
        for indicator, params in INDICATOR_PARAMS.items():
            print(f"  {indicator}: {params}")
            
        print("\n⚖️ INDICATOR WEIGHTS:")
        total_weight = sum(WEIGHTS.values())
        for category, weight in WEIGHTS.items():
            percentage = (weight / total_weight) * 100
            print(f"  {category}: {weight} ({percentage:.1f}%)")
            
        print(f"\n📊 Total Weight Sum: {total_weight}")
        
    def compute_indicators_detailed(self, df, timeframe):
        """Compute indicators with detailed analysis"""
        self.print_subsection(f"COMPUTING INDICATORS - {timeframe}")
        
        try:
            # Compute indicators
            indicators_dict = compute_indicators(df, INDICATOR_PARAMS)
            
            if not indicators_dict:
                print(f"❌ No indicators computed for {timeframe}")
                return None, None
                
            # Get the latest indicator values
            latest_indicators = {}
            for indicator_name, series in indicators_dict.items():
                if len(series) > 0 and pd.notna(series.iloc[-1]):
                    latest_indicators[indicator_name] = series.iloc[-1]
                else:
                    latest_indicators[indicator_name] = None
                
            print(f"✅ Computed {len(latest_indicators)} indicators")
            
            # Print each indicator value
            for indicator, value in latest_indicators.items():
                if pd.notna(value):
                    print(f"  {indicator}: {value:.4f}")
                else:
                    print(f"  {indicator}: NaN")
                    
            # Evaluate signals
            evaluation = evaluate_signals(df, indicators_dict, INDICATOR_PARAMS)
            
            print(f"\n📊 SIGNAL EVALUATION:")
            # Extract directions from IndicatorResult objects for display
            for category, indicator_result in evaluation.items():
                direction = indicator_result.direction
                print(f"  {category}: {direction}")
                
            return latest_indicators, evaluation
            
        except Exception as e:
            print(f"❌ Error computing indicators for {timeframe}: {e}")
            return None, None
            
    def analyze_signal_strength(self, evaluation_results):
        """Analyze signal strength using EXACT production logic."""
        self.print_subsection("SIGNAL STRENGTH ANALYSIS (PRODUCTION LOGIC)")
        
        # Count signals by type (for display)
        evaluation_directions = {k: v.direction for k, v in evaluation_results.items()}
        buy_signals = sum(1 for signal in evaluation_directions.values() if signal == "buy")
        sell_signals = sum(1 for signal in evaluation_directions.values() if signal == "sell")
        neutral_signals = sum(1 for signal in evaluation_directions.values() if signal == "none")
        
        print(f"📊 SIGNAL DISTRIBUTION:")
        print(f"  Buy signals: {buy_signals}")
        print(f"  Sell signals: {sell_signals}")
        print(f"  Neutral signals: {neutral_signals}")
        print(f"  Total categories: {len(evaluation_directions)}")
        
        # Use EXACT production logic: compute_strategy_strength + best_signal
        strategies = compute_strategy_strength(evaluation_results, WEIGHTS)
        best = best_signal(strategies)
        
        print(f"\n💪 STRATEGY ANALYSIS (PRODUCTION):")
        for strategy_name, strategy_data in strategies.items():
            print(f"  {strategy_name.upper()}:")
            print(f"    Direction: {strategy_data['direction']}")
            print(f"    Strength: {strategy_data['strength']:.2f}%")
            print(f"    Contributions: {strategy_data['contributions']}")
        
        print(f"\n🎯 BEST STRATEGY (PRODUCTION):")
        if best:
            print(f"  Strategy: {best['strategy']}")
            print(f"  Direction: {best['direction']}")
            print(f"  Strength: {best['strength']:.2f}%")
            print(f"  Contributions: {best['contributions']}")
        else:
            print(f"  No valid strategy found")
            
        return {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "neutral_signals": neutral_signals,
            "strategies": strategies,
            "best_strategy": best,
            "best_direction": best['direction'] if best else None,
            "base_strength": best['strength'] if best else 0,
            "evaluation_details": evaluation_directions
        }
        
    def analyze_trend_alignment(self):
        """Analyze trend alignment and boosts"""
        self.print_subsection("TREND ALIGNMENT ANALYSIS")
        
        try:
            # Analyze H1 trend
            h1_df = self.market_data.get("1h")
            h4_df = self.market_data.get("4h")
            
            if h1_df is None or h4_df is None:
                print("❌ Missing H1 or H4 data for trend analysis")
                return None, None, 0
                
            # Use the existing ema_trend_direction function
            def convert_trend_to_signal(trend_result):
                if trend_result == "Bullish":
                    return "buy"
                elif trend_result == "Bearish":
                    return "sell"
                else:
                    return "neutral"
                    
            h1_trend_raw = ema_trend_direction(h1_df, short_len=50, long_len=200)
            h4_trend_raw = ema_trend_direction(h4_df, short_len=50, long_len=200)
            
            h1_trend = convert_trend_to_signal(h1_trend_raw)
            h4_trend = convert_trend_to_signal(h4_trend_raw)
            
            print(f"📈 H1 TREND (EMA 50/200): {h1_trend}")
            print(f"📈 H4 TREND (EMA 50/200): {h4_trend}")
            
            # Calculate alignment boost
            alignment_boost = 0
            
            if h1_trend in ["buy", "sell"]:
                alignment_boost += ALIGNMENT_BOOST_H1
                print(f"✅ H1 alignment boost: +{ALIGNMENT_BOOST_H1}%")
            else:
                print(f"❌ No H1 alignment boost")
                
            if h4_trend in ["buy", "sell"]:
                alignment_boost += ALIGNMENT_BOOST_H4
                print(f"✅ H4 alignment boost: +{ALIGNMENT_BOOST_H4}%")
            else:
                print(f"❌ No H4 alignment boost")
                
            print(f"\n🚀 TOTAL ALIGNMENT BOOST: +{alignment_boost}%")
            
            return h1_trend, h4_trend, alignment_boost
            
        except Exception as e:
            print(f"❌ Error analyzing trend alignment: {e}")
            return None, None, 0
            
    def calculate_final_efficiency(self, base_strength, alignment_boost):
        """Calculate final efficiency with all boosts"""
        self.print_subsection("FINAL EFFICIENCY CALCULATION")
        
        final_strength = min(100.0, base_strength + alignment_boost)
        
        print(f"📊 EFFICIENCY BREAKDOWN:")
        print(f"  Base Strength: {base_strength:.2f}%")
        print(f"  Alignment Boost: +{alignment_boost:.2f}%")
        print(f"  Raw Total: {base_strength + alignment_boost:.2f}%")
        print(f"  Final Efficiency (capped): {final_strength:.2f}%")
        
        # Check if signal meets threshold
        meets_threshold = final_strength >= SIGNAL_THRESHOLD
        print(f"\n🎯 THRESHOLD CHECK:")
        print(f"  Required: {SIGNAL_THRESHOLD}%")
        print(f"  Achieved: {final_strength:.2f}%")
        print(f"  Meets Threshold: {'✅ YES' if meets_threshold else '❌ NO'}")
        
        return final_strength, meets_threshold
        
    async def run_comprehensive_analysis(self):
        """Run the complete signal analysis"""
        self.print_separator("XAUUSD SIGNAL GENERATION ANALYSIS")
        print(f"🕐 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏆 Symbol: {self.symbol}")
        
        # Step 1: Fetch market data
        if not await self.fetch_market_data():
            print("❌ Failed to fetch market data. Aborting analysis.")
            return
            
        # Step 2: Analyze configuration
        self.analyze_configuration()
        
        # Step 3: Compute indicators for primary timeframe (15m)
        primary_df = self.market_data.get("15m")
        if primary_df is None:
            print("❌ No primary timeframe data available")
            return
            
        indicators, evaluation = self.compute_indicators_detailed(primary_df, "15m")
        if indicators is None or evaluation is None:
            print("❌ Failed to compute indicators")
            return
            
        # Step 4: Analyze signal strength
        strength_analysis = self.analyze_signal_strength(evaluation)
        
        # Step 5: Analyze trend alignment
        h1_trend, h4_trend, alignment_boost = self.analyze_trend_alignment()
        
        # Step 6: Calculate final efficiency using PRODUCTION logic
        if strength_analysis["best_direction"] and strength_analysis["best_direction"] in ("buy", "sell"):
            final_efficiency, meets_threshold = self.calculate_final_efficiency(
                strength_analysis["base_strength"], 
                alignment_boost
            )
        else:
            final_efficiency = 0
            meets_threshold = False
            
        # Step 7: Generate final report
        self.generate_final_report(strength_analysis, h1_trend, h4_trend, 
                                 alignment_boost, final_efficiency, meets_threshold)
        
    def generate_final_report(self, strength_analysis, h1_trend, h4_trend, 
                            alignment_boost, final_efficiency, meets_threshold):
        """Generate comprehensive final report"""
        self.print_separator("FINAL ANALYSIS REPORT")
        
        print("🎯 SIGNAL DECISION SUMMARY:")
        print(f"  Symbol: {self.symbol}")
        print(f"  Primary Direction: {strength_analysis['best_direction']}")
        print(f"  Base Strength: {strength_analysis['base_strength']:.2f}%")
        print(f"  H1 Trend: {h1_trend}")
        print(f"  H4 Trend: {h4_trend}")
        print(f"  Alignment Boost: +{alignment_boost:.2f}%")
        print(f"  Final Efficiency: {final_efficiency:.2f}%")
        print(f"  Signal Generated: {'✅ YES' if meets_threshold else '❌ NO'}")
        
        print(f"\n📊 DETAILED METRICS:")
        print(f"  Buy Signals: {strength_analysis['buy_signals']}")
        print(f"  Sell Signals: {strength_analysis['sell_signals']}")
        print(f"  Neutral Signals: {strength_analysis['neutral_signals']}")
        print(f"  Base Strength: {strength_analysis['base_strength']:.2f}%")
        print(f"  Best Strategy: {strength_analysis['best_strategy']['strategy'] if strength_analysis['best_strategy'] else 'None'}")
        
        print(f"\n🔍 ANALYSIS INSIGHTS:")
        
        # Signal quality assessment
        if final_efficiency >= 80:
            quality = "EXCELLENT"
        elif final_efficiency >= 70:
            quality = "GOOD"
        elif final_efficiency >= 60:
            quality = "MODERATE"
        elif final_efficiency >= 50:
            quality = "WEAK"
        else:
            quality = "POOR"
            
        print(f"  Signal Quality: {quality}")
        
        # Trend alignment assessment
        if h1_trend == h4_trend and h1_trend in ["buy", "sell"]:
            trend_alignment = "STRONG (H1 & H4 aligned)"
        elif h1_trend in ["buy", "sell"] or h4_trend in ["buy", "sell"]:
            trend_alignment = "MODERATE (partial alignment)"
        else:
            trend_alignment = "WEAK (no clear trend)"
            
        print(f"  Trend Alignment: {trend_alignment}")
        
        # Recommendation
        if meets_threshold and final_efficiency >= 70:
            recommendation = "STRONG SIGNAL - Consider trading"
        elif meets_threshold and final_efficiency >= 60:
            recommendation = "MODERATE SIGNAL - Trade with caution"
        elif meets_threshold:
            recommendation = "WEAK SIGNAL - High risk"
        else:
            recommendation = "NO SIGNAL - Do not trade"
            
        print(f"  Recommendation: {recommendation}")
        
        print(f"\n⚠️ RISK ASSESSMENT:")
        risk_factors = []
        
        if strength_analysis['neutral_signals'] > strength_analysis['buy_signals'] + strength_analysis['sell_signals']:
            risk_factors.append("High number of neutral signals")
            
        if alignment_boost == 0:
            risk_factors.append("No trend alignment boost")
            
        # Check for conflicting signals based on strategy analysis
        if strength_analysis['best_strategy'] and strength_analysis['base_strength'] < 20:
            risk_factors.append("Weak strategy strength")
            
        if final_efficiency < 60:
            risk_factors.append("Low signal efficiency")
            
        if risk_factors:
            for factor in risk_factors:
                print(f"  ⚠️ {factor}")
        else:
            print(f"  ✅ No major risk factors identified")
            
        self.print_separator("ANALYSIS COMPLETE")

async def main():
    """Main function to run the analysis"""
    analyzer = XAUUSDSignalAnalyzer()
    await analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    asyncio.run(main())