import React, { useState } from "react";
import { History, Filter } from "lucide-react";
import { APP_CONFIG } from "../components/config/appConfig";
import RecentSignalsTable from "../components/trading/RecentSignalsTable";

export default function SignalHistory() {
  const [filter, setFilter] = useState("all");

  return (
    <div className="min-h-screen p-4">
      {/* Header */}
      <div className="mb-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div 
              className="w-8 h-8 rounded-xl flex items-center justify-center shadow-lg"
              style={{ 
                background: 'linear-gradient(to bottom right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
                boxShadow: '0 10px 25px rgba(var(--theme-primary), 0.25)'
              }}
            >
              <History className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Signal History</h1>
              <p className="text-xs text-white/60">Track and analyze past trading signals</p>
            </div>
          </div>

          {/* Filters */}
          <div 
            className="flex gap-1.5 p-1 rounded-xl backdrop-blur-xl"
            style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
            }}
          >
            {["all", "BUY", "SELL", "HOLD"].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-300`}
                style={{
                  background: filter === f 
                    ? 'linear-gradient(to right, rgba(var(--theme-primary), 0.3), rgba(var(--theme-secondary), 0.3))' 
                    : 'transparent',
                  color: filter === f ? 'white' : 'rgba(255, 255, 255, 0.6)'
                }}
              >
                {f === "all" ? "All" : f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Signals Table - now using API */}
      <RecentSignalsTable limit={APP_CONFIG.SIGNAL_LIMITS.SIGNAL_HISTORY} />
    </div>
  );
}