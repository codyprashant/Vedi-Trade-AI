import React, { useState, useEffect } from "react";
import { Settings, Activity, TrendingUp, Clock, Zap, CheckCircle2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { strategyApi } from "../components/services/strategyApi";
import IndicatorEditor from "../components/strategies/IndicatorEditor";
import WeightsEditor from "../components/strategies/WeightsEditor";

export default function Strategies() {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [strategyDetail, setStrategyDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [saving, setSaving] = useState(false);

  // Fetch strategies list
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const data = await strategyApi.listStrategies();
        setStrategies(data.strategies || []);
        if (data.strategies && data.strategies.length > 0) {
          setSelectedStrategy(data.strategies[0].id);
        }
      } catch (error) {
        console.error('Error fetching strategies:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStrategies();
  }, []);

  // Fetch strategy details when selected
  useEffect(() => {
    if (!selectedStrategy) return;

    const fetchDetail = async () => {
      try {
        const data = await strategyApi.getStrategy(selectedStrategy);
        setStrategyDetail(data);
      } catch (error) {
        console.error('Error fetching strategy detail:', error);
      }
    };

    fetchDetail();
  }, [selectedStrategy]);

  const handleActivateStrategy = async (strategyId) => {
    try {
      await strategyApi.activateStrategy(strategyId);
      // Refresh strategies list
      const data = await strategyApi.listStrategies();
      setStrategies(data.strategies || []);
    } catch (error) {
      console.error('Error activating strategy:', error);
    }
  };

  const handleSaveIndicator = async (indicatorName, params) => {
    try {
      await strategyApi.updateIndicator(selectedStrategy, indicatorName, params);
      // Refresh strategy detail
      const data = await strategyApi.getStrategy(selectedStrategy);
      setStrategyDetail(data);
    } catch (error) {
      console.error('Error saving indicator:', error);
      throw error;
    }
  };

  const handleSaveWeights = async (weights) => {
    try {
      await strategyApi.updateWeights(selectedStrategy, weights);
      // Refresh strategy detail
      const data = await strategyApi.getStrategy(selectedStrategy);
      setStrategyDetail(data);
    } catch (error) {
      console.error('Error saving weights:', error);
      throw error;
    }
  };

  const handleSaveSchedule = async (interval) => {
    setSaving(true);
    try {
      await strategyApi.updateSchedule(selectedStrategy, parseInt(interval));
      // Refresh strategy detail
      const data = await strategyApi.getStrategy(selectedStrategy);
      setStrategyDetail(data);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveThreshold = async (threshold) => {
    setSaving(true);
    try {
      await strategyApi.updateThreshold(selectedStrategy, parseFloat(threshold));
      // Refresh strategy detail
      const data = await strategyApi.getStrategy(selectedStrategy);
      setStrategyDetail(data);
    } finally {
      setSaving(false);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'indicators', label: 'Indicators', icon: TrendingUp },
    { id: 'weights', label: 'Weights', icon: Zap },
    { id: 'schedule', label: 'Schedule', icon: Clock },
  ];

  if (loading) {
    return (
      <div className="min-h-screen p-4 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'rgb(var(--theme-primary))' }} />
      </div>
    );
  }

  return (
    <div className="min-h-screen p-4">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-1">
          <div 
            className="w-8 h-8 rounded-xl flex items-center justify-center shadow-lg"
            style={{ 
              background: 'linear-gradient(to bottom right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
              boxShadow: '0 10px 25px rgba(var(--theme-primary), 0.25)'
            }}
          >
            <Settings className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Strategy Configuration</h1>
            <p className="text-xs text-white/60">Configure trading strategies and indicator parameters</p>
          </div>
        </div>
      </div>

      {/* Strategy Selector */}
      <div className="mb-4">
        <div 
          className="rounded-xl p-3 backdrop-blur-xl"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
          }}
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {strategies.map(strategy => (
              <button
                key={strategy.id}
                onClick={() => setSelectedStrategy(strategy.id)}
                className={`p-3 rounded-lg transition-all duration-300 text-left`}
                style={{
                  background: selectedStrategy === strategy.id 
                    ? 'linear-gradient(to right, rgba(var(--theme-primary), 0.2), rgba(var(--theme-secondary), 0.2))' 
                    : 'rgba(255, 255, 255, 0.05)',
                  border: `1px solid ${selectedStrategy === strategy.id ? 'rgba(var(--theme-primary), 0.3)' : 'transparent'}`
                }}
              >
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-sm font-bold text-white">{strategy.name}</h3>
                  {strategy.is_active && (
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs text-white/60">
                  <span>Threshold: {(strategy.signal_threshold * 100).toFixed(0)}%</span>
                  <span>Interval: {strategy.run_interval_seconds}s</span>
                </div>
                {!strategy.is_active && (
                  <Button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleActivateStrategy(strategy.id);
                    }}
                    size="sm"
                    className="mt-2 w-full"
                    style={{
                      background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))'
                    }}
                  >
                    Activate
                  </Button>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      {strategyDetail && (
        <>
          <div 
            className="flex gap-2 p-1 rounded-xl backdrop-blur-xl mb-4"
            style={{ 
              background: 'rgba(0, 0, 0, 0.3)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
            }}
          >
            {tabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all duration-300`}
                  style={{
                    background: activeTab === tab.id 
                      ? 'linear-gradient(to right, rgba(var(--theme-primary), 0.3), rgba(var(--theme-secondary), 0.3))' 
                      : 'transparent',
                    border: activeTab === tab.id ? '1px solid rgba(var(--theme-primary), 0.3)' : '1px solid transparent'
                  }}
                >
                  <Icon className={`w-4 h-4 ${activeTab === tab.id ? 'text-white' : 'text-white/50'}`} />
                  <span className={`text-sm font-medium ${activeTab === tab.id ? 'text-white' : 'text-white/50'}`}>
                    {tab.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Tab Content */}
          <div>
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div 
                className="rounded-xl p-4 backdrop-blur-xl"
                style={{ 
                  background: 'rgba(0, 0, 0, 0.3)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
                }}
              >
                <h3 className="text-base font-bold text-white mb-4">{strategyDetail.strategy.name}</h3>
                <p className="text-sm text-white/70 mb-4">{strategyDetail.strategy.description}</p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div 
                    className="p-3 rounded-lg"
                    style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                  >
                    <div className="text-xs text-white/50 mb-1">Status</div>
                    <div className={`text-sm font-bold ${strategyDetail.strategy.is_active ? 'text-green-400' : 'text-white/70'}`}>
                      {strategyDetail.strategy.is_active ? 'Active' : 'Inactive'}
                    </div>
                  </div>
                  <div 
                    className="p-3 rounded-lg"
                    style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                  >
                    <div className="text-xs text-white/50 mb-1">Threshold</div>
                    <div className="text-sm font-bold text-white">
                      {(strategyDetail.strategy.signal_threshold * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div 
                    className="p-3 rounded-lg"
                    style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                  >
                    <div className="text-xs text-white/50 mb-1">Run Interval</div>
                    <div className="text-sm font-bold text-white">
                      {strategyDetail.strategy.run_interval_seconds}s
                    </div>
                  </div>
                  <div 
                    className="p-3 rounded-lg"
                    style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                  >
                    <div className="text-xs text-white/50 mb-1">Timeframes</div>
                    <div className="text-sm font-bold text-white">
                      {strategyDetail.strategy.timeframes.join(', ')}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Indicators Tab */}
            {activeTab === 'indicators' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(strategyDetail.indicator_params).map(([name, params]) => (
                  <IndicatorEditor
                    key={name}
                    indicatorName={name}
                    params={params}
                    onSave={(updatedParams) => handleSaveIndicator(name, updatedParams)}
                  />
                ))}
              </div>
            )}

            {/* Weights Tab */}
            {activeTab === 'weights' && (
              <WeightsEditor
                weights={strategyDetail.weights}
                onSave={handleSaveWeights}
              />
            )}

            {/* Schedule Tab */}
            {activeTab === 'schedule' && (
              <div 
                className="rounded-xl p-4 backdrop-blur-xl"
                style={{ 
                  background: 'rgba(0, 0, 0, 0.3)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
                }}
              >
                <h3 className="text-sm font-bold text-white mb-4">Run Schedule & Signal Threshold</h3>
                
                <div className="space-y-4">
                  {/* Run Interval */}
                  <div>
                    <label className="text-xs text-white/70 mb-2 block">
                      Run Interval (seconds)
                    </label>
                    <div className="flex gap-2">
                      <Input
                        type="number"
                        defaultValue={strategyDetail.strategy.run_interval_seconds}
                        onBlur={(e) => {
                          if (e.target.value !== strategyDetail.strategy.run_interval_seconds.toString()) {
                            handleSaveSchedule(e.target.value);
                          }
                        }}
                        className="bg-white/5 border-white/10 text-white"
                        min="1"
                      />
                    </div>
                    <p className="text-xs text-white/50 mt-1">
                      How often the strategy engine evaluates signals
                    </p>
                  </div>

                  {/* Signal Threshold */}
                  <div>
                    <label className="text-xs text-white/70 mb-2 block">
                      Signal Efficiency Threshold
                    </label>
                    <div className="flex gap-4 items-center">
                      <Input
                        type="range"
                        min="0"
                        max="100"
                        step="1"
                        defaultValue={strategyDetail.strategy.signal_threshold * 100}
                        onMouseUp={(e) => {
                          const newValue = parseFloat(e.target.value) / 100;
                          if (newValue !== strategyDetail.strategy.signal_threshold) {
                            handleSaveThreshold(newValue);
                          }
                        }}
                        className="flex-1"
                        style={{
                          accentColor: 'rgb(var(--theme-primary))'
                        }}
                      />
                      <span className="text-sm font-bold text-white w-12 text-right">
                        {(strategyDetail.strategy.signal_threshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-xs text-white/50 mt-1">
                      Minimum signal strength required to generate a trading signal
                    </p>
                  </div>

                  {saving && (
                    <div className="flex items-center gap-2 text-xs text-green-400">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span>Saving changes...</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}