import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Save, RotateCcw, Loader2 } from "lucide-react";

export default function WeightsEditor({ weights, onSave }) {
  const [editedWeights, setEditedWeights] = useState(weights);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const totalWeight = Object.values(editedWeights).reduce((sum, val) => sum + val, 0);

  const handleChange = (key, value) => {
    setEditedWeights(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    if (Math.abs(totalWeight - 1.0) > 0.01) {
      alert('Weights must sum to 1.0 (100%)');
      return;
    }

    setSaving(true);
    try {
      await onSave(editedWeights);
      setHasChanges(false);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setEditedWeights(weights);
    setHasChanges(false);
  };

  return (
    <div 
      className="rounded-xl p-4 backdrop-blur-xl"
      style={{ 
        background: 'rgba(0, 0, 0, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
      }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-white">Contribution Weights</h3>
        <div className="text-xs">
          <span className="text-white/50">Total: </span>
          <span className={`font-bold ${Math.abs(totalWeight - 1.0) > 0.01 ? 'text-red-400' : 'text-green-400'}`}>
            {(totalWeight * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        {Object.entries(editedWeights).map(([key, value]) => (
          <div key={key}>
            <label className="text-xs text-white/70 mb-1 block">{key}</label>
            <div className="flex gap-2 items-center">
              <Input
                type="number"
                value={value}
                onChange={(e) => handleChange(key, e.target.value)}
                className="bg-white/5 border-white/10 text-white text-sm"
                step="0.001"
                min="0"
                max="1"
              />
              <span className="text-xs text-white/50 whitespace-nowrap">
                {(value * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {hasChanges && (
        <div className="flex gap-2">
          <Button
            onClick={handleSave}
            disabled={saving}
            size="sm"
            className="flex-1"
            style={{
              background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))'
            }}
          >
            {saving ? (
              <>
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-3 h-3 mr-1" />
                Save Weights
              </>
            )}
          </Button>
          <Button
            onClick={handleReset}
            disabled={saving}
            size="sm"
            variant="outline"
            className="border-white/10"
          >
            <RotateCcw className="w-3 h-3" />
          </Button>
        </div>
      )}
    </div>
  );
}