import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Save, RotateCcw, Loader2 } from "lucide-react";

export default function IndicatorEditor({ indicatorName, params, onSave }) {
  const [editedParams, setEditedParams] = useState(params);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const handleChange = (key, value) => {
    setEditedParams(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(editedParams);
      setHasChanges(false);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setEditedParams(params);
    setHasChanges(false);
  };

  return (
    <div 
      className="rounded-xl p-4 backdrop-blur-xl"
      style={{ 
        background: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}
    >
      <h4 className="text-sm font-bold text-white mb-3">{indicatorName}</h4>
      
      <div className="grid grid-cols-2 gap-3 mb-3">
        {Object.entries(editedParams).map(([key, value]) => (
          <div key={key}>
            <label className="text-xs text-white/70 mb-1 block capitalize">
              {key.replace('_', ' ')}
            </label>
            <Input
              type="number"
              value={value}
              onChange={(e) => handleChange(key, e.target.value)}
              className="bg-white/5 border-white/10 text-white text-sm"
              step={key.includes('stddev') ? '0.1' : '1'}
            />
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
                Save
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