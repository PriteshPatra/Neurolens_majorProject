"use client";

import { useSettings } from "@/context/SettingsContext";
import { Settings as SettingsIcon, Save } from "lucide-react";
import { useState } from "react";

export default function SettingsPage() {
  const { settings, updateSettings } = useSettings();
  const [localSettings, setLocalSettings] = useState(settings);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    updateSettings(localSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="flex items-center gap-2" style={{ marginBottom: '2rem' }}>
        <SettingsIcon size={32} color="var(--accent-primary)" />
        <h2 style={{ margin: 0 }}>System Configuration</h2>
      </div>

      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <h3 className="glass-header">Brightness Controls</h3>
        
        <div className="slider-container">
          <div className="slider-header">
            <span>Minimum Brightness</span>
            <span>{localSettings.minBrightness}%</span>
          </div>
          <input 
            type="range" 
            min="0" max="100" 
            value={localSettings.minBrightness}
            onChange={(e) => setLocalSettings({...localSettings, minBrightness: Number(e.target.value)})}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>The lowest screen brightness level allowed during high stress.</p>
        </div>

        <div className="slider-container">
          <div className="slider-header">
            <span>Maximum Brightness</span>
            <span>{localSettings.maxBrightness}%</span>
          </div>
          <input 
            type="range" 
            min="0" max="100" 
            value={localSettings.maxBrightness}
            onChange={(e) => setLocalSettings({...localSettings, maxBrightness: Number(e.target.value)})}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>The highest screen brightness level allowed when calm.</p>
        </div>

        <div className="slider-container">
          <div className="slider-header">
            <span>Smoothing Factor</span>
            <span>{localSettings.smoothingFactor.toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            min="0.01" max="1.0" step="0.01"
            value={localSettings.smoothingFactor}
            onChange={(e) => setLocalSettings({...localSettings, smoothingFactor: Number(e.target.value)})}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Controls how quickly the brightness adjusts. Lower is smoother.</p>
        </div>
      </div>

      <div className="glass-panel" style={{ marginBottom: '2rem' }}>
        <h3 className="glass-header">Notifications & Alerts</h3>
        
        <div className="slider-container">
          <div className="slider-header">
            <span>Stress Notification Threshold</span>
            <span style={{ color: localSettings.stressThreshold >= 80 ? 'var(--color-danger)' : 'var(--text-primary)' }}>
              {localSettings.stressThreshold}%
            </span>
          </div>
          <input 
            type="range" 
            min="0" max="100" 
            value={localSettings.stressThreshold}
            onChange={(e) => setLocalSettings({...localSettings, stressThreshold: Number(e.target.value)})}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Alert triggers when stress exceeds this level.</p>
        </div>

        <div className="slider-container">
          <div className="slider-header">
            <span>Notification Cooldown (Seconds)</span>
            <span>{localSettings.notificationCooldown}s</span>
          </div>
          <input 
            type="range" 
            min="5" max="60" 
            value={localSettings.notificationCooldown}
            onChange={(e) => setLocalSettings({...localSettings, notificationCooldown: Number(e.target.value)})}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Minimum time between consecutive high-stress alerts.</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button onClick={handleSave} className="btn btn-primary" style={{ flex: 1 }}>
          <Save size={18} /> Save Configuration
        </button>
        {saved && <span style={{ color: 'var(--color-success)' }}>Settings saved!</span>}
      </div>
    </div>
  );
}
