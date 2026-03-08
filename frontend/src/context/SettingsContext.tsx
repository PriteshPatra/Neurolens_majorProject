"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface Settings {
  minBrightness: number;
  maxBrightness: number;
  smoothingFactor: number;
  stressThreshold: number;
  notificationCooldown: number;
}

interface SettingsContextType {
  settings: Settings;
  updateSettings: (newSettings: Partial<Settings>) => void;
}

const defaultSettings: Settings = {
  minBrightness: 30,
  maxBrightness: 100,
  smoothingFactor: 0.1,
  stressThreshold: 80,
  notificationCooldown: 10,
};

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(defaultSettings);

  // Load from localeStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('neurolens_settings');
    if (saved) {
      try {
        setSettings(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse settings", e);
      }
    }
  }, []);

  const updateSettings = (newSettings: Partial<Settings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...newSettings };
      localStorage.setItem('neurolens_settings', JSON.stringify(updated));
      return updated;
    });
  };

  return (
    <SettingsContext.Provider value={{ settings, updateSettings }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}
