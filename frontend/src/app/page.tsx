"use client";

import { useState, useCallback, useEffect } from 'react';
import WebcamCapture from '@/components/WebcamCapture';
import EmotionDisplay from '@/components/EmotionDisplay';
import StressGauge from '@/components/StressGauge';
import RealTimeChart from '@/components/RealTimeChart';
import { useSettings } from '@/context/SettingsContext';
import { computeBrightness } from '@/lib/api';
import { AlertTriangle, Power } from 'lucide-react';

export default function Dashboard() {
  const { settings } = useSettings();
  const [isActive, setIsActive] = useState(false);
  
  const [currentEmotion, setCurrentEmotion] = useState('N/A');
  const [currentConfidence, setCurrentConfidence] = useState(0);
  const [currentStress, setCurrentStress] = useState(0);
  
  const [stressHistory, setStressHistory] = useState<number[]>([]);
  const [fullSessionStress, setFullSessionStress] = useState<number[]>([]);
  
  const [adaptiveBrightness, setAdaptiveBrightness] = useState<number>(100);
  
  const [notification, setNotification] = useState<string | null>(null);
  const [lastNotificationTime, setLastNotificationTime] = useState<number>(0);

  const [emotionHistory, setEmotionHistory] = useState<string[]>([]);

  // Request browser notification permissions on mount
  useEffect(() => {
    if ('Notification' in window && Notification.permission !== 'granted' && Notification.permission !== 'denied') {
      Notification.requestPermission();
    }
  }, []);

  const handleAnalyticsUpdate = useCallback(async (data: { emotion: string; stress: number; confidence: number }) => {
    setCurrentConfidence(data.confidence);

    setEmotionHistory(prev => {
      const newHistory = [...prev, data.emotion];
      if (newHistory.length > 30) newHistory.shift(); // 6 fps * 5 seconds = 30 frames

      if (newHistory.length > 0) {
        // 1. Fast Emotion tracking (last 6 frames = 1 second) for UI responsiveness
        const recentHistory = newHistory.slice(-6);
        const counts: Record<string, number> = {};
        let maxCount = 0;
        let fastEmotion = 'N/A';
        for (const em of recentHistory) {
          counts[em] = (counts[em] || 0) + 1;
          if (counts[em] > maxCount) {
             maxCount = counts[em];
             fastEmotion = em;
          }
        }
        setCurrentEmotion(fastEmotion);

        // 2. Slow Stress tracking (map all 30 frames to stress and average them)
        const STRESS_MAPPING: Record<string, number> = {
            "Happy": 0, "Neutral": 10, "Surprise": 60, "Sad": 75,
            "Contempt": 80, "Disgust": 85, "Fear": 95, "Anger": 100, "N/A": 0
        };
        const totalStress = newHistory.reduce((acc, em) => acc + (STRESS_MAPPING[em] ?? 0), 0);
        const smoothedStress = Math.round(totalStress / newHistory.length);
        setCurrentStress(smoothedStress);

        // 3. Update charts and history with stable stress
        setStressHistory(prevStress => {
          const newStressHistory = [...prevStress, smoothedStress];
          if (newStressHistory.length > 100) newStressHistory.shift();
          return newStressHistory;
        });
        setFullSessionStress(prevFull => [...prevFull, smoothedStress]);

        // 4. Update adaptive brightness
        try {
          setAdaptiveBrightness(prevBrightness => {
            computeBrightness(
              smoothedStress, 
              settings.minBrightness, 
              settings.maxBrightness, 
              settings.smoothingFactor, 
              prevBrightness
            ).then(res => {
              if (res) setAdaptiveBrightness(res.new_brightness);
            });
            return prevBrightness;
          });
        } catch (e) {
          console.error(e);
        }

        // 5. Trigger notifications robustly based on stable stress
        // Use functional state update for lastNotificationTime to ensure latest value is used natively without stale closures
        setLastNotificationTime(prevTime => {
          const currentTime = Date.now() / 1000;
          if (smoothedStress >= settings.stressThreshold && (currentTime - prevTime > settings.notificationCooldown)) {
            const msg = `⚠️ High Stress Detected (${smoothedStress}%)! Consider taking a short break.`;
            setNotification(msg);
            setTimeout(() => setNotification(null), 5000);
            
            // Trigger native OS notification via backend
            import('@/lib/api').then(api => {
                api.triggerOSNotification("Neurolens: High Stress Alert", msg);
            });
            
            // Trigger native Browser notification (bulletproof cross-platform)
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification("Neurolens Alert", { body: msg, icon: "/favicon.ico" });
            }
            
            return currentTime;
          }
          return prevTime;
        });
      }
      return newHistory;
    });
  }, [settings]);

  const avgStress = fullSessionStress.length > 0 
    ? (fullSessionStress.reduce((a, b) => a + b, 0) / fullSessionStress.length).toFixed(1) 
    : 0;
    
  const peakStress = fullSessionStress.length > 0 
    ? Math.max(...fullSessionStress) 
    : 0;

  let stressName = "Calm";
  if (currentStress >= 40 && currentStress < 75) stressName = "Moderate";
  else if (currentStress >= 75) stressName = "High Stress";

  return (
    <>
      {notification && (
        <div className="glass-panel animate-slide-in" style={{ 
          position: 'fixed', top: '20px', left: '50%', transform: 'translateX(-50%)', 
          zIndex: 1000, background: 'rgba(239, 68, 68, 0.9)', color: 'white',
          display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1rem 2rem', border: '1px solid #f87171'
        }}>
          <AlertTriangle />
          <strong style={{ fontSize: '1.1rem' }}>{notification}</strong>
        </div>
      )}

      <div className="flex justify-between items-center" style={{ marginBottom: '1.5rem' }}>
        <div>
          <h2>Live Analysis Dashboard</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Real-time emotion and stress monitoring</p>
        </div>
        
        <button 
          onClick={() => setIsActive(!isActive)}
          className={`btn ${isActive ? 'btn-secondary' : 'btn-primary'}`}
          style={{ background: isActive ? 'rgba(239, 68, 68, 0.2)' : '', color: isActive ? '#f87171' : '', borderColor: isActive ? 'rgba(239, 68, 68, 0.5)' : '' }}
        >
          <Power size={18} /> {isActive ? 'Stop Monitoring' : 'Start Monitoring'}
        </button>
      </div>

      <div className="grid grid-cols-2">
        {/* Left Column - Video Feed & Chart */}
        <div className="flex-col gap-4">
          <WebcamCapture onAnalyticsUpdate={handleAnalyticsUpdate} isActive={isActive} />
          
          <div style={{ marginTop: '1.5rem' }}>
            <RealTimeChart data={stressHistory} />
          </div>
        </div>

        {/* Right Column - Status, Emotion, Gauges */}
        <div className="flex-col" style={{ gap: '1.5rem' }}>
          
          <div className="grid grid-cols-2" style={{ gap: '1.5rem' }}>
            <EmotionDisplay emotion={currentEmotion} confidence={currentConfidence} />
            
            <div className="glass-panel items-center justify-between" style={{ display: 'flex', borderLeft: '4px solid var(--accent-secondary)' }}>
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Adaptive Brightness</p>
                <h3 style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--accent-secondary)', margin: 0 }}>
                  {Math.round(adaptiveBrightness)}%
                </h3>
              </div>
            </div>
          </div>

          <StressGauge stressName={stressName} stressValue={currentStress} />

          <div className="glass-panel">
            <h3 className="glass-header">Session Analytics</h3>
            <div className="grid grid-cols-2">
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Average Stress</p>
                <p style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--text-primary)' }}>{avgStress}%</p>
              </div>
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Peak Stress</p>
                <p style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--color-danger)' }}>{peakStress}%</p>
              </div>
            </div>
          </div>
          
        </div>
      </div>
    </>
  );
}
