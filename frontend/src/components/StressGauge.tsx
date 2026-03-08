import { Settings } from 'lucide-react';

interface StressGaugeProps {
  stressName: string; // e.g., "Calm", "Moderate", "High Stress"
  stressValue: number; // 0-100
}

export default function StressGauge({ stressName, stressValue }: StressGaugeProps) {
  let color = 'var(--color-success)';
  if (stressValue >= 40 && stressValue < 75) color = 'var(--color-warning)';
  else if (stressValue >= 75) color = 'var(--color-danger)';

  return (
    <div className="glass-panel" style={{ flex: 1 }}>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>Overall Stress Level</p>
      
      <div className="flex items-center justify-between" style={{ marginBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '2.5rem', margin: 0, color }}>{Math.round(stressValue)}%</h2>
          <span className="badge" style={{ background: `${color}1A`, color, marginTop: '0.5rem' }}>
            {stressName}
          </span>
        </div>
      </div>

      <div className="progress-container">
        <div 
          className="progress-bar" 
          style={{ width: `${stressValue}%`, background: color }}
        />
      </div>
      <div className="flex justify-between" style={{ marginTop: '0.5rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
        <span>0</span>
        <span>50</span>
        <span>100</span>
      </div>
    </div>
  );
}
