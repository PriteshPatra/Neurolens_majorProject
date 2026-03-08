import { Activity, Frown, Meh, Smile, Zap } from 'lucide-react';

interface EmotionDisplayProps {
  emotion: string;
  confidence: number;
}

export default function EmotionDisplay({ emotion, confidence }: EmotionDisplayProps) {
  // Determine color based on CSS variables defined in globals.css
  const colorVar = `var(--emotion-${emotion.toLowerCase()})`;
  const fallbackColor = 'var(--emotion-none)';

  let Icon = Activity;
  if (['Happy'].includes(emotion)) Icon = Smile;
  else if (['Neutral'].includes(emotion)) Icon = Meh;
  else if (['Sad', 'Fear'].includes(emotion)) Icon = Frown;
  else if (['Anger', 'Disgust', 'Contempt'].includes(emotion)) Icon = Zap;

  return (
    <div className="glass-panel items-center justify-between" style={{ display: 'flex', borderLeft: `4px solid ${colorVar || fallbackColor}` }}>
      <div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Detected Emotion</p>
        <h3 style={{ fontSize: '1.5rem', fontWeight: 600, color: colorVar || fallbackColor, margin: 0, textTransform: 'capitalize' }}>
          {emotion}
        </h3>
      </div>
      <div style={{ textAlign: 'right' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Confidence</p>
        <div className="badge" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)' }}>
          {(confidence * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
}
