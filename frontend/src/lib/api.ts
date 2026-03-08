// Create lib/api.ts for wrapping backend interaction

// Since Next.js and FastAPI run on different ports but might be accessed via network, it's safer
// to point directly to the known FastAPI host or configure Next.js rewrites. For simplicity here:
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export interface PredictResponse {
  emotion: string;
  stress_level: number;
  confidence: number;
  emotion_color: string;
  annotated_image: string;
}

export interface BrightnessResponse {
  target_brightness: number;
  new_brightness: number;
}

export async function predictFrame(imageBase64: string, confidence: number = 0.6): Promise<PredictResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64, confidence }),
    });
    if (!res.ok) throw new Error('Prediction request failed');
    return await res.json();
  } catch (err) {
    console.error('Error predicting frame:', err);
    return null;
  }
}

export async function computeBrightness(
  stressLevel: number,
  minBrightness: number,
  maxBrightness: number,
  smoothingFactor: number,
  currentBrightness: number
): Promise<BrightnessResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/brightness`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        stress_level: stressLevel,
        min_brightness: minBrightness,
        max_brightness: maxBrightness,
        smoothing_factor: smoothingFactor,
        current_brightness: currentBrightness,
      }),
    });
    if (!res.ok) throw new Error('Brightness request failed');
    return await res.json();
  } catch (err) {
    console.error('Error computing brightness:', err);
    return null;
  }
}

export async function triggerOSNotification(title: string, message: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/notify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, message }),
    });
    return res.ok;
  } catch (err) {
    console.error('Error triggering OS notification:', err);
    return false;
  }
}
