"use client";

import { useEffect, useRef, useState, useCallback } from 'react';
import { predictFrame } from '@/lib/api';
import { Camera, CameraOff, Loader2 } from 'lucide-react';

interface WebcamCaptureProps {
  onAnalyticsUpdate: (data: { emotion: string; stress: number; confidence: number }) => void;
  isActive: boolean;
}

export default function WebcamCapture({ onAnalyticsUpdate, isActive }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Initialize camera
  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setError(null);
    } catch (err) {
      console.error('Failed to start camera:', err);
      setError('Please allow camera access to use this feature.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  }, [stream]);

  // Manage camera lifecycle
  useEffect(() => {
    if (isActive && !stream) {
      startCamera();
    } else if (!isActive && stream) {
      stopCamera();
    }
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  // Processing loop
  useEffect(() => {
    if (!isActive || !stream) return;

    let processingInterval: NodeJS.Timeout;

    const processFrame = async () => {
      if (!videoRef.current || !canvasRef.current || isProcessing) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // JPEG gives better performance vs PNG
        const base64Image = canvas.toDataURL('image/jpeg', 0.8);

        setIsProcessing(true);
        try {
          const res = await predictFrame(base64Image, 0.25);
          if (res) {
            onAnalyticsUpdate({
              emotion: res.emotion,
              stress: res.stress_level,
              confidence: res.confidence,
            });
          }
        } catch (err) {
          console.error("Frame processing failed", err);
        } finally {
          setIsProcessing(false);
        }
      }
    };

    processingInterval = setInterval(processFrame, 166); // ~6 fps
    return () => clearInterval(processingInterval);
  }, [isActive, stream, isProcessing, onAnalyticsUpdate]);

  return (
    <div className="glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
      <div className="glass-header" style={{ margin: 0, padding: '1rem', borderBottom: '1px solid var(--panel-border)' }}>
        <h2 style={{ margin: 0, fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Camera size={20} className="text-gradient" /> Webcam Feed
        </h2>
        {isActive ? (
          <span className="badge" style={{ background: 'rgba(34, 197, 94, 0.1)', color: 'var(--color-success)' }}>
            <span className="animate-pulse" style={{ width: 8, height: 8, borderRadius: '50%', background: 'currentColor', display: 'inline-block' }}></span>
            Active
          </span>
        ) : (
          <span className="badge" style={{ background: 'rgba(239, 68, 68, 0.1)', color: 'var(--color-danger)' }}>
            <CameraOff size={14} /> Paused
          </span>
        )}
      </div>

      <div style={{ position: 'relative', width: '100%', aspectRatio: '4/3', background: '#000', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'cover',
            display: (isActive && !error) ? 'block' : 'none'
          }} 
        />

        {error && (
          <div style={{ color: 'var(--color-danger)', textAlign: 'center', position: 'absolute' }}>
            <CameraOff size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <p>{error}</p>
          </div>
        )}
        
        {!isActive && !error && (
          <div style={{ color: 'var(--text-muted)', textAlign: 'center', position: 'absolute' }}>
            <CameraOff size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <p>Camera is turned off.</p>
          </div>
        )}
      </div>
    </div>
  );
}
