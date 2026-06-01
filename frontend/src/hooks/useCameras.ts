import { useEffect, useState, useCallback } from 'react';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export function useCameras() {
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Ensure we have permission to enumerate labels
      try {
        await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (e) {
        // Permission may be denied; we'll still try to enumerate devices
      }

      const list = await navigator.mediaDevices.enumerateDevices();
      const cams = list
        .filter((d) => d.kind === 'videoinput')
        .map((d) => ({ deviceId: d.deviceId, label: d.label || 'Camera' }));
      setDevices(cams);
    } catch (err: any) {
      setError(String(err?.message ?? err || 'Failed to enumerate devices'));
      setDevices([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const handler = () => refresh();
    navigator.mediaDevices?.addEventListener?.('devicechange', handler);
    return () => navigator.mediaDevices?.removeEventListener?.('devicechange', handler);
  }, [refresh]);

  return { devices, loading, error, refresh } as const;
}

export default useCameras;
