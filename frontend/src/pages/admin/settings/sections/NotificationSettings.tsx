import { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Switch } from '@/components/ui/Switch';
import { fetchSettings, saveSettings } from '@/services/settingsService';

const defaults = { email: true, sms: false, system: true, attendance: true, frequency: 'realtime' };

export function NotificationSettings() {
  const [prefs, setPrefs] = useState(defaults);
  const [hasChanges, setHasChanges] = useState(false);
  const [saved, setSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSettings()
      .then((data) => {
        setPrefs({
          email: data['notifications.email'] === 'true',
          sms: data['notifications.sms'] === 'true',
          system: data['notifications.system'] === 'true',
          attendance: data['notifications.attendance'] === 'true',
          frequency: data['notifications.frequency'] ?? defaults.frequency,
        });
      })
      .catch(() => {});
  }, []);

  const toggle = (key: 'email' | 'sms' | 'system' | 'attendance') => {
    setPrefs(prev => ({ ...prev, [key]: !prev[key] }));
    setHasChanges(true);
    setSaved(false);
    setError('');
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError('');
    try {
      await saveSettings({
        'notifications.email': prefs.email,
        'notifications.sms': prefs.sms,
        'notifications.system': prefs.system,
        'notifications.attendance': prefs.attendance,
        'notifications.frequency': prefs.frequency,
      });
      setHasChanges(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch {
      setError('Failed to save preferences. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const rows: { key: 'email' | 'sms' | 'system' | 'attendance'; title: string; desc: string }[] = [
    { key: 'email', title: 'Email Notifications', desc: 'Receive daily summaries and critical alerts via email.' },
    { key: 'sms', title: 'SMS Notifications', desc: 'Get immediate SMS text messages for urgent system warnings.' },
    { key: 'system', title: 'In-App System Alerts', desc: 'Show toast notifications and badge counts inside the dashboard.' },
    { key: 'attendance', title: 'Attendance Alerts', desc: 'Notify me when unexpected absence anomalies occur.' },
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Notifications</h2>
        <p className="text-sm text-gray-400">Control how and when you receive system alerts and updates.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-6">
          <div className="space-y-4">
            {rows.map(({ key, title, desc }) => (
              <div key={key} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5 gap-4">
                <div>
                  <h4 className="text-white font-medium text-sm">{title}</h4>
                  <p className="text-gray-400 text-xs mt-1">{desc}</p>
                </div>
                <Switch checked={prefs[key]} onChange={() => toggle(key)} />
              </div>
            ))}
          </div>

          <div className="space-y-2 max-w-sm pt-4 border-t border-white/5">
            <label className="text-sm font-medium text-gray-400 ml-1">Notification Frequency</label>
            <Select
              options={[
                { value: 'realtime', label: 'Real-time (Immediate)' },
                { value: 'daily', label: 'Daily Digest' },
                { value: 'weekly', label: 'Weekly Summary' },
              ]}
              value={prefs.frequency}
              onChange={(e) => { setPrefs(p => ({ ...p, frequency: e.target.value })); setHasChanges(true); setSaved(false); }}
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 text-sm text-red-400">
              <AlertCircle size={14} /> {error}
            </div>
          )}
          {saved && (
            <div className="flex items-center gap-2 text-sm text-emerald-400">
              <CheckCircle size={14} /> Preferences saved.
            </div>
          )}

          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges || isSaving} onClick={handleSave} isLoading={isSaving} className="min-w-[140px]">Save Preferences</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
