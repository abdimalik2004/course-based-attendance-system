import { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { useThemeStore } from '@/store/useThemeStore';
import { fetchSettings, saveSettings } from '@/services/settingsService';

const defaults = { defaultView: 'admin', dateFormat: 'ddmmyyyy', timeFormat: '12h' };

export function SystemPreferences() {
  const { theme, setTheme } = useThemeStore();
  const [prefs, setPrefs] = useState(defaults);
  const [hasChanges, setHasChanges] = useState(false);
  const [saved, setSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');

  // Load preferences from API
  useEffect(() => {
    fetchSettings()
      .then((data) => {
        setPrefs({
          defaultView: data['preferences.default_view'] ?? defaults.defaultView,
          dateFormat: data['preferences.date_format'] ?? defaults.dateFormat,
          timeFormat: data['preferences.time_format'] ?? defaults.timeFormat,
        });
      })
      .catch(() => {});
  }, []);

  const update = (key: keyof typeof defaults, value: string) => {
    setPrefs(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
    setSaved(false);
    setError('');
  };

  const handleThemeChange = (value: string) => {
    if (value === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setTheme(prefersDark ? 'dark' : 'light');
    } else {
      setTheme(value as 'light' | 'dark');
    }
    setHasChanges(true);
    setSaved(false);
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError('');
    try {
      await saveSettings({
        'preferences.default_view': prefs.defaultView,
        'preferences.date_format': prefs.dateFormat,
        'preferences.time_format': prefs.timeFormat,
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

  const handleReset = () => {
    fetchSettings()
      .then((data) => {
        setPrefs({
          defaultView: data['preferences.default_view'] ?? defaults.defaultView,
          dateFormat: data['preferences.date_format'] ?? defaults.dateFormat,
          timeFormat: data['preferences.time_format'] ?? defaults.timeFormat,
        });
      })
      .catch(() => setPrefs(defaults));
    setHasChanges(false);
    setSaved(false);
    setError('');
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">System Preferences</h2>
        <p className="text-sm text-gray-400">Configure layout, theme, and display formatting.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Theme — writes to Zustand (persisted to localStorage) */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Theme Mode</label>
              <Select
                options={[
                  { value: 'dark', label: 'Dark Mode' },
                  { value: 'light', label: 'Light Mode' },
                  { value: 'system', label: 'System Default' },
                ]}
                value={theme}
                onChange={(e) => handleThemeChange(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Default Dashboard View</label>
              <Select
                options={[
                  { value: 'admin', label: 'Admin Metrics' },
                  { value: 'attendance', label: 'Live Attendance View' },
                  { value: 'users', label: 'Users Management Table' },
                ]}
                value={prefs.defaultView}
                onChange={(e) => update('defaultView', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Date Format</label>
              <Select
                options={[
                  { value: 'ddmmyyyy', label: 'DD/MM/YYYY' },
                  { value: 'mmddyyyy', label: 'MM/DD/YYYY' },
                  { value: 'yyyymmdd', label: 'YYYY/MM/DD' },
                ]}
                value={prefs.dateFormat}
                onChange={(e) => update('dateFormat', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Time Format</label>
              <Select
                options={[
                  { value: '12h', label: '12-hour (AM/PM)' },
                  { value: '24h', label: '24-hour' },
                ]}
                value={prefs.timeFormat}
                onChange={(e) => update('timeFormat', e.target.value)}
              />
            </div>
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
            <Button disabled={!hasChanges || isSaving} onClick={handleSave} isLoading={isSaving} className="min-w-[140px]">Save Changes</Button>
            <Button variant="ghost" onClick={handleReset} disabled={isSaving}>Reset</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
