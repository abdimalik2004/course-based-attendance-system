import { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { fetchSettings, saveSettings } from '@/services/settingsService';

const defaults = {
  systemName: 'Heegan',
  orgName: 'Heegan Educational Institution',
  timezone: 'Africa/Mogadishu',
  language: 'en',
};

export function GeneralSettings() {
  const [form, setForm] = useState(defaults);
  const [saved, setSaved] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSettings()
      .then((data) => {
        setForm({
          systemName: data['general.system_name'] ?? defaults.systemName,
          orgName: data['general.org_name'] ?? defaults.orgName,
          timezone: data['general.timezone'] ?? defaults.timezone,
          language: data['general.language'] ?? defaults.language,
        });
      })
      .catch(() => {});
  }, []);

  const update = (key: keyof typeof defaults, value: string) => {
    setForm(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
    setSaved(false);
    setError('');
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError('');
    try {
      await saveSettings({
        'general.system_name': form.systemName,
        'general.org_name': form.orgName,
        'general.timezone': form.timezone,
        'general.language': form.language,
      });
      setHasChanges(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch {
      setError('Failed to save settings. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    fetchSettings()
      .then((data) => {
        setForm({
          systemName: data['general.system_name'] ?? defaults.systemName,
          orgName: data['general.org_name'] ?? defaults.orgName,
          timezone: data['general.timezone'] ?? defaults.timezone,
          language: data['general.language'] ?? defaults.language,
        });
      })
      .catch(() => {});
    setHasChanges(false);
    setSaved(false);
    setError('');
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">General Settings</h2>
        <p className="text-sm text-gray-400">Manage basic system configuration and properties.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300 ml-1">System Name</label>
            <Input value={form.systemName} onChange={(e) => update('systemName', e.target.value)} className="bg-white/5 border-white/10" />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300 ml-1">Organization Name</label>
            <Input value={form.orgName} onChange={(e) => update('orgName', e.target.value)} className="bg-white/5 border-white/10" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">System Timezone</label>
              <Select
                options={[
                  { value: 'Africa/Mogadishu', label: '(GMT+03:00) Mogadishu — East Africa Time (EAT)' },
                  { value: 'Africa/Nairobi',   label: '(GMT+03:00) Nairobi — East Africa Time (EAT)' },
                  { value: 'Africa/Djibouti',  label: '(GMT+03:00) Djibouti — East Africa Time (EAT)' },
                  { value: 'Asia/Riyadh',      label: '(GMT+03:00) Riyadh, Kuwait — Arabia Standard Time' },
                  { value: 'Asia/Aden',        label: '(GMT+03:00) Aden — Arabia Standard Time' },
                  { value: 'Asia/Dubai',       label: '(GMT+04:00) Dubai, Abu Dhabi — Gulf Standard Time' },
                  { value: 'Asia/Muscat',      label: '(GMT+04:00) Muscat — Gulf Standard Time' },
                  { value: 'UTC',              label: '(GMT+00:00) UTC — Coordinated Universal Time' },
                ]}
                value={form.timezone}
                onChange={(e) => update('timezone', e.target.value)}
              />
              {form.timezone === 'Africa/Mogadishu' && (
                <p className="text-xs text-emerald-400 ml-1 mt-1 flex items-center gap-1">
                  <span>✓</span> Mogadishu time (EAT, UTC+3) — matches server clock
                </p>
              )}
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Default Language</label>
              <Select
                options={[
                  { value: 'en', label: 'English' },
                  { value: 'ar', label: 'Arabic' },
                ]}
                value={form.language}
                onChange={(e) => update('language', e.target.value)}
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
              <CheckCircle size={14} /> Settings saved successfully.
            </div>
          )}

          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges || isSaving} onClick={handleSave} isLoading={isSaving} className="min-w-[140px]">Save Changes</Button>
            <Button variant="ghost" onClick={handleCancel} disabled={isSaving}>Cancel</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
