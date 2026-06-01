import { useState, useEffect } from 'react';
import { Save, Activity, Settings2, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { fetchSettings, saveSettings } from '@/services/settingsService';

export default function Settings() {
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [hasChanges, setHasChanges] = useState(false);

  const [requireApproval, setRequireApproval] = useState(true);
  const [notifyOnLeave, setNotifyOnLeave] = useState(true);

  // Load from API on mount
  useEffect(() => {
    fetchSettings()
      .then((data) => {
        if (data['hr.require_approval'] !== undefined) {
          setRequireApproval(data['hr.require_approval'] === 'true');
        }
        if (data['hr.notify_on_leave'] !== undefined) {
          setNotifyOnLeave(data['hr.notify_on_leave'] === 'true');
        }
      })
      .catch(() => {});
  }, []);

  const handleToggle = (setter: (v: boolean) => void, value: boolean) => {
    setter(value);
    setHasChanges(true);
    setSaved(false);
    setError('');
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError('');
    try {
      await saveSettings({
        'hr.require_approval': requireApproval,
        'hr.notify_on_leave': notifyOnLeave,
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

  const handleReset = () => {
    fetchSettings()
      .then((data) => {
        setRequireApproval(data['hr.require_approval'] !== undefined ? data['hr.require_approval'] === 'true' : true);
        setNotifyOnLeave(data['hr.notify_on_leave'] !== undefined ? data['hr.notify_on_leave'] === 'true' : true);
      })
      .catch(() => {
        setRequireApproval(true);
        setNotifyOnLeave(true);
      });
    setHasChanges(false);
    setSaved(false);
    setError('');
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">HR Settings</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Configure default HR behaviors and status definitions.</p>
      </div>

      <div className="grid gap-6">
        {/* General Settings */}
        <div className="glass-card rounded-2xl border border-gray-200 dark:border-white/10 overflow-hidden">
          <div className="p-4 border-b border-gray-200 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 flex items-center gap-2">
            <Settings2 size={18} className="text-gray-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">General Configurations</h2>
          </div>
          <div className="p-6 space-y-6">

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">Require Approval for Leave</h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Require admin approval before marking a teacher as "On Leave".</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={requireApproval}
                  onChange={(e) => handleToggle(setRequireApproval, e.target.checked)}
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 dark:peer-focus:ring-primary/10 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-primary"></div>
              </label>
            </div>

            <hr className="border-gray-200 dark:border-white/5" />

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">Leave Notifications</h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Send an email notification when a teacher's status changes to "On Leave".</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={notifyOnLeave}
                  onChange={(e) => handleToggle(setNotifyOnLeave, e.target.checked)}
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 dark:peer-focus:ring-primary/10 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-primary"></div>
              </label>
            </div>

          </div>
        </div>

        {/* Status Definitions (read-only) */}
        <div className="glass-card rounded-2xl border border-gray-200 dark:border-white/10 overflow-hidden">
          <div className="p-4 border-b border-gray-200 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 flex items-center gap-2">
            <Activity size={18} className="text-gray-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Status Definitions</h2>
          </div>
          <div className="p-6 space-y-4">
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">Current definitions for teacher statuses across the system. These cannot be changed directly here.</p>

            <div className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-500/5 rounded-xl border border-green-100 dark:border-green-500/10">
              <span className="w-2.5 h-2.5 rounded-full bg-green-500 mt-1.5 shrink-0"></span>
              <div>
                <h4 className="text-sm font-semibold text-gray-900 dark:text-green-100">Active</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Teacher is currently active, assigned to courses, and teaching.</p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-500/5 rounded-xl border border-amber-100 dark:border-amber-500/10">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 mt-1.5 shrink-0"></span>
              <div>
                <h4 className="text-sm font-semibold text-gray-900 dark:text-amber-100">On Leave</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Teacher is temporarily inactive (e.g., sabbatical, sick leave). No new course assignments allowed.</p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-red-50 dark:bg-red-500/5 rounded-xl border border-red-100 dark:border-red-500/10">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 mt-1.5 shrink-0"></span>
              <div>
                <h4 className="text-sm font-semibold text-gray-900 dark:text-red-100">Inactive</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Teacher has left the institution or is permanently inactive. Read-only access for past records.</p>
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-sm text-red-500">
            <AlertCircle size={14} /> {error}
          </div>
        )}
        {saved && (
          <div className="flex items-center gap-2 text-sm text-emerald-500">
            <CheckCircle size={14} /> Settings saved successfully.
          </div>
        )}

        <div className="flex justify-end gap-3 pb-6">
          <Button variant="secondary" type="button" onClick={handleReset} disabled={isSaving}>
            Reset to Defaults
          </Button>
          <Button onClick={handleSave} isLoading={isSaving} disabled={!hasChanges || isSaving} className="min-w-[120px]">
            <Save className="mr-2 h-4 w-4" />
            Save Changes
          </Button>
        </div>

      </div>
    </div>
  );
}
