import { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Switch } from '@/components/ui/Switch';
import { api } from '@/services/api';
import { fetchSettings, saveSettings } from '@/services/settingsService';

export function SecuritySettings() {
  const [sessionTimeout, setSessionTimeout] = useState('30m');
  const [securitySaved, setSecuritySaved] = useState(false);
  const [securityError, setSecurityError] = useState('');
  const [isSavingSecurity, setIsSavingSecurity] = useState(false);
  const [hasSecurityChanges, setHasSecurityChanges] = useState(false);

  // Password change state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState('');
  const [passwordError, setPasswordError] = useState('');

  // Load session timeout from API
  useEffect(() => {
    fetchSettings()
      .then((data) => {
        if (data['security.session_timeout']) {
          setSessionTimeout(data['security.session_timeout']);
        }
      })
      .catch(() => {});
  }, []);

  const handleChangePassword = async () => {
    setPasswordError('');
    setPasswordSuccess('');

    if (!currentPassword || !newPassword || !confirmPassword) {
      setPasswordError('All password fields are required.');
      return;
    }
    if (newPassword.length < 6) {
      setPasswordError('New password must be at least 6 characters.');
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('New password and confirmation do not match.');
      return;
    }

    setIsSubmitting(true);
    try {
      await api.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
      setPasswordSuccess('Password updated successfully.');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setPasswordError(typeof detail === 'string' ? detail : 'Failed to update password. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSaveSecurity = async () => {
    setIsSavingSecurity(true);
    setSecurityError('');
    try {
      await saveSettings({ 'security.session_timeout': sessionTimeout });
      setHasSecurityChanges(false);
      setSecuritySaved(true);
      setTimeout(() => setSecuritySaved(false), 3000);
    } catch {
      setSecurityError('Failed to save security settings.');
    } finally {
      setIsSavingSecurity(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Security</h2>
        <p className="text-sm text-gray-400">Manage password constraints and authentication settings.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-8">

          {/* Change Password */}
          <div className="space-y-4">
            <h3 className="text-white font-medium">Change Password</h3>
            <div className="grid grid-cols-1 gap-4 max-w-md">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-400 ml-1">Current Password</label>
                <Input
                  type="password"
                  placeholder="Enter your current password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="bg-white/5 border-white/10"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-400 ml-1">New Password</label>
                <Input
                  type="password"
                  placeholder="Minimum 6 characters"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="bg-white/5 border-white/10"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-400 ml-1">Confirm New Password</label>
                <Input
                  type="password"
                  placeholder="Re-enter your new password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="bg-white/5 border-white/10"
                />
                {confirmPassword.length > 0 && (
                  confirmPassword !== newPassword ? (
                    <p className="text-xs text-red-400 ml-1 flex items-center gap-1">
                      <AlertCircle size={11} />
                      Passwords do not match.
                    </p>
                  ) : (
                    <p className="text-xs text-emerald-400 ml-1 flex items-center gap-1">
                      <CheckCircle size={11} />
                      Passwords matched.
                    </p>
                  )
                )}
              </div>
            </div>

            {passwordError && (
              <div className="flex items-center gap-2 text-sm text-red-400 max-w-md">
                <AlertCircle size={14} className="shrink-0" />
                <span>{passwordError}</span>
              </div>
            )}
            {passwordSuccess && (
              <div className="flex items-center gap-2 text-sm text-emerald-400 max-w-md">
                <CheckCircle size={14} className="shrink-0" />
                <span>{passwordSuccess}</span>
              </div>
            )}

            <div className="flex items-center gap-3 pt-2">
              <Button
                disabled={isSubmitting}
                onClick={handleChangePassword}
                isLoading={isSubmitting}
                className="min-w-[160px]"
              >
                Update Password
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  setCurrentPassword(''); setNewPassword('');
                  setConfirmPassword(''); setPasswordError(''); setPasswordSuccess('');
                }}
              >
                Cancel
              </Button>
            </div>
          </div>

          <div className="h-px bg-white/5 w-full" />

          {/* Session & 2FA */}
          <div className="space-y-6">
            <h3 className="text-white font-medium">Additional Security</h3>

            <div className="flex items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5 opacity-70">
              <div>
                <h4 className="text-white font-medium text-sm flex items-center gap-2">
                  Two-Factor Authentication
                  <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/20 font-normal">
                    Coming Soon
                  </span>
                </h4>
                <p className="text-gray-400 text-xs mt-1">Require an extra security step during login.</p>
              </div>
              <Switch checked={false} onChange={() => {}} disabled={true} />
            </div>

            <div className="space-y-2 max-w-md">
              <label className="text-sm font-medium text-gray-400 ml-1">Session Timeout</label>
              <Select
                options={[
                  { value: '15m', label: '15 Minutes' },
                  { value: '30m', label: '30 Minutes' },
                  { value: '1h', label: '1 Hour' },
                ]}
                value={sessionTimeout}
                onChange={(e) => { setSessionTimeout(e.target.value); setHasSecurityChanges(true); setSecuritySaved(false); }}
              />
            </div>

            {securityError && (
              <div className="flex items-center gap-2 text-sm text-red-400">
                <AlertCircle size={14} /> {securityError}
              </div>
            )}
            {securitySaved && (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <CheckCircle size={14} /> Security settings saved.
              </div>
            )}

            <div className="pt-2">
              <Button
                disabled={!hasSecurityChanges || isSavingSecurity}
                onClick={handleSaveSecurity}
                isLoading={isSavingSecurity}
                className="min-w-[160px]"
              >
                Save Security Settings
              </Button>
            </div>
          </div>

        </CardContent>
      </Card>
    </div>
  );
}
