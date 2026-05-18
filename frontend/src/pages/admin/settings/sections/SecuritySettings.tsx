import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Switch } from '@/components/ui/Switch';

export function SecuritySettings() {
  const [hasChanges, setHasChanges] = useState(false);
  const [is2FAEnabled, setIs2FAEnabled] = useState(true);

  const handleToggle = (val: boolean) => {
    setIs2FAEnabled(val);
    setHasChanges(true);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Security</h2>
        <p className="text-sm text-gray-400">Manage password constraints and 2FA authentication methods.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-8">
          
          <div className="space-y-4">
             <h3 className="text-white font-medium">Change Password</h3>
             <div className="grid grid-cols-1 gap-4 max-w-md">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400 ml-1">Current Password</label>
                  <Input type="password" placeholder="••••••••" onChange={() => setHasChanges(true)} className="bg-white/5 border-white/10" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400 ml-1">New Password</label>
                  <Input type="password" placeholder="••••••••" onChange={() => setHasChanges(true)} className="bg-white/5 border-white/10" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400 ml-1">Confirm New Password</label>
                  <Input type="password" placeholder="••••••••" onChange={() => setHasChanges(true)} className="bg-white/5 border-white/10" />
                </div>
             </div>
          </div>

          <div className="h-px bg-white/5 w-full" />

          <div className="space-y-6">
             <h3 className="text-white font-medium">Additional Security</h3>
             
             <div className="flex items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5">
                <div>
                   <h4 className="text-white font-medium text-sm">Two-Factor Authentication</h4>
                   <p className="text-gray-400 text-xs mt-1">Require an extra security step during login.</p>
                </div>
                <Switch checked={is2FAEnabled} onChange={handleToggle} />
             </div>

             <div className="space-y-2 max-w-md">
                <label className="text-sm font-medium text-gray-400 ml-1">Session Timeout</label>
                <Select 
                  options={[
                    { value: '15m', label: '15 Minutes' },
                    { value: '30m', label: '30 Minutes' },
                    { value: '1h', label: '1 Hour' },
                  ]}
                  defaultValue="30m"
                  onChange={() => setHasChanges(true)}
                />
             </div>
          </div>
          
          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges} className="min-w-[140px]">Update Security</Button>
            <Button variant="ghost" onClick={() => setHasChanges(false)}>Cancel</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
