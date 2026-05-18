import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Switch } from '@/components/ui/Switch';

export function NotificationSettings() {
  const [hasChanges, setHasChanges] = useState(false);
  
  const [toggles, setToggles] = useState({
    email: true,
    sms: false,
    system: true,
    attendance: true
  });

  const handleToggle = (key: keyof typeof toggles) => {
    setToggles(prev => ({ ...prev, [key]: !prev[key] }));
    setHasChanges(true);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Notifications</h2>
        <p className="text-sm text-gray-400">Control how and when you receive system alerts and updates.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-6">
          
          <div className="space-y-4">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5 gap-4">
               <div>
                 <h4 className="text-white font-medium text-sm">Email Notifications</h4>
                 <p className="text-gray-400 text-xs mt-1">Receive daily summaries and critical alerts via email.</p>
               </div>
               <Switch checked={toggles.email} onChange={() => handleToggle('email')} />
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5 gap-4">
               <div>
                 <h4 className="text-white font-medium text-sm">SMS Notifications</h4>
                 <p className="text-gray-400 text-xs mt-1">Get immediate SMS text messages for urgent system warnings.</p>
               </div>
               <Switch checked={toggles.sms} onChange={() => handleToggle('sms')} />
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5 gap-4">
               <div>
                 <h4 className="text-white font-medium text-sm">In-App System Alerts</h4>
                 <p className="text-gray-400 text-xs mt-1">Show toast notifications and badge counts inside the dashboard.</p>
               </div>
               <Switch checked={toggles.system} onChange={() => handleToggle('system')} />
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-white/10 bg-white/5 gap-4">
               <div>
                 <h4 className="text-white font-medium text-sm">Attendance Alerts</h4>
                 <p className="text-gray-400 text-xs mt-1">Notify me when unexpected absence anomalies occur.</p>
               </div>
               <Switch checked={toggles.attendance} onChange={() => handleToggle('attendance')} />
            </div>

          </div>

          <div className="space-y-2 max-w-sm pt-4 border-t border-white/5">
            <label className="text-sm font-medium text-gray-400 ml-1">Notification Frequency</label>
            <Select 
              options={[
                { value: 'realtime', label: 'Real-time (Immediate)' },
                { value: 'daily', label: 'Daily Digest' },
                { value: 'weekly', label: 'Weekly Summary' },
              ]}
              defaultValue="realtime"
              onChange={() => setHasChanges(true)}
            />
          </div>
          
          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges} className="min-w-[140px]">Save Preferences</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
