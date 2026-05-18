import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';

export function SystemPreferences() {
  const [hasChanges, setHasChanges] = useState(false);

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">System Preferences</h2>
        <p className="text-sm text-gray-400">Configure global layout logic, local themes, and formatting.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-6">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Theme Mode</label>
              <Select 
                options={[
                  { value: 'dark', label: 'Dark Mode (Neon)' },
                  { value: 'light', label: 'Light Mode (Clean)' },
                  { value: 'system', label: 'System Default' },
                ]}
                defaultValue="dark"
                onChange={() => setHasChanges(true)}
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Default Dashboard View</label>
              <Select 
                options={[
                  { value: 'admin', label: 'Admin Metrics' },
                  { value: 'attendance', label: 'Live Attendance View' },
                  { value: 'users', label: 'Users Management Table' }
                ]}
                defaultValue="admin"
                onChange={() => setHasChanges(true)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Date Format</label>
              <Select 
                options={[
                  { value: 'ddmmyyyy', label: 'DD/MM/YYYY' },
                  { value: 'mmddyyyy', label: 'MM/DD/YYYY' },
                  { value: 'yyyyddmm', label: 'YYYY/MM/DD' },
                ]}
                defaultValue="ddmmyyyy"
                onChange={() => setHasChanges(true)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Time Format</label>
              <Select 
                options={[
                  { value: '12h', label: '12-hour (AM/PM)' },
                  { value: '24h', label: '24-hour' },
                ]}
                defaultValue="12h"
                onChange={() => setHasChanges(true)}
              />
            </div>
          </div>
          
          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges} className="min-w-[140px]">Save Changes</Button>
            <Button variant="ghost" onClick={() => setHasChanges(false)}>Reset</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
