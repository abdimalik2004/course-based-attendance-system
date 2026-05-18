import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';

export function GeneralSettings() {
  const [hasChanges, setHasChanges] = useState(false);

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
            <Input 
              defaultValue="Heegan" 
              onChange={() => setHasChanges(true)} 
              className="bg-white/5 border-white/10"
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300 ml-1">Organization Name</label>
            <Input 
              defaultValue="Heegan Educational Institution" 
              onChange={() => setHasChanges(true)} 
              className="bg-white/5 border-white/10"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Timezone</label>
              <Select 
                options={[
                  { value: 'gmt4', label: '(GMT+04:00) Abu Dhabi' },
                  { value: 'gmt3', label: '(GMT+03:00) Riyadh' },
                  { value: 'gmt0', label: '(GMT+00:00) UTC' },
                ]}
                defaultValue="gmt4"
                onChange={() => setHasChanges(true)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Default Language</label>
              <Select 
                options={[
                  { value: 'en', label: 'English' },
                  { value: 'ar', label: 'Arabic' },
                ]}
                defaultValue="en"
                onChange={() => setHasChanges(true)}
              />
            </div>
          </div>
          
          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges} className="min-w-[140px]">Save Changes</Button>
            <Button variant="ghost" onClick={() => setHasChanges(false)}>Cancel</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
