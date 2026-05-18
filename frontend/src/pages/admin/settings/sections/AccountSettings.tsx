import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import placeholderFace from '@/assets/logo.png'; 

export function AccountSettings() {
  const [hasChanges, setHasChanges] = useState(false);

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Account Settings</h2>
        <p className="text-sm text-gray-400">Update your profile information and administrative details.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-8">
          
          <div className="flex flex-col sm:flex-row items-center gap-6">
            <div className="relative">
              <img src={placeholderFace} alt="Admin Profile" className="w-24 h-24 rounded-full object-cover border-4 border-white/10" />
              <button className="absolute bottom-0 right-0 bg-primary w-8 h-8 rounded-full flex items-center justify-center text-white border-2 border-[#111827] hover:bg-primary-hover transition-colors">
                 <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/></svg>
              </button>
            </div>
            <div>
              <h3 className="text-white font-medium text-lg">Profile Picture</h3>
              <p className="text-gray-400 text-sm mb-3">PNG, JPG up to 5MB.</p>
              <Button variant="secondary" size="sm">Remove Picture</Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Admin Name</label>
              <Input 
                defaultValue="System Administrator" 
                onChange={() => setHasChanges(true)} 
                className="bg-white/5 border-white/10"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
              <Input 
                type="email"
                defaultValue="admin@heegan.edu" 
                onChange={() => setHasChanges(true)} 
                className="bg-white/5 border-white/10"
              />
            </div>
            <div className="space-y-2 lg:col-span-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Phone Number (Optional)</label>
              <Input 
                type="tel"
                placeholder="+971 XX XXX XXXX"
                onChange={() => setHasChanges(true)} 
                className="bg-white/5 border-white/10"
              />
            </div>
          </div>
          
          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges} className="min-w-[140px]">Update Profile</Button>
            <Button variant="ghost" onClick={() => setHasChanges(false)}>Cancel</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
