import { 
  Settings, 
  UserCircle, 
  Shield, 
  Bell, 
  SlidersHorizontal, 
  Lock
} from 'lucide-react';
import { cn } from '@/utils/cn';

export type SettingsTab = 'general' | 'account' | 'security' | 'notifications' | 'preferences' | 'access';

interface SettingsSidebarProps {
  activeTab: SettingsTab;
  onTabChange: (tab: SettingsTab) => void;
}

const tabs: { id: SettingsTab; label: string; icon: React.ElementType }[] = [
  { id: 'general', label: 'General Settings', icon: Settings },
  { id: 'account', label: 'Account Settings', icon: UserCircle },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'preferences', label: 'System Preferences', icon: SlidersHorizontal },
  { id: 'access', label: 'Access Control', icon: Lock },
];

export function SettingsSidebar({ activeTab, onTabChange }: SettingsSidebarProps) {
  return (
    <div className="w-full lg:w-64 shrink-0 flex flex-col gap-1">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-sm font-medium",
              isActive 
                ? "bg-primary/10 text-primary border border-primary/20 shadow-[inset_0_0_10px_rgba(37,99,235,0.1)]" 
                : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-white/5 border border-transparent"
            )}
          >
            <Icon size={18} className={isActive ? "text-primary drop-shadow-[0_0_8px_rgba(37,99,235,0.6)]" : "text-gray-500"} />
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
