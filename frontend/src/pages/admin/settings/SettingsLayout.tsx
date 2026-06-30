import { useState } from 'react';
import { SettingsSidebar, type SettingsTab } from './components/SettingsSidebar';
import { GeneralSettings } from './sections/GeneralSettings';
import { AccountSettings } from './sections/AccountSettings';
import { SecuritySettings } from './sections/SecuritySettings';
import { NotificationSettings } from './sections/NotificationSettings';
import { SystemPreferences } from './sections/SystemPreferences';
import { AccessControlSettings } from './sections/AccessControlSettings';
import { AISettings } from './sections/AISettings';

export default function SettingsLayout() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');

  const renderContent = () => {
    switch (activeTab) {
      case 'general':
        return <GeneralSettings />;
      case 'account':
        return <AccountSettings />;
      case 'security':
        return <SecuritySettings />;
      case 'notifications':
        return <NotificationSettings />;
      case 'preferences':
        return <SystemPreferences />;
      case 'access':
        return <AccessControlSettings />;
      case 'ai':
        return <AISettings />;
      default:
        return <GeneralSettings />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1 border-b border-gray-200 dark:border-white/10 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
          System Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Manage system configuration and preferences
        </p>
      </div>

      {/* Split Layout */}
      <div className="flex flex-col lg:flex-row gap-8">
        
        {/* Left Side: Navigation */}
        <SettingsSidebar activeTab={activeTab} onTabChange={setActiveTab} />
        
        {/* Right Side: Rendered Section Content */}
        <div className="flex-1 max-w-4xl">
           {renderContent()}
        </div>
      </div>
    </div>
  );
}
