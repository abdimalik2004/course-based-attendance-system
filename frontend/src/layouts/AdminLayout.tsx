import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import { useSidebarStore } from '@/store/useSidebarStore';
import { cn } from '@/utils/cn';

export default function AdminLayout() {
  const { isCollapsed } = useSidebarStore();
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg text-gray-900 dark:text-gray-100 transition-colors duration-300">
      <Sidebar />
      
      <div 
        className={cn(
          "flex flex-col min-h-screen transition-all duration-300 ease-out",
          // On mobile, content is always full-width. On desktop, adjust margin based on sidebar state.
          isMobile ? "ml-0" : (isCollapsed ? "ml-[80px]" : "ml-[280px]")
        )}
      >
        <Topbar />
        
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-x-hidden">
          {/* Framer motion page transitions could be added here if needed */}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
