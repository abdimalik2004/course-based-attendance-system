import { useState, useEffect, useRef } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import { useSidebarStore } from '@/store/useSidebarStore';
import { cn } from '@/utils/cn';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { LogOut, User, UserCog, Bell, Menu, FileText } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { useUIStore } from '@/store/useUIStore';
import { motion, AnimatePresence } from 'framer-motion';
import logoUrl from '@/assets/logo.png';
import lightLogoUrl from '@/assets/light-logo.png';

function HRTopbar() {
  const { logout } = useAuthStore();
  const navigate = useNavigate();
  const { toggleSidebar } = useSidebarStore();
  const { openEditProfile } = useUIStore();
  
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);
  const [notificationDropdownOpen, setNotificationDropdownOpen] = useState(false);
  
  const profileRef = useRef<HTMLDivElement>(null);
  const notificationRef = useRef<HTMLDivElement>(null);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setProfileDropdownOpen(false);
      }
      if (notificationRef.current && !notificationRef.current.contains(event.target as Node)) {
        setNotificationDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };
  
  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-gray-200 dark:border-white/10 bg-white/80 dark:bg-dark-bg/80 px-4 backdrop-blur-xl sm:px-6">
      <div className="flex items-center gap-4">
        <button
          onClick={toggleSidebar}
          className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-white/10 lg:hidden"
        >
          <Menu size={20} />
        </button>
        {/* System Logo in Topbar as requested */}
        <div className="hidden sm:flex items-center gap-2 lg:hidden">
            <img src={lightLogoUrl} alt="Logo Light" className="dark:hidden h-8 w-8 rounded-md object-cover" />
            <img src={logoUrl} alt="Logo Dark" className="hidden dark:block h-8 w-8 rounded-md object-cover" />
            <span className="font-semibold text-gray-900 dark:text-white">Heegan HR</span>
        </div>
        <h1 className="text-xl font-semibold text-gray-800 dark:text-white sm:hidden lg:block">
          HR Dashboard
        </h1>
      </div>

      <div className="flex items-center gap-3 sm:gap-4">
        <ThemeToggle />
        
        {/* Notifications Dropdown */}
        <div className="relative" ref={notificationRef}>
          <button 
            onClick={() => {
                setNotificationDropdownOpen(!notificationDropdownOpen);
                setProfileDropdownOpen(false);
            }}
            className="relative rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-white/10 focus:outline-none"
          >
            <Bell size={20} />
            <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-primary ring-2 ring-white dark:ring-dark-bg" />
          </button>

          <AnimatePresence>
            {notificationDropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.15, ease: 'easeOut' }}
                className="absolute right-0 top-12 w-80 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-4 shadow-xl glass-card overflow-hidden z-50"
              >
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Notifications</h3>
                  <button className="text-xs text-primary hover:underline">Mark all as read</button>
                </div>
                <div className="space-y-3 max-h-[300px] overflow-y-auto custom-scrollbar">
                  {/* Mock Notification 1 */}
                  <div className="flex gap-3 items-start p-2 hover:bg-gray-50 dark:hover:bg-white/5 rounded-lg transition-colors cursor-pointer">
                    <div className="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-500/20 flex items-center justify-center shrink-0">
                      <User size={14} className="text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-800 dark:text-gray-200"><span className="font-medium">Ali Hassan</span> requested leave.</p>
                      <span className="text-xs text-gray-500">2 hours ago</span>
                    </div>
                  </div>
                  {/* Mock Notification 2 */}
                  <div className="flex gap-3 items-start p-2 hover:bg-gray-50 dark:hover:bg-white/5 rounded-lg transition-colors cursor-pointer">
                    <div className="h-8 w-8 rounded-full bg-green-100 dark:bg-green-500/20 flex items-center justify-center shrink-0">
                      <FileText size={14} className="text-green-600 dark:text-green-400" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-800 dark:text-gray-200">Monthly HR report generated.</p>
                      <span className="text-xs text-gray-500">1 day ago</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="hidden lg:block lg:h-6 lg:w-px lg:bg-gray-200 dark:lg:bg-white/10" aria-hidden="true" />
        
        {/* Profile Dropdown */}
        <div className="relative flex items-center gap-2 pl-2 sm:border-l sm:border-gray-200 sm:dark:border-white/10" ref={profileRef}>
          <button 
            onClick={() => {
                setProfileDropdownOpen(!profileDropdownOpen);
                setNotificationDropdownOpen(false);
            }}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-300 transition-colors hover:bg-gray-200 dark:hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <User size={18} />
          </button>
          <span className="hidden text-sm font-medium text-gray-700 dark:text-gray-200 sm:block">
            HR Manager
          </span>

          <AnimatePresence>
            {profileDropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.15, ease: 'easeOut' }}
                className="absolute right-0 top-12 w-48 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-1.5 shadow-xl glass-card overflow-hidden z-50"
              >
                <div className="px-3 py-2 border-b border-gray-100 dark:border-white/5 mb-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">HR Admin</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 truncate">hr@heegan.edu</p>
                </div>
                
                <button
                  onClick={() => {
                    setProfileDropdownOpen(false);
                    openEditProfile();
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-white/10 dark:hover:text-white transition-colors text-left"
                >
                  <UserCog size={16} />
                  <span>Edit Profile</span>
                </button>

                <div className="my-1 border-t border-gray-100 dark:border-white/5"></div>

                <button
                  onClick={handleLogout}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-500/10 transition-colors text-left"
                >
                  <LogOut size={16} />
                  <span>Logout</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}

export default function HRLayout() {
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
          "flex flex-col min-h-screen transition-all duration-300 ease-out relative",
          isMobile ? "ml-0" : (isCollapsed ? "ml-[80px]" : "ml-[280px]")
        )}
      >
        <HRTopbar />
        
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-x-hidden relative">
          {/* Ambient Background Accents */}
          <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-[120px] pointer-events-none" />
          <Outlet />
        </main>
      </div>
    </div>
  );
}
