import { useState, useRef, useEffect } from 'react';
import { Menu, LogOut, UserCog, KeyRound } from 'lucide-react';
import { useSidebarStore } from '@/store/useSidebarStore';
import { useAuthStore } from '@/store/useAuthStore';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { UserAvatar } from '@/components/ui/UserAvatar';
import { NotificationDropdown } from '@/components/ui/NotificationDropdown';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

interface TopbarProps {
  title?: string;
}

export function Topbar({ title = 'Dashboard' }: TopbarProps) {
  const { toggleSidebar } = useSidebarStore();
  const { logout, user } = useAuthStore();
  const navigate = useNavigate();

  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close profile dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
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
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-gray-200 dark:border-white/10 bg-white/50 dark:bg-dark-bg/50 px-4 backdrop-blur-xl sm:px-6">
      <div className="flex items-center gap-4">
        <button
          onClick={toggleSidebar}
          className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-white/10 lg:hidden"
        >
          <Menu size={20} />
        </button>
        <h1 className="text-xl font-semibold text-gray-800 dark:text-white">
          {title}
        </h1>
      </div>

      <div className="flex items-center gap-3 sm:gap-4">
        <ThemeToggle />

        {/* Real Notifications */}
        <NotificationDropdown />

        <div className="relative flex items-center gap-2 pl-2 sm:border-l sm:border-gray-200 sm:dark:border-white/10" ref={dropdownRef}>
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-300 transition-colors hover:bg-gray-200 dark:hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-primary/50 overflow-hidden"
          >
            <UserAvatar imageUrl={user?.profile_image_url} username={user?.username} size={36} />
          </button>
          <span className="hidden text-sm font-medium text-gray-700 dark:text-gray-200 sm:block">
            {user?.username || 'Admin'}
          </span>

          {/* Profile Dropdown */}
          <AnimatePresence>
            {dropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.15, ease: 'easeOut' }}
                className="absolute right-0 top-12 w-48 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-1.5 shadow-xl glass-card overflow-hidden z-50"
              >
                <div className="px-3 py-2 border-b border-gray-100 dark:border-white/5 mb-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">{user?.username || 'Admin'}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{user?.email || 'admin@heegan.edu'}</p>
                </div>

                <button
                  onClick={() => {
                    setDropdownOpen(false);
                    navigate('/admin/settings');
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-white/10 dark:hover:text-white transition-colors text-left"
                >
                  <UserCog size={16} />
                  <span>Edit Profile</span>
                </button>

                <button
                  onClick={() => {
                    navigate('/admin/settings');
                    setDropdownOpen(false);
                  }}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-white/10 dark:hover:text-white transition-colors text-left"
                >
                  <KeyRound size={16} />
                  <span>Change Password</span>
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
