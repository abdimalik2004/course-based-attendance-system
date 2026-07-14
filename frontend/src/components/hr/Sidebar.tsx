import { useState, useEffect } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  Users,
  FileText,
  Settings,
  Menu,
  LogOut,
} from 'lucide-react';
import { useSidebarStore } from '@/store/useSidebarStore';
import { useAuthStore } from '@/store/useAuthStore';
import { cn } from '@/utils/cn';
import logoUrl from '@/assets/logo.png';
import lightLogoUrl from '@/assets/light-logo.png';

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
}

interface NavGroup {
  heading: string;
  items: NavItem[];
}

const navGroups: NavGroup[] = [
  {
    heading: 'Overview',
    items: [
      { icon: LayoutDashboard, label: 'Dashboard', path: '/hr/dashboard' },
    ],
  },
  {
    heading: 'Staff',
    items: [
      { icon: Users, label: 'Teachers', path: '/hr/teachers' },
    ],
  },
  {
    heading: 'Tools',
    items: [
      { icon: FileText, label: 'Reports', path: '/hr/reports' },
    ],
  },
  {
    heading: 'System',
    items: [
      { icon: Settings, label: 'Settings', path: '/hr/settings' },
    ],
  },
];

export function Sidebar() {
  const { isCollapsed, toggleSidebar, setCollapsed } = useSidebarStore();
  const { user, logout } = useAuthStore();
  const location = useLocation();
  const navigate = useNavigate();
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setIsMobile(true);
        setCollapsed(true);
      } else {
        setIsMobile(false);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [setCollapsed]);

  useEffect(() => {
    if (isMobile) setCollapsed(true);
  }, [location.pathname, isMobile, setCollapsed]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const displayName = (() => {
    const parts = (user?.full_name ?? '').trim().split(/\s+/).filter(Boolean);
    if (parts.length >= 2) return `${parts[0]} ${parts[1]}`;
    if (parts.length === 1) return parts[0];
    return user?.username ?? 'HR';
  })();

  const avatarInitial = displayName.charAt(0).toUpperCase() || 'H';

  return (
    <>
      {/* Mobile Backdrop */}
      <AnimatePresence>
        {isMobile && !isCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setCollapsed(true)}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          />
        )}
      </AnimatePresence>

      <motion.aside
        initial={false}
        animate={{
          width: isCollapsed ? (isMobile ? 0 : 80) : 280,
          x: isMobile && isCollapsed ? -280 : 0,
        }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className="fixed inset-y-0 left-0 z-50 flex h-screen flex-col border-r border-gray-200 dark:border-white/10 bg-white dark:bg-dark-bg/95 backdrop-blur-xl select-none"
      >
        {/* Header / Logo */}
        <div className="flex h-16 shrink-0 items-center justify-between px-4 border-b border-gray-200 dark:border-white/10">
          <div className="flex items-center gap-3 overflow-hidden">
            <button
              onClick={isCollapsed ? toggleSidebar : undefined}
              className={cn(
                'flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gray-100 dark:bg-white/5 p-1 transition-colors',
                isCollapsed && 'hover:bg-gray-200 dark:hover:bg-white/10 cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary/50',
                !isCollapsed && 'cursor-default',
              )}
              aria-label={isCollapsed ? 'Expand Sidebar' : undefined}
            >
              <img src={lightLogoUrl} alt="Logo" className="dark:hidden h-full w-full rounded-lg object-cover" />
              <img src={logoUrl} alt="Logo" className="hidden dark:block h-full w-full rounded-lg object-cover" />
            </button>
            <AnimatePresence mode="wait">
              {!isCollapsed && (
                <motion.span
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  className="whitespace-nowrap text-lg font-bold text-gray-900 dark:text-white"
                >
                  Heegan
                </motion.span>
              )}
            </AnimatePresence>
          </div>

          {!isMobile && !isCollapsed && (
            <button
              onClick={toggleSidebar}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-white/10 dark:hover:text-white transition-colors"
              aria-label="Collapse Sidebar"
            >
              <Menu size={20} />
            </button>
          )}
        </div>

        {/* Navigation Groups */}
        <nav className="flex-1 space-y-0.5 overflow-y-auto overflow-x-hidden p-3 custom-scrollbar">
          {navGroups.map((group) => (
            <div key={group.heading} className="mb-4">
              <AnimatePresence mode="wait">
                {!isCollapsed && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="px-3 pb-1 pt-0.5 text-[10px] font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500"
                  >
                    {group.heading}
                  </motion.p>
                )}
              </AnimatePresence>

              {group.items.map((item) => {
                const isActive =
                  location.pathname === item.path || location.pathname.startsWith(`${item.path}/`);

                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={cn(
                      'relative flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-200 group',
                      isActive
                        ? 'bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white'
                        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5 hover:text-gray-900 dark:hover:text-gray-200',
                    )}
                    title={isCollapsed ? item.label : undefined}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeNavIndicatorHr"
                        className="absolute left-0 top-1/2 h-1/2 w-1 -translate-y-1/2 rounded-r-full bg-primary"
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      />
                    )}
                    <div className="flex shrink-0 items-center justify-center">
                      <item.icon
                        size={20}
                        className={cn(
                          'transition-colors',
                          isActive ? 'text-gray-900 dark:text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.1)]' : 'text-gray-500',
                        )}
                      />
                    </div>

                    <AnimatePresence mode="wait">
                      {!isCollapsed && (
                        <motion.span
                          initial={{ opacity: 0, width: 0 }}
                          animate={{ opacity: 1, width: 'auto' }}
                          exit={{ opacity: 0, width: 0 }}
                          className="whitespace-nowrap font-medium text-sm"
                        >
                          {item.label}
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </NavLink>
                );
              })}
            </div>
          ))}
        </nav>

        {/* Footer — user info + logout */}
        <div className="shrink-0 border-t border-gray-200 dark:border-white/10 p-3">
          <div
            onClick={() => navigate('/hr/profile')}
            title={isCollapsed ? 'View profile' : undefined}
            className={cn('flex items-center gap-3 rounded-xl px-3 py-2.5 cursor-pointer hover:bg-gray-100 dark:hover:bg-white/5 transition-colors', isCollapsed ? 'justify-center' : '')}
          >
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/20 text-primary dark:text-primary-accent font-semibold text-sm">
              {avatarInitial}
            </div>

            <AnimatePresence mode="wait">
              {!isCollapsed && (
                <motion.div
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  className="flex flex-1 items-center justify-between min-w-0"
                >
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold text-gray-900 dark:text-white">{displayName}</p>
                    <p className="truncate text-[10px] text-gray-500 dark:text-gray-400">HR Officer</p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleLogout(); }}
                    title="Log out"
                    className="ml-2 shrink-0 flex h-7 w-7 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-white/10 dark:hover:text-white transition-colors"
                  >
                    <LogOut size={15} />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.aside>
    </>
  );
}

export default Sidebar;
