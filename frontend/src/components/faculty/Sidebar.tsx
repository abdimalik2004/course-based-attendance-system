import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard, 
  Users, 
  Calendar,
  ClipboardList,
  Menu
} from 'lucide-react';
import { useSidebarStore } from '@/store/useSidebarStore';
import { cn } from '@/utils/cn';
import logoUrl from '@/assets/logo.png';
import lightLogoUrl from '@/assets/light-logo.png';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', path: '/faculty/dashboard' },
  { icon: Users, label: 'Assign Teacher', path: '/faculty/assign-teacher' },
  { icon: Calendar, label: 'Schedule Course', path: '/faculty/schedule' },
  { icon: ClipboardList, label: 'Attendance List', path: '/faculty/attendance-list' },
];

export function Sidebar() {
  const { isCollapsed, toggleSidebar, setCollapsed } = useSidebarStore();
  const location = useLocation();
  const [isMobile, setIsMobile] = useState(false);

  // Handle responsive behavior
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

  // Close sidebar on mobile when navigating
  useEffect(() => {
    if (isMobile) setCollapsed(true);
  }, [location.pathname, isMobile, setCollapsed]);

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
          width: isCollapsed ? (isMobile ? 0 : 80) : 220,
          x: isMobile && isCollapsed ? -220 : 0
        }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-screen flex-col border-r border-gray-200 dark:border-white/10 bg-white dark:bg-dark-bg/95 backdrop-blur-xl",
          "select-none"
        )}
      >
        {/* Sidebar Header / Logo */}
        <div className="flex h-16 shrink-0 items-center justify-between px-4 border-b border-gray-200 dark:border-white/10">
          <div className="flex items-center gap-3 overflow-hidden">
            <button 
              onClick={isCollapsed ? toggleSidebar : undefined}
              className={cn(
                "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gray-100 dark:bg-white/5 p-1 transition-colors",
                isCollapsed && "hover:bg-gray-200 dark:hover:bg-white/10 cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary/50",
                !isCollapsed && "cursor-default"
              )}
              aria-label={isCollapsed ? "Expand Sidebar" : undefined}
            >
              <img src={lightLogoUrl} alt="Logo Light" className="dark:hidden h-full w-full rounded-lg object-cover" />
              <img src={logoUrl} alt="Logo Dark" className="hidden dark:block h-full w-full rounded-lg object-cover" />
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

          {/* Desktop Toggle Button (Hamburger) */}
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

        {/* Navigation Links */}
        <nav className="flex-1 space-y-1 overflow-y-auto overflow-x-hidden p-3 custom-scrollbar">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path || location.pathname.startsWith(`${item.path}/`);
            
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={cn(
                  "relative flex items-center gap-3 rounded-xl px-3 py-3 transition-all duration-200 group",
                  isActive
                    ? "bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5 hover:text-gray-900 dark:hover:text-gray-200"
                )}
                title={isCollapsed ? item.label : undefined}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeNavIndicatorFaculty"
                    className="absolute left-0 top-1/2 h-1/2 w-1 -translate-y-1/2 rounded-r-full bg-primary"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                <div className="flex shrink-0 items-center justify-center">
                  <item.icon size={22} className={cn(
                    "transition-colors",
                    isActive ? "text-gray-900 dark:text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.1)]" : "text-gray-500"
                  )} />
                </div>
                
                <AnimatePresence mode="wait">
                  {!isCollapsed && (
                    <motion.span
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: 'auto' }}
                      exit={{ opacity: 0, width: 0 }}
                      className="whitespace-nowrap font-medium"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </NavLink>
            );
          })}
        </nav>
      </motion.aside>
    </>
  );
}

export default Sidebar;
