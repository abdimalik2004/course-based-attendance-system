import { motion, AnimatePresence } from 'framer-motion';
import { X, Building, GraduationCap, User, Fingerprint, Briefcase, Activity } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import type { Teacher } from '@/services/hrService';
import { useHrStore } from '@/store/useHrStore';

interface ViewTeacherModalProps {
  teacher: Teacher | null;
  isOpen: boolean;
  onClose: () => void;
}

export function ViewTeacherModal({ teacher, isOpen, onClose }: ViewTeacherModalProps) {
  const { faculties, departments } = useHrStore();

  if (!teacher) return null;

  const facultyName = faculties.find(f => f.id === teacher.facultyId)?.name || teacher.facultyId;
  const departmentName = departments.find(d => d.id === teacher.departmentId)?.name || teacher.departmentId;

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-0">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            className="relative w-full max-w-lg glass-card overflow-hidden rounded-2xl border border-gray-200 dark:border-white/10 shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-gray-200 dark:border-white/10 p-6 bg-gray-50/50 dark:bg-white/5">
              <div className="flex items-center gap-4">
                <div className="h-14 w-14 rounded-full bg-primary/10 flex items-center justify-center text-primary text-xl font-bold border border-primary/20">
                  {teacher.fullName.split(' ').map(n => n[0]).join('').substring(0, 2)}
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                    {teacher.fullName}
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                    {teacher.id}
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-200 dark:hover:bg-white/10 focus:outline-none"
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6">
              
              {/* Status & Role Section */}
              <div className="flex flex-wrap gap-3">
                <Badge variant={
                  teacher.status === 'Active' ? 'success' : 
                  teacher.status === 'On Leave' ? 'warning' : 'danger'
                } className="flex items-center gap-1.5 py-1 px-3">
                  <Activity size={14} />
                  {teacher.status}
                </Badge>
                <Badge variant="secondary" className="flex items-center gap-1.5 py-1 px-3">
                  <Briefcase size={14} />
                  {teacher.role}
                </Badge>
              </div>

              {/* Details Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1 p-3 bg-gray-50 dark:bg-white/5 rounded-xl border border-gray-100 dark:border-white/5">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
                    <Building size={16} />
                    <span className="font-medium">Faculty</span>
                  </div>
                  <p className="text-gray-900 dark:text-gray-100 font-medium pl-6">{facultyName}</p>
                </div>

                <div className="space-y-1 p-3 bg-gray-50 dark:bg-white/5 rounded-xl border border-gray-100 dark:border-white/5">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
                    <GraduationCap size={16} />
                    <span className="font-medium">Department</span>
                  </div>
                  <p className="text-gray-900 dark:text-gray-100 font-medium pl-6">{departmentName}</p>
                </div>

                <div className="space-y-1 p-3 bg-gray-50 dark:bg-white/5 rounded-xl border border-gray-100 dark:border-white/5">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
                    <User size={16} />
                    <span className="font-medium">User Account ID</span>
                  </div>
                  <p className="text-gray-900 dark:text-gray-100 font-mono font-medium pl-6">{teacher.userId}</p>
                </div>

                <div className="space-y-1 p-3 bg-gray-50 dark:bg-white/5 rounded-xl border border-gray-100 dark:border-white/5">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-1">
                    <Fingerprint size={16} />
                    <span className="font-medium">System ID</span>
                  </div>
                  <p className="text-gray-900 dark:text-gray-100 font-mono font-medium pl-6">{teacher.id}</p>
                </div>
              </div>

            </div>

            {/* Footer */}
            <div className="border-t border-gray-200 dark:border-white/10 p-4 bg-gray-50/50 dark:bg-white/5 flex justify-end">
              <Button onClick={onClose} variant="secondary">
                Close
              </Button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
