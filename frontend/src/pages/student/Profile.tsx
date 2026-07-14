import { motion } from 'framer-motion';
import {
  User,
  Building2,
  GraduationCap,
  Phone,
  Mail,
  Calendar,
  Hash,
  CheckCircle2,
  AlertCircle,
  BookOpen,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import dashboardService from '@/services/dashboardService';
import { useAuthStore } from '@/store/useAuthStore';

interface ProfileFieldProps {
  icon: React.ElementType;
  label: string;
  value: string | null | undefined;
  mono?: boolean;
  highlight?: 'green' | 'orange' | 'blue';
}

function ProfileField({ icon: Icon, label, value, mono, highlight }: ProfileFieldProps) {
  const colorMap = {
    green: 'text-emerald-600 dark:text-emerald-400',
    orange: 'text-orange-500 dark:text-orange-400',
    blue: 'text-blue-600 dark:text-blue-400',
  };
  return (
    <div className="flex items-start gap-3 py-3 border-b border-gray-100 dark:border-white/5 last:border-0">
      <div className="p-2 rounded-lg bg-gray-100 dark:bg-white/10 text-gray-500 dark:text-gray-400 shrink-0 mt-0.5">
        <Icon size={15} />
      </div>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wider font-semibold text-gray-400 dark:text-gray-500">
          {label}
        </p>
        {value ? (
          <p
            className={`text-sm font-medium mt-0.5 ${
              highlight ? colorMap[highlight] : 'text-gray-900 dark:text-white'
            } ${mono ? 'font-mono' : ''}`}
          >
            {value}
          </p>
        ) : (
          <p className="text-sm text-gray-400 dark:text-gray-500 mt-0.5">Not set</p>
        )}
      </div>
    </div>
  );
}

function SkeletonCard({ rows = 4 }: { rows?: number }) {
  return (
    <div className="glass-card p-6 rounded-2xl animate-pulse space-y-4">
      <div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-1/3" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="h-8 w-8 bg-gray-200 dark:bg-white/10 rounded-lg shrink-0" />
          <div className="flex-1 space-y-1.5">
            <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-16" />
            <div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-32" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function StudentProfile() {
  const { user } = useAuthStore();

  const { data: profile, isLoading, isError } = useQuery({
    queryKey: ['studentProfile'],
    queryFn: () => dashboardService.studentProfile(),
    staleTime: 1000 * 60 * 5,
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">My Profile</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">Your student profile and academic information.</p>
        </div>
        <div className="glass-card rounded-2xl p-6 animate-pulse flex items-center gap-5">
          <div className="w-16 h-16 rounded-2xl bg-gray-200 dark:bg-white/10 shrink-0" />
          <div className="space-y-2 flex-1">
            <div className="h-5 bg-gray-200 dark:bg-white/10 rounded w-40" />
            <div className="h-3.5 bg-gray-200 dark:bg-white/10 rounded w-28" />
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <SkeletonCard rows={4} />
          <SkeletonCard rows={3} />
          <SkeletonCard rows={3} />
        </div>
      </div>
    );
  }

  if (isError || !profile || Object.keys(profile).length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">My Profile</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">Your student profile and academic information.</p>
        </div>
        <div className="glass-card rounded-2xl p-10 text-center max-w-sm mx-auto">
          <div className="w-14 h-14 rounded-2xl bg-amber-500/10 text-amber-500 flex items-center justify-center mx-auto mb-4">
            <AlertCircle size={24} />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Profile Not Found</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 leading-relaxed">
            Your account hasn't been linked to a student record yet. Contact the admissions office.
          </p>
        </div>
      </div>
    );
  }

  const isApproved = String(profile.status ?? '').toLowerCase() === 'approved';
  const isPending = String(profile.status ?? '').toLowerCase() === 'pending';
  const initials = (profile.full_name ?? user?.full_name ?? '?')
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('');

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">My Profile</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Your student profile as registered in the system.
          </p>
        </div>
        <div
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold shrink-0 ${
            isApproved
              ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400'
              : isPending
              ? 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400'
              : 'bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-400'
          }`}
        >
          {isApproved ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
          {profile.status ?? 'Unknown'}
        </div>
      </div>

      {/* Hero card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="glass-card rounded-2xl p-6 flex items-center gap-5"
      >
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/30 to-purple-500/30 text-primary dark:text-white flex items-center justify-center text-2xl font-bold shrink-0">
          {initials}
        </div>
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            {profile.full_name ?? user?.full_name ?? '—'}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            Student · <span className="font-mono text-primary dark:text-primary-accent">{profile.student_number ?? user?.studentNumber ?? '—'}</span>
          </p>
          {profile.faculty_name && (
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              {profile.faculty_name}{profile.department_name ? ` · ${profile.department_name}` : ''}
            </p>
          )}
        </div>
      </motion.div>

      {/* Three info cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Identity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.05 }}
          className="glass-card p-6 rounded-2xl"
        >
          <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
            <User size={14} />
            Identity
          </h3>
          <ProfileField
            icon={Hash}
            label="Student Number"
            value={profile.student_number}
            mono
          />
          <ProfileField
            icon={GraduationCap}
            label="Full Name"
            value={profile.full_name}
          />
          <ProfileField
            icon={Hash}
            label="Username"
            value={profile.username}
            mono
          />
          <ProfileField
            icon={CheckCircle2}
            label="Status"
            value={profile.status}
            highlight={isApproved ? 'green' : isPending ? 'orange' : undefined}
          />
        </motion.div>

        {/* Organization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.1 }}
          className="glass-card p-6 rounded-2xl"
        >
          <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
            <Building2 size={14} />
            Organization
          </h3>
          <ProfileField
            icon={Building2}
            label="Faculty"
            value={profile.faculty_name}
          />
          <ProfileField
            icon={BookOpen}
            label="Department"
            value={profile.department_name}
          />
          <ProfileField
            icon={Calendar}
            label="Date of Birth"
            value={
              profile.date_of_birth
                ? new Date(profile.date_of_birth).toLocaleDateString([], {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                  })
                : null
            }
          />
          <ProfileField
            icon={Calendar}
            label="Enrolled On"
            value={
              profile.enrolled_at
                ? new Date(profile.enrolled_at).toLocaleDateString([], {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                  })
                : null
            }
          />
        </motion.div>

        {/* Contact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.15 }}
          className="glass-card p-6 rounded-2xl"
        >
          <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
            <Mail size={14} />
            Contact
          </h3>
          <ProfileField icon={Mail} label="Email" value={profile.email} />
          <ProfileField icon={Phone} label="Phone" value={profile.phone} />
          <div className="mt-4 pt-3 border-t border-gray-100 dark:border-white/5">
            <p className="text-xs text-gray-400 dark:text-gray-500 leading-relaxed">
              Contact the admissions office to update your personal information.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
