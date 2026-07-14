import { motion } from 'framer-motion';
import { User, Mail, Hash, ShieldCheck, CheckCircle2, XCircle, Pencil, Camera } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '@/store/useAuthStore';
import { useUIStore } from '@/store/useUIStore';
import { EditProfileModal } from '@/components/ui/EditProfileModal';
import { api } from '@/services/api';

const API_URL = import.meta.env.VITE_API_URL ?? '';

function resolveUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.startsWith('http') ? url : `${API_URL}${url}`;
}

interface MeResponse {
  id: number;
  username: string;
  email: string | null;
  is_active: boolean;
  faculty_id: number | null;
  faculty_name: string | null;
  teacher_id: number | null;
  student_id: number | null;
  student_number: string | null;
  role_names: string[];
  profile_image_url: string | null;
  full_name: string | null;
}

interface ProfileFieldProps {
  icon: React.ElementType;
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}

function ProfileField({ icon: Icon, label, value, mono }: ProfileFieldProps) {
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
          <p className={`text-sm font-medium mt-0.5 text-gray-900 dark:text-white ${mono ? 'font-mono' : ''}`}>
            {value}
          </p>
        ) : (
          <p className="text-sm text-gray-400 dark:text-gray-500 mt-0.5 italic">Not set</p>
        )}
      </div>
    </div>
  );
}

function SkeletonCard({ rows = 3 }: { rows?: number }) {
  return (
    <div className="glass-card p-6 rounded-2xl animate-pulse space-y-4">
      <div className="h-3.5 bg-gray-200 dark:bg-white/10 rounded w-1/4" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="h-8 w-8 bg-gray-200 dark:bg-white/10 rounded-lg shrink-0" />
          <div className="flex-1 space-y-1.5">
            <div className="h-2 bg-gray-200 dark:bg-white/10 rounded w-14" />
            <div className="h-3.5 bg-gray-200 dark:bg-white/10 rounded w-32" />
          </div>
        </div>
      ))}
    </div>
  );
}

const ROLE_LABELS: Record<string, string> = {
  SUPER_ADMIN: 'Super Admin',
  ACADEMIA:    'Academia',
  FACULTY:     'Faculty',
  HR:          'HR Officer',
  ADMISSIONS:  'Admissions',
  TEACHER:     'Teacher',
  STUDENT:     'Student',
};

export default function UserProfile() {
  const { user } = useAuthStore();
  const { openEditProfile } = useUIStore();

  const { data: me, isLoading } = useQuery<MeResponse>({
    queryKey: ['me'],
    queryFn: () => api.get('/auth/me').then((r) => r.data),
    staleTime: 1000 * 60 * 5,
  });

  // Derive display values — prefer fresh /me data, fall back to auth store
  const fullName   = me?.full_name   ?? user?.full_name   ?? null;
  const username   = me?.username    ?? user?.username    ?? '—';
  const email      = me?.email       ?? user?.email       ?? null;
  const isActive   = me?.is_active   ?? true;
  const roles      = me?.role_names  ?? (user?.role ? [user.role] : []);
  const photoUrl   = resolveUrl(me?.profile_image_url ?? user?.profile_image_url);

  // Role-based display name fallback when full_name is not set
  const roleFallback = (() => {
    const primaryRole = roles[0] ?? '';
    if (primaryRole === 'SUPER_ADMIN') return 'Administrator';
    if (primaryRole === 'ACADEMIA')    return 'Academia Office';
    if (primaryRole === 'ADMISSIONS')  return 'Admission Office';
    if (primaryRole === 'HR')          return 'Human Resource Office';
    if (primaryRole === 'FACULTY')     return me?.faculty_name ?? 'Faculty Name';
    return null;
  })();

  // What to show: real name → role fallback → username
  const displayName = fullName ?? roleFallback ?? username;

  // Label for the name field changes based on role
  const nameFieldLabel = (() => {
    const primaryRole = roles[0] ?? '';
    if (primaryRole === 'FACULTY')  return 'Faculty Name';
    if (primaryRole === 'TEACHER')  return 'Teacher Name';
    if (['SUPER_ADMIN', 'ACADEMIA', 'ADMISSIONS', 'HR'].includes(primaryRole)) return 'Office';
    return 'Full Name';
  })();

  const initials = displayName
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('') || '?';

  const primaryRoleLabel = roles.length > 0 ? (ROLE_LABELS[roles[0]] ?? roles[0]) : '—';

  return (
    <>
      <EditProfileModal />

      <div className="space-y-6">
        {/* Page header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">My Profile</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">Your account information and settings.</p>
          </div>
          <button
            onClick={openEditProfile}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-primary text-white text-sm font-semibold hover:bg-primary/90 transition-colors shrink-0"
          >
            <Pencil size={15} />
            Edit Profile
          </button>
        </div>

        {/* Hero card */}
        {isLoading ? (
          <div className="glass-card rounded-2xl p-6 flex items-center gap-5 animate-pulse">
            <div className="w-20 h-20 rounded-2xl bg-gray-200 dark:bg-white/10 shrink-0" />
            <div className="space-y-2 flex-1">
              <div className="h-5 bg-gray-200 dark:bg-white/10 rounded w-40" />
              <div className="h-3.5 bg-gray-200 dark:bg-white/10 rounded w-24" />
            </div>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="glass-card rounded-2xl p-6 flex items-center gap-5"
          >
            {/* Avatar */}
            <div className="relative shrink-0">
              <div className="w-20 h-20 rounded-2xl overflow-hidden bg-gradient-to-br from-primary/30 to-purple-500/20 flex items-center justify-center text-2xl font-bold text-primary dark:text-white border border-gray-200 dark:border-white/10">
                {photoUrl ? (
                  <img src={photoUrl} alt="Profile" className="w-full h-full object-cover" />
                ) : (
                  initials
                )}
              </div>
              <button
                onClick={openEditProfile}
                title="Change photo"
                className="absolute -bottom-1.5 -right-1.5 flex h-7 w-7 items-center justify-center rounded-full bg-primary text-white shadow-md hover:bg-primary/90 transition-colors"
              >
                <Camera size={13} />
              </button>
            </div>

            <div className="min-w-0">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white truncate">
                {displayName}
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                {primaryRoleLabel}
                {roles.length > 1 && (
                  <span className="text-gray-400 dark:text-gray-500">
                    {' '}+ {roles.length - 1} more
                  </span>
                )}
              </p>
              {/* Active status badge */}
              <div className={`inline-flex items-center gap-1.5 mt-2 px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                isActive
                  ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400'
                  : 'bg-rose-100 text-rose-700 dark:bg-rose-500/15 dark:text-rose-400'
              }`}>
                {isActive ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                {isActive ? 'Active' : 'Inactive'}
              </div>
            </div>
          </motion.div>
        )}

        {/* Info cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Account */}
          {isLoading ? <SkeletonCard rows={3} /> : (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.05 }}
              className="glass-card p-6 rounded-2xl"
            >
              <h3 className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-1 flex items-center gap-2">
                <User size={13} /> Account
              </h3>
              <ProfileField icon={User}  label={nameFieldLabel} value={displayName} />
              <ProfileField icon={Hash}  label="Username"  value={username} mono />
              <ProfileField icon={Mail}  label="Email"     value={email} />
            </motion.div>
          )}

          {/* Roles & Permissions */}
          {isLoading ? <SkeletonCard rows={2} /> : (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              className="glass-card p-6 rounded-2xl"
            >
              <h3 className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                <ShieldCheck size={13} /> Roles & Access
              </h3>

              {roles.length === 0 ? (
                <p className="text-sm text-gray-400 italic">No roles assigned</p>
              ) : (
                <div className="space-y-2">
                  {roles.map((r) => (
                    <div key={r} className="flex items-center gap-3 py-2 border-b border-gray-100 dark:border-white/5 last:border-0">
                      <div className="p-2 rounded-lg bg-primary/10 text-primary dark:text-primary-accent shrink-0">
                        <ShieldCheck size={14} />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {ROLE_LABELS[r] ?? r}
                        </p>
                        <p className="text-[10px] text-gray-400 dark:text-gray-500 font-mono uppercase">{r}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-4 pt-3 border-t border-gray-100 dark:border-white/5">
                <p className="text-xs text-gray-400 dark:text-gray-500 leading-relaxed">
                  Role changes must be made by a system administrator.
                </p>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </>
  );
}
