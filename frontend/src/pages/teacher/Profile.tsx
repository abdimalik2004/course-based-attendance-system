import { motion } from "framer-motion";
import {
  User,
  Building2,
  Briefcase,
  GraduationCap,
  Phone,
  Mail,
  Calendar,
  Hash,
  Link as LinkIcon,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { useTeacherStore, useTeacherId } from "@/store/useTeacherStore";

interface ProfileFieldProps {
  icon: React.ElementType;
  label: string;
  value: string | null | undefined;
  mono?: boolean;
  highlight?: "green" | "orange" | "blue";
}

function ProfileField({ icon: Icon, label, value, mono, highlight }: ProfileFieldProps) {
  const colorMap = {
    green: "text-emerald-600 dark:text-emerald-400",
    orange: "text-orange-500 dark:text-orange-400",
    blue: "text-blue-600 dark:text-blue-400",
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
              highlight ? colorMap[highlight] : "text-gray-900 dark:text-white"
            } ${mono ? "font-mono" : ""}`}
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

export default function TeacherProfile() {
  const profile = useTeacherStore((s) => s.profile);
  const profileLoading = useTeacherStore((s) => s.profileLoading);
  const profileError = useTeacherStore((s) => s.profileError);
  const { isUnlinked } = useTeacherId();

  if (profileLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">My Profile</h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Your teacher profile information.</p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="glass-card p-6 rounded-2xl animate-pulse space-y-4">
              <div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-1/2" />
              {Array.from({ length: 4 }).map((_, j) => (
                <div key={j} className="flex items-center gap-3">
                  <div className="h-8 w-8 bg-gray-200 dark:bg-white/10 rounded-lg shrink-0" />
                  <div className="flex-1 space-y-1.5">
                    <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-16" />
                    <div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-32" />
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (isUnlinked || profileError) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">My Profile</h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Your teacher profile information.</p>
        </div>
        <div className="glass-card rounded-2xl p-10 text-center max-w-sm mx-auto">
          <div className="w-14 h-14 rounded-2xl bg-amber-500/10 text-amber-500 flex items-center justify-center mx-auto mb-4">
            <LinkIcon size={24} />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {profileError ? "Profile Load Error" : "Account Not Linked"}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 leading-relaxed">
            {profileError ?? "Your account hasn't been linked to a teacher profile yet. Contact HR to link your account."}
          </p>
        </div>
      </div>
    );
  }

  if (!profile) return null;

  const statusIsActive = String(profile.status ?? "").toLowerCase() === "active";

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">My Profile</h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
            Your teacher profile as registered in the HR system.
          </p>
        </div>
        <div
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold shrink-0 ${
            statusIsActive
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400"
              : "bg-orange-100 text-orange-700 dark:bg-orange-500/20 dark:text-orange-400"
          }`}
        >
          {statusIsActive ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
          {profile.status}
        </div>
      </div>

      {/* Avatar / name hero card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="glass-card rounded-2xl p-6 flex items-center gap-5"
      >
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/30 to-purple-500/30 text-primary flex items-center justify-center text-2xl font-bold shrink-0">
          {profile.full_name?.charAt(0)?.toUpperCase() ?? "T"}
        </div>
        <div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
            {profile.full_name}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5 capitalize">
            {profile.role} · {profile.teacher_number}
          </p>
          {profile.linked_username && (
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1 flex items-center gap-1">
              <LinkIcon size={11} />
              Linked to <span className="font-mono">{profile.linked_username}</span>
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
          <ProfileField icon={Hash} label="Teacher Number" value={profile.teacher_number} mono />
          <ProfileField
            icon={GraduationCap}
            label="Full Name"
            value={profile.full_name}
          />
          <ProfileField
            icon={Briefcase}
            label="Role"
            value={profile.role ? profile.role.charAt(0).toUpperCase() + profile.role.slice(1) : null}
          />
          <ProfileField
            icon={CheckCircle2}
            label="Status"
            value={profile.status}
            highlight={statusIsActive ? "green" : "orange"}
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
            value={profile.faculty_name ?? null}
          />
          <ProfileField
            icon={Briefcase}
            label="Department"
            value={profile.department_name ?? null}
          />
          <ProfileField
            icon={Calendar}
            label="Hire Date"
            value={
              profile.hire_date
                ? new Date(profile.hire_date).toLocaleDateString([], {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })
                : null
            }
          />
          <ProfileField
            icon={LinkIcon}
            label="Linked Account"
            value={profile.linked_username}
            mono
            highlight={profile.linked_username ? "blue" : undefined}
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
              Contact HR to update your profile information or link a different account.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
