import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useUsersStore } from '@/store/useUsersStore';
import type { User } from '@/types/users.types';

const editUserSchema = z.object({
  username: z.string().min(3, 'Username must be at least 3 characters'),
  email: z.string().email('Invalid email address'),
  role: z.string().min(1, 'Role is required'),
  facultyId: z.string().optional(),
  status: z.enum(['Active', 'Inactive']),
}).refine(data => {
  // Require facultyId if role is Faculty
  if (String(data.role || '').toUpperCase() === 'FACULTY' && !data.facultyId) {
    return false;
  }
  return true;
}, {
  message: 'Faculty is required when role is Faculty',
  path: ['facultyId']
});

type EditUserFormData = z.infer<typeof editUserSchema>;

const ROLE_LABEL_MAP: Record<string, string> = {
  SUPER_ADMIN: 'Administrator',
  ACADEMIA: 'Academic Office',
  ADMISSIONS: 'Admissions Office',
  HR: 'Human Resources',
  FACULTY: 'Faculty Admin',
  TEACHER: 'Teacher',
  STUDENT: 'Student',
};

interface EditUserModalProps {
  isOpen: boolean;
  onClose: () => void;
  user: User | null;
}

export function EditUserModal({ isOpen, onClose, user }: EditUserModalProps) {
  const { editUser, roles, faculties, fetchRolesAndFaculties } = useUsersStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
    reset,
    setValue
  } = useForm<EditUserFormData>({
    resolver: zodResolver(editUserSchema),
  });

  const selectedRole = watch('role');
  const isFacultyRole = String(selectedRole || '').toUpperCase() === 'FACULTY';

  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      fetchRolesAndFaculties();
      if (user) {
        setValue('username', user.username);
        setValue('email', user.email);
        setValue('role', user.role);
        setValue('status', user.status);
        if (user.facultyId) {
          setValue('facultyId', user.facultyId);
        }
      }
    } else {
      reset();
    }
  }, [isOpen, user, fetchRolesAndFaculties, reset, setValue]);

  const onSubmit = async (data: EditUserFormData) => {
    if (!user) return;
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      await editUser(user.id, {
        ...data,
        facultyId: isFacultyRole ? data.facultyId : null
      });
      onClose();
    } catch (err: any) {
      setSubmitError(err.message || "Failed to update user. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Edit User"
      className="md:max-w-md"
    >
      {user ? (
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Username
            </label>
            <Input
              placeholder="e.g. John Doe"
              {...register('username')}
              error={errors.username?.message}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Email
            </label>
            <Input
              type="email"
              placeholder="e.g. john@example.com"
              {...register('email')}
              error={errors.email?.message}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Role <span className="text-primary">*</span>
            </label>
            <Select
              options={roles.map(r => ({ value: r.name, label: ROLE_LABEL_MAP[r.name] ?? r.name }))}
              {...register('role')}
              error={errors.role?.message}
            />
          </div>

          {isFacultyRole && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1 mt-4">
                Faculty ID
              </label>
              <Select
                placeholder="Select Faculty"
                options={faculties.map(f => ({ value: f.id, label: f.name }))}
                {...register('facultyId')}
                error={errors.facultyId?.message}
              />
            </motion.div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
              Status
            </label>
            <Select
              options={[
                { value: 'Active', label: 'Active' },
                { value: 'Inactive', label: 'Inactive' }
              ]}
              {...register('status')}
              error={errors.status?.message}
            />
          </div>

          {submitError && (
            <div className="flex items-center gap-2 text-sm text-red-500 bg-red-50 dark:bg-red-500/10 rounded-lg px-3 py-2 mt-2">
              <span>{submitError}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-100 dark:border-white/5 mt-6">
            <Button
              type="button"
              variant="secondary"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" isLoading={isSubmitting}>
              Save Changes
            </Button>
          </div>
        </form>
      ) : null}
    </Modal>
  );
}
