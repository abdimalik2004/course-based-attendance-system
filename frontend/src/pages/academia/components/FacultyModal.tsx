import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const facultySchema = z.object({
  name: z.string().min(3, 'Name must be at least 3 characters'),
  code: z.string().min(2, 'Code is required'),
  years: z.coerce.number().min(1, 'Years must be at least 1').max(10, 'Years must not exceed 10'),
});

type FacultyFormData = z.infer<typeof facultySchema>;

export function FacultyModal() {
  const { facultyModal, closeModal, addFaculty, updateFaculty } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const { isOpen, mode, record } = facultyModal;
  const isViewMode = mode === 'view';

  const { register, handleSubmit, formState: { errors }, reset } = useForm<FacultyFormData>({
    resolver: zodResolver(facultySchema),
  });

  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      if (record) {
        reset({ name: record.name, code: record.code, years: record.years });
      } else {
        reset({ name: '', code: '', years: 4 });
      }
    }
  }, [isOpen, record, reset]);

  const onSubmit = async (data: FacultyFormData) => {
    if (isViewMode) return;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (mode === 'edit' && record) {
        await updateFaculty(record.id, data);
      } else {
        await addFaculty(data);
      }
      closeModal('faculty');
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        (mode === 'edit' ? 'Failed to update faculty' : 'Failed to create faculty');
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  const titles = {
    create: 'Create Faculty',
    edit: 'Edit Faculty',
    view: 'Faculty Details'
  };

  return (
    <Modal isOpen={isOpen} onClose={() => closeModal('faculty')} title={titles[mode]} className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty Name</label>
          <Input
            placeholder="Enter faculty name"
            {...register('name')}
            error={errors.name?.message}
            disabled={isViewMode}
            className={isViewMode ? 'bg-gray-50 dark:bg-dark-bg text-gray-500' : ''}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Code</label>
          <Input
            placeholder="e.g. ENG"
            {...register('code')}
            error={errors.code?.message}
            disabled={isViewMode}
            className={isViewMode ? 'bg-gray-50 dark:bg-dark-bg text-gray-500' : ''}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Years</label>
          <Input
            type="number"
            placeholder="e.g., 3 or 4"
            {...register('years')}
            error={errors.years?.message}
            disabled={isViewMode}
            className={isViewMode ? 'bg-gray-50 dark:bg-dark-bg text-gray-500' : ''}
          />
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => closeModal('faculty')}>
            {isViewMode ? 'Close' : 'Cancel'}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === 'edit' ? 'Save Changes' : 'Create Faculty'}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
