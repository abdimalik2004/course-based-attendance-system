import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const courseSchema = z.object({
  facultyId: z.string().min(1, 'Faculty is required'),
  departmentId: z.string().min(1, 'Department is required'),
  title: z.string().min(3, 'Title is required'),
});

type CourseFormData = z.infer<typeof courseSchema>;

export function CourseModal() {
  const { courseModal, closeModal, addCourse, updateCourse, faculties, departments } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const { isOpen, mode, record } = courseModal;
  const isViewMode = mode === 'view';

  const { register, handleSubmit, formState: { errors }, reset, watch, setValue } = useForm<CourseFormData>({
    resolver: zodResolver(courseSchema),
  });

  const selectedFacultyId = watch('facultyId');
  const availableDepartments = departments.filter(d => d.facultyId === selectedFacultyId);

  useEffect(() => {
    if (isOpen) {
      setSubmitError(null);
      if (record) {
        reset({ facultyId: record.facultyId, departmentId: record.departmentId, title: record.title });
      } else {
        reset({ facultyId: '', departmentId: '', title: '' });
      }
    }
  }, [isOpen, record, reset]);

  const onSubmit = async (data: CourseFormData) => {
    if (isViewMode) return;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      if (mode === 'edit' && record) {
        await updateCourse(record.id, data);
      } else {
        await addCourse(data);
      }
      closeModal('course');
    } catch (error: any) {
      const msg =
        error?.response?.data?.detail ??
        error?.response?.data?.error?.message ??
        (error instanceof Error ? error.message : null) ??
        (mode === 'edit' ? 'Failed to update course' : 'Failed to create course');
      setSubmitError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  const titles = {
    create: 'Create Course',
    edit: 'Edit Course',
    view: 'Course Details'
  };

  return (
    <Modal isOpen={isOpen} onClose={() => closeModal('course')} title={titles[mode]} className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {submitError && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {submitError}
          </div>
        )}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty Name</label>
          <Select
            options={[
              { value: '', label: 'Select Faculty...' },
              ...faculties.map(f => ({ value: f.id, label: f.name }))
            ]}
            {...register('facultyId')}
            onChange={(e) => {
              register('facultyId').onChange(e);
              // Clear department silently — don't trigger validation until the user submits
              setValue('departmentId', '', { shouldValidate: false });
            }}
            error={errors.facultyId?.message}
            disabled={isViewMode}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department</label>
          <Select
            options={[
              { value: '', label: 'Select Department...' },
              ...availableDepartments.map(d => ({ value: d.id, label: d.name }))
            ]}
            {...register('departmentId')}
            error={errors.departmentId?.message}
            disabled={isViewMode || !selectedFacultyId}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Course Title</label>
          <Input
            placeholder="e.g. Introduction to Physics"
            {...register('title')}
            error={errors.title?.message}
            disabled={isViewMode}
            className={isViewMode ? 'bg-gray-50 dark:bg-dark-bg text-gray-500' : ''}
          />
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => closeModal('course')}>
            {isViewMode ? 'Close' : 'Cancel'}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === 'edit' ? 'Save Changes' : 'Create Course'}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
