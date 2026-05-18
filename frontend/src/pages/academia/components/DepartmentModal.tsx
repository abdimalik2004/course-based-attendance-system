import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const departmentSchema = z.object({
  facultyId: z.string().min(1, 'Faculty is required'),
  name: z.string().min(3, 'Name is required'),
  code: z.string().min(2, 'Code must be at least 2 characters'),
});

type DepartmentFormData = z.infer<typeof departmentSchema>;

export function DepartmentModal() {
  const { departmentModal, closeModal, addDepartment, updateDepartment, faculties } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { isOpen, mode, record } = departmentModal;
  const isViewMode = mode === 'view';

  const { register, handleSubmit, formState: { errors }, reset } = useForm<DepartmentFormData>({
    resolver: zodResolver(departmentSchema),
  });

  useEffect(() => {
    if (isOpen) {
      if (record) {
        reset({ facultyId: record.facultyId, name: record.name, code: record.code });
      } else {
        reset({ facultyId: '', name: '', code: '' });
      }
    }
  }, [isOpen, record, reset]);

  const onSubmit = async (data: DepartmentFormData) => {
    if (isViewMode) return;
    setIsSubmitting(true);
    try {
      if (mode === 'edit' && record) {
        await updateDepartment(record.id, data);
      } else {
        await addDepartment(data);
      }
      closeModal('department');
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const titles = {
    create: 'Create Department',
    edit: 'Edit Department',
    view: 'Department Details'
  };

  return (
    <Modal isOpen={isOpen} onClose={() => closeModal('department')} title={titles[mode]} className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty Name</label>
          <Select 
            options={[
              { value: '', label: 'Select Faculty...' },
              ...faculties.map(f => ({ value: f.id, label: f.name }))
            ]}
            {...register('facultyId')} 
            error={errors.facultyId?.message} 
            disabled={isViewMode}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department Name</label>
          <Input 
            placeholder="e.g. Computer Science" 
            {...register('name')} 
            error={errors.name?.message} 
            disabled={isViewMode}
            className={isViewMode ? 'bg-gray-50 dark:bg-dark-bg text-gray-500' : ''}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department Code</label>
          <Input 
            placeholder="e.g. CS" 
            {...register('code')} 
            error={errors.code?.message} 
            disabled={isViewMode}
            className={isViewMode ? 'bg-gray-50 dark:bg-dark-bg text-gray-500' : ''}
          />
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => closeModal('department')}>
            {isViewMode ? 'Close' : 'Cancel'}
          </Button>
          {!isViewMode && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === 'edit' ? 'Save Changes' : 'Create Department'}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
