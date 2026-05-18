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
  code: z.string().min(2, 'Code must be at least 2 characters'),
  name: z.string().min(3, 'Name is required'),
  facultyId: z.string().min(1, 'Faculty is required'),
  head: z.string().min(3, 'Head name is required'),
  status: z.enum(['Active', 'Inactive', 'Draft']),
});

type DepartmentFormData = z.infer<typeof departmentSchema>;

export function AddDepartmentModal() {
  const { isDepartmentModalOpen, setDepartmentModalOpen, addDepartment, faculties } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, handleSubmit, formState: { errors }, reset } = useForm<DepartmentFormData>({
    resolver: zodResolver(departmentSchema),
    defaultValues: { status: 'Active' }
  });

  useEffect(() => {
    if (isDepartmentModalOpen) reset();
  }, [isDepartmentModalOpen, reset]);

  const onSubmit = async (data: DepartmentFormData) => {
    setIsSubmitting(true);
    try {
      await addDepartment({ ...data, activeCourses: 0 }); // Initialize with 0 courses
      reset();
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isDepartmentModalOpen} onClose={() => setDepartmentModalOpen(false)} title="Create Department" className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty</label>
          <Select 
            options={faculties.map(f => ({ value: f.id, label: f.name }))}
            {...register('facultyId')} 
            error={errors.facultyId?.message} 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department Code</label>
          <Input placeholder="e.g. CS" {...register('code')} error={errors.code?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department Name</label>
          <Input placeholder="e.g. Computer Science" {...register('name')} error={errors.name?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Head of Department</label>
          <Input placeholder="e.g. Dr. Grace Hopper" {...register('head')} error={errors.head?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Status</label>
          <Select options={[{ value: 'Active', label: 'Active' }, { value: 'Inactive', label: 'Inactive' }]} {...register('status')} error={errors.status?.message} />
        </div>
        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => setDepartmentModalOpen(false)}>Cancel</Button>
          <Button type="submit" isLoading={isSubmitting}>Create Department</Button>
        </div>
      </form>
    </Modal>
  );
}
