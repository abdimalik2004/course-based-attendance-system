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
  code: z.string().min(2, 'Code must be at least 2 characters'),
  name: z.string().min(3, 'Name is required'),
  departmentId: z.string().min(1, 'Department is required'),
  level: z.string().min(3, 'Level is required'),
  credits: z.coerce.number().min(1, 'Credits must be at least 1'),
  status: z.enum(['Active', 'Inactive', 'Draft']),
});

type CourseFormData = z.infer<typeof courseSchema>;

export function AddCourseModal() {
  const { isCourseModalOpen, setCourseModalOpen, addCourse, departments } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, handleSubmit, formState: { errors }, reset } = useForm<CourseFormData>({
    resolver: zodResolver(courseSchema),
    defaultValues: { status: 'Active', credits: 3, level: 'Undergraduate' }
  });

  useEffect(() => {
    if (isCourseModalOpen) reset();
  }, [isCourseModalOpen, reset]);

  const onSubmit = async (data: CourseFormData) => {
    setIsSubmitting(true);
    try {
      await addCourse(data);
      reset();
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isCourseModalOpen} onClose={() => setCourseModalOpen(false)} title="Create Course" className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department</label>
          <Select 
            options={departments.map(d => ({ value: d.id, label: d.name }))}
            {...register('departmentId')} 
            error={errors.departmentId?.message} 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Course Code</label>
          <Input placeholder="e.g. CS101" {...register('code')} error={errors.code?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Course Name</label>
          <Input placeholder="e.g. Intro to Programming" {...register('name')} error={errors.name?.message} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Level</label>
            <Select 
              options={[
                { value: 'Undergraduate', label: 'Undergraduate' },
                { value: 'Graduate', label: 'Graduate' },
                { value: 'PhD', label: 'PhD' }
              ]}
              {...register('level')} 
              error={errors.level?.message} 
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Credits</label>
            <Input type="number" placeholder="3" {...register('credits')} error={errors.credits?.message} />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Status</label>
          <Select options={[{ value: 'Active', label: 'Active' }, { value: 'Inactive', label: 'Inactive' }]} {...register('status')} error={errors.status?.message} />
        </div>
        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => setCourseModalOpen(false)}>Cancel</Button>
          <Button type="submit" isLoading={isSubmitting}>Create Course</Button>
        </div>
      </form>
    </Modal>
  );
}
