import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const classSchema = z.object({
  code: z.string().min(2, 'Code must be at least 2 characters'),
  name: z.string().min(3, 'Name is required'),
  courseId: z.string().min(1, 'Course is required'),
  semester: z.string().min(3, 'Semester is required'),
  room: z.string().min(1, 'Room is required'),
  instructor: z.string().min(3, 'Instructor is required'),
  status: z.enum(['Active', 'Inactive', 'Draft']),
});

type ClassFormData = z.infer<typeof classSchema>;

export function AddClassModal() {
  const { isClassModalOpen, setClassModalOpen, addClass, courses } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, handleSubmit, formState: { errors }, reset } = useForm<ClassFormData>({
    resolver: zodResolver(classSchema),
    defaultValues: { status: 'Active' }
  });

  useEffect(() => {
    if (isClassModalOpen) reset();
  }, [isClassModalOpen, reset]);

  const onSubmit = async (data: ClassFormData) => {
    setIsSubmitting(true);
    try {
      await addClass(data);
      reset();
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isClassModalOpen} onClose={() => setClassModalOpen(false)} title="Create Class" className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Course</label>
          <Select 
            options={courses.map(c => ({ value: c.id, label: c.name }))}
            {...register('courseId')} 
            error={errors.courseId?.message} 
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Class Code</label>
            <Input placeholder="e.g. CS101-A" {...register('code')} error={errors.code?.message} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Semester</label>
            <Input placeholder="e.g. Fall 2026" {...register('semester')} error={errors.semester?.message} />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Class Name / Section</label>
          <Input placeholder="e.g. Intro to Programming A" {...register('name')} error={errors.name?.message} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Room</label>
            <Input placeholder="e.g. Room 101" {...register('room')} error={errors.room?.message} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Instructor</label>
            <Input placeholder="e.g. Prof. Smith" {...register('instructor')} error={errors.instructor?.message} />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Status</label>
          <Select options={[{ value: 'Active', label: 'Active' }, { value: 'Inactive', label: 'Inactive' }]} {...register('status')} error={errors.status?.message} />
        </div>
        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => setClassModalOpen(false)}>Cancel</Button>
          <Button type="submit" isLoading={isSubmitting}>Create Class</Button>
        </div>
      </form>
    </Modal>
  );
}
