import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const facultySchema = z.object({
  code: z.string().min(2, 'Code must be at least 2 characters'),
  name: z.string().min(3, 'Name is required'),
  dean: z.string().min(3, 'Dean name is required'),
  established: z.string().min(4, 'Year is required'),
  status: z.enum(['Active', 'Inactive', 'Draft']),
});

type FacultyFormData = z.infer<typeof facultySchema>;

export function AddFacultyModal() {
  const { isFacultyModalOpen, setFacultyModalOpen, addFaculty } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, handleSubmit, formState: { errors }, reset } = useForm<FacultyFormData>({
    resolver: zodResolver(facultySchema),
    defaultValues: { status: 'Active' }
  });

  useEffect(() => {
    if (isFacultyModalOpen) reset();
  }, [isFacultyModalOpen, reset]);

  const onSubmit = async (data: FacultyFormData) => {
    setIsSubmitting(true);
    try {
      await addFaculty(data);
      reset();
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isFacultyModalOpen} onClose={() => setFacultyModalOpen(false)} title="Create Faculty" className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty Code</label>
          <Input placeholder="e.g. SCI" {...register('code')} error={errors.code?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty Name</label>
          <Input placeholder="e.g. Faculty of Science" {...register('name')} error={errors.name?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Dean</label>
          <Input placeholder="e.g. Dr. Alan Turing" {...register('dean')} error={errors.dean?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Established Year</label>
          <Input placeholder="e.g. 2026" {...register('established')} error={errors.established?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Status</label>
          <Select options={[{ value: 'Active', label: 'Active' }, { value: 'Inactive', label: 'Inactive' }]} {...register('status')} error={errors.status?.message} />
        </div>
        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => setFacultyModalOpen(false)}>Cancel</Button>
          <Button type="submit" isLoading={isSubmitting}>Create Faculty</Button>
        </div>
      </form>
    </Modal>
  );
}
