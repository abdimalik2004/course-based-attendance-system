import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const structureSchema = z.object({
  academicYear: z.string().min(4, 'Year must be e.g. 2026-2027'),
  term: z.string().min(3, 'Term name required'),
  startDate: z.string().min(1, 'Start Date is required'),
  endDate: z.string().min(1, 'End Date is required'),
  status: z.enum(['Active', 'Inactive', 'Draft']),
});

type StructureFormData = z.infer<typeof structureSchema>;

export function AddStructureModal() {
  const { structureModal, closeModal, addStructure } = useAcademiaStore();
  const isOpen = structureModal?.isOpen || false;
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, handleSubmit, formState: { errors }, reset } = useForm<StructureFormData>({
    resolver: zodResolver(structureSchema),
    defaultValues: { status: 'Draft' }
  });

  useEffect(() => {
    if (isOpen) reset();
  }, [isOpen, reset]);

  const onSubmit = async (data: StructureFormData) => {
    setIsSubmitting(true);
    try {
      await addStructure(data);
      reset();
      closeModal('structure');
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={() => closeModal('structure')} title="Create Academic Term" className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Academic Year</label>
          <Input placeholder="e.g. 2026-2027" {...register('academicYear')} error={errors.academicYear?.message} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Term Name</label>
          <Input placeholder="e.g. Fall Semester" {...register('term')} error={errors.term?.message} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Start Date</label>
            <Input type="date" {...register('startDate')} error={errors.startDate?.message} />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">End Date</label>
            <Input type="date" {...register('endDate')} error={errors.endDate?.message} />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Status</label>
          <Select options={[{ value: 'Active', label: 'Active' }, { value: 'Draft', label: 'Draft' }, { value: 'Inactive', label: 'Inactive' }]} {...register('status')} error={errors.status?.message} />
        </div>
        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => closeModal('structure')}>Cancel</Button>
          <Button type="submit" isLoading={isSubmitting}>Create Term</Button>
        </div>
      </form>
    </Modal>
  );
}
