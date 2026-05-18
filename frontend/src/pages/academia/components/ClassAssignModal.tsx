import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal } from '@/components/ui/Modal';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const schema = z.object({
  classId: z.string().min(1, 'Class is required'),
  courseId: z.string().min(1, 'Course is required'),
  facultyId: z.string().min(1, 'Faculty is required'),
  departmentId: z.string().min(1, 'Department is required'),
});

type FormData = z.infer<typeof schema>;

export function ClassAssignModal() {
  const { classAssignModal, closeModal, addClassAssignment, updateClassAssignment, classes, courses, faculties, departments } = useAcademiaStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isOpen = classAssignModal?.isOpen || false;
  const mode = classAssignModal?.mode || 'create';
  const record = classAssignModal?.record;

  const { register, handleSubmit, formState: { errors }, reset, watch, setValue } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  const selectedFacultyId = watch('facultyId');
  const [filteredDepartments, setFilteredDepartments] = useState<{value: string, label: string}[]>([]);

  useEffect(() => {
    if (isOpen) {
      if (record && mode !== 'create') {
        reset({
          classId: record.classId,
          courseId: record.courseId,
          facultyId: record.facultyId,
          departmentId: record.departmentId
        });
      } else {
        reset({ classId: '', courseId: '', facultyId: '', departmentId: '' });
      }
    }
  }, [isOpen, mode, record, reset]);

  // Dynamic Department filtering
  useEffect(() => {
    if (selectedFacultyId) {
      const filtered = departments
        .filter(d => d.facultyId === selectedFacultyId)
        .map(d => ({ value: d.id, label: d.name }));
      
      setFilteredDepartments(filtered);
      if (mode === 'create') {
        setValue('departmentId', '');
      }
    } else {
      setFilteredDepartments([]);
    }
  }, [selectedFacultyId, departments, setValue]);

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    try {
      if (mode === 'edit' && record) {
        await updateClassAssignment(record.id, data);
      } else {
        await addClassAssignment(data);
      }
      closeModal('classAssign');
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={() => closeModal('classAssign')} title="Assign Class to Course" className="md:max-w-md">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Class Name</label>
          <Select 
            options={[
              { value: '', label: 'Select Class...' },
              ...classes.map(c => ({ value: c.id, label: c.name }))
            ]}
            {...register('classId')} 
            error={errors.classId?.message} 
            disabled={mode === 'view'}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Course Title</label>
          <Select 
            options={[
              { value: '', label: 'Select Course...' },
              ...courses.map(c => ({ value: c.id, label: c.title }))
            ]}
            {...register('courseId')} 
            error={errors.courseId?.message} 
            disabled={mode === 'view'}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Faculty</label>
          <Select 
            options={[
              { value: '', label: 'Select Faculty...' },
              ...faculties.map(f => ({ value: f.id, label: f.name }))
            ]}
            {...register('facultyId')} 
            error={errors.facultyId?.message} 
            disabled={mode === 'view'}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">Department</label>
          <Select 
            options={[
              { value: '', label: 'Select Department...' },
              ...filteredDepartments
            ]}
            {...register('departmentId')} 
            error={errors.departmentId?.message} 
            disabled={!selectedFacultyId || filteredDepartments.length === 0 || mode === 'view'}
          />
          {!selectedFacultyId && <p className="text-xs text-gray-500 mt-1 ml-1">Select a faculty to filter departments.</p>}
          {selectedFacultyId && filteredDepartments.length === 0 && <p className="text-xs text-orange-500 mt-1 ml-1">No departments found for this faculty.</p>}
        </div>

        <div className="flex items-center justify-end gap-3 pt-6 mt-6">
          <Button type="button" variant="ghost" onClick={() => closeModal('classAssign')}>
            {mode === 'view' ? 'Close' : 'Cancel'}
          </Button>
          {mode !== 'view' && (
            <Button type="submit" isLoading={isSubmitting}>
              {mode === 'edit' ? 'Save Changes' : 'Assign Class'}
            </Button>
          )}
        </div>
      </form>
    </Modal>
  );
}
