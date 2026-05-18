import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, X } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAttendanceStore } from '@/store/useAttendanceStore';
import { useAcademiaStore } from '@/store/useAcademiaStore';
import ScannerInterface from '@/components/attendance/ScannerInterface';
import { cn } from '@/utils/cn';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

const adminAttendanceSchema = z.object({
  faculty_id: z.string().min(1, 'Faculty is required'),
  department_id: z.string().min(1, 'Department is required'),
  course_id: z.string().min(1, 'Course is required'),
  class_id: z.string().min(1, 'Class is required'),
  session_type: z.enum(['Lecture', 'Lab', 'Tutorial'], {
    message: 'Session type is required',
  }),
  camera_index: z.coerce.number().int().min(0, 'Camera index must be 0 or greater'),
  notes: z.string().optional(),
});

type AdminAttendanceForm = z.infer<typeof adminAttendanceSchema>;

export default function Attendance() {
  const { sessionState, startSession, resetSession } = useAttendanceStore();
  const { faculties, departments, courses, classes, isLoading: academiaLoading, error: academiaError, fetchData } = useAcademiaStore();

  // Reset session when navigating to this page
  useEffect(() => {
    resetSession();
  }, [resetSession]);

  useEffect(() => {
    if (!academiaLoading && faculties.length === 0) {
      fetchData();
    }
  }, [fetchData, faculties.length, academiaLoading]);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors },
  } = useForm<AdminAttendanceForm>({
    resolver: zodResolver(adminAttendanceSchema),
    defaultValues: {
      faculty_id: '',
      department_id: '',
      course_id: '',
      class_id: '',
      session_type: '' as any,
      camera_index: 0,
      notes: '',
    },
  });

  const selectedFacultyId = watch('faculty_id');
  const selectedDepartmentId = watch('department_id');

  const filteredDepartments = departments.filter((d) => d.facultyId === selectedFacultyId);
  const filteredCourses = courses.filter((c) => c.departmentId === selectedDepartmentId);
  const filteredClasses = classes.filter((cls) => cls.departmentId === selectedDepartmentId || cls.facultyId === selectedFacultyId);

  const handleCourseChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const courseId = e.target.value;
    setValue('course_id', courseId, { shouldValidate: true });
    setValue('class_id', '', { shouldValidate: true });
  };

  const handleFacultyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setValue('faculty_id', e.target.value, { shouldValidate: true });
    setValue('department_id', '', { shouldValidate: true });
    setValue('course_id', '', { shouldValidate: true });
    setValue('class_id', '', { shouldValidate: true });
  };

  const handleDepartmentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setValue('department_id', e.target.value, { shouldValidate: true });
    setValue('course_id', '', { shouldValidate: true });
    setValue('class_id', '', { shouldValidate: true });
  };

  const onFormSubmit = (data: AdminAttendanceForm) => {
    console.log("Starting session with data:", data);
    startSession();
  };

  const isScanningActive = sessionState === 'starting' || sessionState === 'waiting_for_face' || sessionState === 'face_detected' || sessionState === 'scanning';

  return (
    <div className="relative flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] p-4 overflow-hidden">
      
      {/* Dynamic Background Glow based on state */}
      <div 
        className={cn(
          "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full blur-[120px] opacity-20 pointer-events-none transition-colors duration-1000",
          sessionState === 'idle' && "bg-transparent",
          isScanningActive && "bg-primary",
          sessionState === 'success' && "bg-emerald-500",
          sessionState === 'failed' && "bg-rose-500",
          (sessionState === 'low_light' || sessionState === 'partial_face' || sessionState === 'already_marked') && "bg-yellow-500"
        )} 
      />

      <AnimatePresence mode="wait">
        {sessionState === 'idle' ? (
          <motion.div
            key="start-screen"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, filter: "blur(10px)" }}
            transition={{ duration: 0.4 }}
            className="w-full max-w-2xl z-10"
          >
            <Card className="glass-card shadow-2xl shadow-primary/10 border-white/5">
              <CardContent className="p-8">
                <div className="mb-6 space-y-2 text-center">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
                    Start Attendance Session
                  </h2>
                  <p className="text-gray-500 dark:text-gray-400">
                    Configure the session details before starting the face recognition scanner.
                  </p>
                </div>
                
                <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6 text-left">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Faculty */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Faculty <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('faculty_id')}
                        onChange={handleFacultyChange}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.faculty_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select Faculty...</option>
                        {faculties.map(f => (
                          <option key={f.id} value={f.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">{f.name}</option>
                        ))}
                      </select>
                      {errors.faculty_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.faculty_id.message}</p>}
                    </div>

                    {/* Department */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Department <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('department_id')}
                        onChange={handleDepartmentChange}
                        disabled={!selectedFacultyId}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.department_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select Department...</option>
                        {filteredDepartments.map(d => (
                          <option key={d.id} value={d.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">{d.name}</option>
                        ))}
                      </select>
                      {errors.department_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.department_id.message}</p>}
                    </div>

                    {/* Course Name */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Course Name <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('course_id')}
                        onChange={handleCourseChange}
                        disabled={!selectedDepartmentId}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.course_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select a course...</option>
                        {filteredCourses.map((course) => (
                          <option key={course.id} value={course.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                            {course.title || course.name}
                          </option>
                        ))}
                      </select>
                      {errors.course_id && (
                        <p className="text-xs text-red-500 ml-1 mt-1">{errors.course_id.message}</p>
                      )}
                    </div>

                    {/* Class / Section */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Class / Section <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('class_id')}
                        disabled={!selectedDepartmentId || academiaLoading}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] disabled:opacity-50 disabled:cursor-not-allowed ${errors.class_id ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")' }}
                      >
                        <option value="" disabled className="text-gray-500">
                          {selectedDepartmentId ? 'Select a class...' : 'Select department first'}
                        </option>
                        {filteredClasses.map((cls) => (
                          <option key={cls.id} value={cls.id} className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                            {cls.name}
                          </option>
                        ))}
                      </select>
                      {errors.class_id && <p className="text-xs text-red-500 ml-1 mt-1">{errors.class_id.message}</p>}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Session Type */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Session Type <span className="text-red-500">*</span>
                      </label>
                      <select
                        {...register('session_type')}
                        className={`flex h-12 w-full rounded-xl glass-input px-4 py-2 text-sm text-gray-900 dark:text-gray-100 transition-all appearance-none bg-no-repeat bg-[right_1rem_center] bg-[length:1em_1em] ${errors.session_type ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`}
                        style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`}}
                      >
                        <option value="" disabled className="text-gray-500">Select Session Type...</option>
                        <option value="Lecture" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">Lecture</option>
                        <option value="Lab" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">Lab</option>
                        <option value="Tutorial" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">Tutorial</option>
                      </select>
                      {errors.session_type && (
                        <p className="text-xs text-red-500 ml-1 mt-1">{errors.session_type.message}</p>
                      )}
                    </div>

                    {/* Camera Index */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                        Camera Index <span className="text-red-500">*</span>
                      </label>
                      <Input
                        {...register('camera_index')}
                        type="number"
                        min="0"
                        placeholder="0"
                        className="glass-input"
                        error={errors.camera_index?.message}
                      />
                    </div>
                  </div>

                  {/* Notes */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                      Notes (Optional)
                    </label>
                    <textarea
                      {...register('notes')}
                      placeholder="Add any specific notes for this session..."
                      className="w-full rounded-xl glass-input px-4 py-3 text-sm text-gray-900 dark:text-gray-100 transition-all min-h-[100px] resize-y placeholder:text-gray-400 focus:border-primary focus:ring-primary dark:focus:border-primary-accent dark:focus:ring-primary-accent"
                    />
                  </div>

                  <div className="flex items-center justify-end gap-4 pt-4 border-t border-gray-100 dark:border-white/10">
                    <Button type="button" variant="secondary" onClick={() => reset()}>
                      <X size={18} className="mr-2" />
                      Cancel
                    </Button>
                    <Button type="submit" className="bg-gradient-brand hover:shadow-lg hover:shadow-primary/20 text-white border-0">
                      <Play size={18} className="mr-2 fill-white" />
                      Start Session
                    </Button>
                  </div>
                </form>

              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <ScannerInterface />
        )}
      </AnimatePresence>
    </div>
  );
}
