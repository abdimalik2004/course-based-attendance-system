import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { X, Save, Building, GraduationCap, User } from "lucide-react";
import { useHrStore } from "@/store/useHrStore";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";

const teacherSchema = z.object({
  fullName: z.string().min(2, "Full name is required"),
  facultyId: z.string().min(1, "Faculty is required"),
  departmentId: z.string().min(1, "Department is required"),
});

type TeacherFormValues = z.infer<typeof teacherSchema>;

interface AddTeacherModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AddTeacherModal({ isOpen, onClose }: AddTeacherModalProps) {
  const {
    faculties,
    departments,
    fetchFaculties,
    fetchDepartments,
    addTeacher,
  } = useHrStore();

  const {
    register,
    handleSubmit,
    watch,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<TeacherFormValues>({
    resolver: zodResolver(teacherSchema),
    defaultValues: {
      fullName: "",
      facultyId: "",
      departmentId: "",
    },
  });

  const selectedFacultyId = watch("facultyId");
  const facultyOptions = faculties.map((faculty) => ({
    value: faculty.id,
    label: faculty.name,
  }));
  const departmentOptions = departments
    .filter((department) => department.facultyId === selectedFacultyId)
    .map((department) => ({ value: department.id, label: department.name }));

  useEffect(() => {
    if (isOpen) {
      fetchFaculties();
      fetchDepartments();
    } else {
      reset();
    }
  }, [isOpen, fetchFaculties, fetchDepartments, reset]);

  const onSubmit = async (data: TeacherFormValues) => {
    await addTeacher({
      fullName: data.fullName,
      facultyId: data.facultyId,
      departmentId: data.departmentId,
    });
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-[60] bg-black/50 backdrop-blur-sm"
          />

          {/* Modal */}
          <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 sm:p-6 pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="w-full max-w-lg glass-card rounded-2xl shadow-2xl pointer-events-auto overflow-hidden flex flex-col max-h-full"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-white/10 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-brand" />
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                  Add New Teacher
                </h2>
                <button
                  onClick={onClose}
                  className="rounded-full p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-white/10 transition-colors"
                >
                  <X size={20} />
                </button>
              </div>

              {/* Form Body */}
              <div className="p-6 overflow-y-auto custom-scrollbar">
                <form
                  id="add-teacher-form"
                  onSubmit={handleSubmit(onSubmit)}
                  className="space-y-6"
                >
                  {/* Full Name */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                      Full Name
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500">
                        <User size={18} />
                      </div>
                      <Input
                        {...register("fullName")}
                        type="text"
                        placeholder="e.g. Dr. Ahmed Ali"
                        className="pl-11"
                        error={errors.fullName?.message}
                      />
                    </div>
                  </div>

                  {/* Faculty */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                      Faculty
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500">
                        <Building size={18} />
                      </div>
                      <div className="pl-0">
                        <Select
                          {...register("facultyId")}
                          options={facultyOptions}
                          placeholder="Select Faculty"
                          error={errors.facultyId?.message}
                          className="pl-11"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Department (Dependent Dropdown) */}
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                      Department
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500">
                        <GraduationCap size={18} />
                      </div>
                      <div className="pl-0">
                        <Select
                          {...register("departmentId")}
                          options={departmentOptions}
                          placeholder={
                            selectedFacultyId
                              ? "Select Department"
                              : "Select Faculty First"
                          }
                          error={errors.departmentId?.message}
                          disabled={!selectedFacultyId}
                          className="pl-11"
                        />
                      </div>
                    </div>
                  </div>
                </form>
              </div>

              {/* Footer */}
              <div className="p-6 border-t border-gray-200 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 flex justify-end gap-3">
                <Button variant="secondary" onClick={onClose} type="button">
                  Cancel
                </Button>
                <Button
                  type="submit"
                  form="add-teacher-form"
                  isLoading={isSubmitting}
                  className="min-w-[120px]"
                >
                  <Save className="mr-2 h-4 w-4" />
                  Save
                </Button>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
