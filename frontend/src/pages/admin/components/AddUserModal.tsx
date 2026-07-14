import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff } from "lucide-react";
import { motion } from "framer-motion";
import { Modal } from "@/components/ui/Modal";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { useUsersStore } from "@/store/useUsersStore";
import { useHrStore } from "@/store/useHrStore";
import { useAdmissionStore } from "@/store/useAdmissionStore";

// Internal/backend-only roles that should not be assignable to users via the UI
const HIDDEN_ROLES = new Set<string>();

const roleLabelMap: Record<string, string> = {
  SUPER_ADMIN: "Admin",
  ADMISSIONS: "Admission",
  HR: "HR",
  ACADEMIA: "Academia",
  FACULTY: "Faculty",
  TEACHER: "Teacher",
  STUDENT: "Student",
};

const userSchema = z
  .object({
    username: z.string().min(3, "Username must be at least 3 characters"),
    email: z.string().email("Invalid email address"),
    password: z.string().min(6, "Password must be at least 6 characters"),
    role: z.string().min(1, "Role is required"),
    facultyId: z.string().optional(),
    teacherId: z.string().optional(),
    studentId: z.string().optional(),
  })
  .refine(
    (data) => {
      if (
        String(data.role || "").toUpperCase() === "FACULTY" &&
        !data.facultyId
      ) {
        return false;
      }
      return true;
    },
    {
      message: "Faculty is required when role is Faculty",
      path: ["facultyId"],
    },
  )
  .refine(
    (data) => {
      if (
        String(data.role || "").toUpperCase() === "TEACHER" &&
        !data.teacherId
      ) {
        return false;
      }
      return true;
    },
    {
      message: "Teacher ID is required when role is Teacher",
      path: ["teacherId"],
    },
  )
  .refine(
    (data) => {
      if (
        String(data.role || "").toUpperCase() === "STUDENT" &&
        !data.studentId
      ) {
        return false;
      }
      return true;
    },
    {
      message: "Student ID is required when role is Student",
      path: ["studentId"],
    },
  );

type UserFormData = z.infer<typeof userSchema>;

export function AddUserModal() {
  const {
    isModalOpen,
    setModalOpen,
    addUser,
    roles,
    faculties,
    users,
    fetchRolesAndFaculties,
    fetchUsers,
  } = useUsersStore();
  const { teachers, fetchTeachers } = useHrStore();
  const { students, fetchAdmissionData } = useAdmissionStore();
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Build sets of already-linked IDs so we can filter dropdowns
  const linkedTeacherIds = new Set(users.map((u) => u.teacherId).filter(Boolean) as string[]);
  const linkedStudentIds = new Set(users.map((u) => u.studentId).filter(Boolean) as string[]);
  const linkedFacultyIds = new Set(
    users
      .filter((u) => String(u.role).toUpperCase() === "FACULTY")
      .map((u) => u.facultyId)
      .filter(Boolean) as string[]
  );

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
    reset,
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      role: "",
      facultyId: "",
      teacherId: "",
      studentId: "",
    },
  });

  const selectedRole = watch("role");
  const normalizedRole = String(selectedRole || "").toUpperCase();
  const isFacultyRole = normalizedRole === "FACULTY";
  const isTeacherRole = normalizedRole === "TEACHER";
  const isStudentRole = normalizedRole === "STUDENT";
  const visibleRoles = roles.filter((role) => !HIDDEN_ROLES.has(role.name));

  // Reset dependent ID fields to empty string whenever the role changes so the
  // placeholder option is always shown first when a new role is selected.
  useEffect(() => {
    setValue("facultyId", "");
    setValue("teacherId", "");
    setValue("studentId", "");
  }, [normalizedRole, setValue]);

  useEffect(() => {
    if (isModalOpen) {
      fetchRolesAndFaculties();
      fetchTeachers();
      fetchAdmissionData();
      fetchUsers();
      reset();
      setSubmitError(null);
    }
  }, [
    isModalOpen,
    fetchRolesAndFaculties,
    fetchTeachers,
    fetchAdmissionData,
    fetchUsers,
    reset,
  ]);

  const onSubmit = async (data: UserFormData) => {
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      await addUser({
        ...data,
        facultyId: isFacultyRole ? data.facultyId : null,
        teacherId: isTeacherRole ? data.teacherId : undefined,
        studentId: isStudentRole ? data.studentId : undefined,
      });
      setModalOpen(false);
      reset();
    } catch (error: any) {
      setSubmitError(error?.message ?? "Failed to create user");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isModalOpen}
      onClose={() => setModalOpen(false)}
      title="Add New User"
      className="md:max-w-md"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Username
          </label>
          <Input
            placeholder="e.g. John Doe"
            {...register("username")}
            error={errors.username?.message}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Email
          </label>
          <Input
            type="email"
            placeholder="e.g. john@example.com"
            {...register("email")}
            error={errors.email?.message}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Password
          </label>
          <div className="relative">
            <Input
              type={showPassword ? "text" : "password"}
              placeholder="Minimum 6 characters"
              {...register("password")}
              error={errors.password?.message}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1">
            Role <span className="text-primary">*</span>
          </label>
          <Select
            placeholder="Select Role"
            options={visibleRoles.map((r) => ({
              value: r.name,
              label: roleLabelMap[r.name] ?? r.name,
            }))}
            {...register("role")}
            error={errors.role?.message}
          />
        </div>

        {isFacultyRole && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1 mt-4">
              Faculty ID
            </label>
            <Select
              placeholder="Select Faculty"
              options={faculties
                .filter((f) => !linkedFacultyIds.has(String(f.id)))
                .map((f) => ({ value: f.id, label: f.name }))}
              {...register("facultyId")}
              error={errors.facultyId?.message}
            />
          </motion.div>
        )}

        {isTeacherRole && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1 mt-4">
              Teacher ID <span className="text-primary">*</span>
            </label>
            <Select
              placeholder="Select Teacher"
              options={teachers
                .filter((teacher) => !linkedTeacherIds.has(String(teacher.id)))
                .map((teacher) => ({
                  value: teacher.id,
                  label: `${teacher.id} - ${teacher.fullName}`,
                }))}
              {...register("teacherId")}
              error={errors.teacherId?.message}
            />
          </motion.div>
        )}

        {isStudentRole && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5 ml-1 mt-4">
              Student ID <span className="text-primary">*</span>
            </label>
            <Select
              placeholder="Select Student"
              options={students
                .filter((student) =>
                  student.status === "approved" &&
                  !linkedStudentIds.has(String(student.id))
                )
                .map((student) => ({
                  value: student.id,
                  label: `${student.studentNumber} - ${student.fullName}`,
                }))}
              {...register("studentId")}
              error={errors.studentId?.message}
            />
          </motion.div>
        )}

        {submitError && (
          <div className="rounded-lg bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30 px-4 py-3 text-sm text-red-700 dark:text-red-400">
            {submitError}
          </div>
        )}

        <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-100 dark:border-white/5 mt-6">
          <Button
            type="button"
            variant="ghost"
            onClick={() => setModalOpen(false)}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button type="submit" isLoading={isSubmitting}>
            Create User
          </Button>
        </div>
      </form>
    </Modal>
  );
}
