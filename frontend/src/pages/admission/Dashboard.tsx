import { useNavigate } from "react-router-dom";
import {
  Users,
  GraduationCap,
  FileText,
  BadgeAlert,
  ClipboardCheck,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { StatCard } from "@/components/ui/StatCard";
import { Card, CardContent } from "@/components/ui/Card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import admissionService from "@/services/admissionService";
import { useAdmissionStore } from "@/store/useAdmissionStore";

export default function Dashboard() {
  const navigate = useNavigate();
  const { dashboardStats } = useAdmissionStore();

  const recentStudentsQuery = useQuery({
    queryKey: ["recentAdmissions"],
    queryFn: () => admissionService.listRecentStudents({ skip: 0, limit: 5 }),
  });

  const metrics = [
    {
      id: 1,
      title: "Total Students",
      value: dashboardStats.totalStudents,
      icon: GraduationCap,
      color: "text-green-500",
      path: "/admission/students",
    },
    {
      id: 2,
      title: "New Admissions",
      value: dashboardStats.newAdmissions,
      icon: Users,
      color: "text-blue-500",
      path: "/academia/faculties",
    },
    {
      id: 3,
      title: "Pending Admissions",
      value: dashboardStats.pendingApplications,
      icon: BadgeAlert,
      color: "text-amber-500",
      path: "/admission/students?status=pending",
    },
    {
      id: 4,
      title: "Rejected Applications",
      value: dashboardStats.rejectedApplications,
      icon: ClipboardCheck,
      color: "text-rose-500",
      path: "/admission/students?status=rejected",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
          Admission Dashboard
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Overview of student admissions and registered enrollments.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <Card
            key={metric.id}
            className="relative overflow-hidden group cursor-pointer hover:border-primary/50 hover:shadow-md transition-all duration-300"
            onClick={() => navigate(metric.path)}
          >
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                    {metric.title}
                  </p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                    {metric.value}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-gray-200 dark:bg-white/5 border border-gray-300 dark:border-white/10">
                  <metric.icon size={20} className={metric.color} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-7">
        <Card className="md:col-span-4 lg:col-span-5">
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Recent Admissions
            </h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Student</TableHead>
                  <TableHead>Student Number</TableHead>
                  <TableHead>Faculty ID</TableHead>
                  <TableHead className="text-right">Department ID</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentStudentsQuery.isLoading ? (
                  <TableRow>
                    <TableCell
                      colSpan={4}
                      className="h-32 text-center text-gray-500"
                    >
                      Loading latest student records...
                    </TableCell>
                  </TableRow>
                ) : recentStudentsQuery.data?.items.length ? (
                  recentStudentsQuery.data.items.map((student: any) => (
                    <TableRow key={student.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {student.full_name}
                      </TableCell>
                      <TableCell>{student.student_number}</TableCell>
                      <TableCell>{student.faculty_id}</TableCell>
                      <TableCell className="text-right text-gray-500 dark:text-gray-400">
                        {student.department_id}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={4}
                      className="h-32 text-center text-gray-500"
                    >
                      No student records found in the database.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        <Card className="md:col-span-3 lg:col-span-2">
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Quick Actions
            </h3>
            <div className="space-y-4">
              <button
                onClick={() => navigate("/admission/students")}
                className="w-full flex items-center justify-between p-4 rounded-xl border border-gray-200 dark:border-white/10 hover:border-primary dark:hover:border-primary bg-gray-50 dark:bg-white/5 transition-colors group"
              >
                <span className="font-medium text-gray-700 dark:text-gray-300 group-hover:text-primary dark:group-hover:text-primary-accent transition-colors">
                  Manage Students
                </span>
                <Users
                  size={18}
                  className="text-gray-400 group-hover:text-primary transition-colors"
                />
              </button>

              <button
                onClick={() => navigate("/academia/faculties")}
                className="w-full flex items-center justify-between p-4 rounded-xl border border-gray-200 dark:border-white/10 hover:border-primary dark:hover:border-primary bg-gray-50 dark:bg-white/5 transition-colors group"
              >
                <span className="font-medium text-gray-700 dark:text-gray-300 group-hover:text-primary dark:group-hover:text-primary-accent transition-colors">
                  Review Faculty Structure
                </span>
                <FileText
                  size={18}
                  className="text-gray-400 group-hover:text-primary transition-colors"
                />
              </button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
