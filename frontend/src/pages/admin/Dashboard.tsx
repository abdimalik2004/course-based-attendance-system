import { useNavigate } from "react-router-dom";
import {
  Users,
  GraduationCap,
  Presentation,
  Activity,
  FileText,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/Card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import dashboardService from "@/services/dashboardService";

const RECENT_ACTIVITY = [
  {
    id: 1,
    user: "John Doe",
    action: "System Login",
    time: "2 mins ago",
    status: "Success",
  },
  {
    id: 2,
    user: "Sarah Smith",
    action: "Updated Schedule",
    time: "15 mins ago",
    status: "Success",
  },
  {
    id: 3,
    user: "Admin",
    action: "Database Backup",
    time: "1 hour ago",
    status: "Pending",
  },
  {
    id: 4,
    user: "Dr. Jane",
    action: "Failed Auth",
    time: "3 hours ago",
    status: "Failed",
  },
  {
    id: 5,
    user: "Michael B.",
    action: "Created Course",
    time: "5 hours ago",
    status: "Success",
  },
];

export default function AdminDashboard() {
  const navigate = useNavigate();

  const { data, isLoading, error } = useQuery({
    queryKey: ["adminOverview"],
    queryFn: () => dashboardService.adminOverview(),
    staleTime: 1000 * 60 * 2,
  });

  const metrics = [
    {
      id: 1,
      title: "Total Students",
      value: isLoading ? "—" : (data?.totalStudents ?? "—"),
      icon: GraduationCap,
      color: "text-green-500",
      path: "/admin/users",
    },
    {
      id: 2,
      title: "Total Teachers",
      value: isLoading ? "—" : (data?.totalTeachers ?? "—"),
      icon: Presentation,
      color: "text-purple-500",
      path: "/admin/users",
    },
    {
      id: 3,
      title: "Total Faculties",
      value: isLoading ? "—" : (data?.totalFaculties ?? "—"),
      icon: Users,
      color: "text-blue-500",
      path: "/admin/faculties",
    },
    {
      id: 4,
      title: "Attendance Rate",
      value: isLoading
        ? "—"
        : data?.attendanceRate
          ? `${data.attendanceRate}%`
          : "—",
      icon: Activity,
      color: "text-primary-accent",
      path: "/admin/attendance",
    },
  ];

  if (error) console.error("Failed to load admin overview", error);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
          Overview
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Welcome back, Admin. Here's what's happening today.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <Card
            key={metric.id}
            className="relative overflow-hidden group cursor-pointer hover:border-primary/50 hover:shadow-md transition-all duration-300"
            onClick={() => navigate(metric.path)}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-brand opacity-5 rounded-full blur-2xl -mr-10 -mt-10 group-hover:opacity-10 transition-opacity" />
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
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Recent Activity
            </h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {RECENT_ACTIVITY.map((activity) => (
                  <TableRow key={activity.id}>
                    <TableCell className="font-medium">
                      {activity.user}
                    </TableCell>
                    <TableCell>{activity.action}</TableCell>
                    <TableCell>
                      <span
                        className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${activity.status === "Success" ? "bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400" : activity.status === "Failed" ? "bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-400" : "bg-yellow-100 text-yellow-700 dark:bg-yellow-500/10 dark:text-yellow-400"}`}
                      >
                        {activity.status}
                      </span>
                    </TableCell>
                    <TableCell className="text-right text-gray-500 dark:text-gray-400">
                      {activity.time}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>

        <Card className="md:col-span-3 lg:col-span-2">
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Quick Actions
            </h3>
            <div className="space-y-4">
              <button
                onClick={() => navigate("/admin/users")}
                className="w-full flex items-center justify-between p-4 rounded-xl border border-gray-200 dark:border-white/10 hover:border-primary dark:hover:border-primary bg-gray-50 dark:bg-white/5 transition-colors group"
              >
                <span className="font-medium text-gray-700 dark:text-gray-300 group-hover:text-primary dark:group-hover:text-primary-accent transition-colors">
                  Add New User
                </span>
                <Users
                  size={18}
                  className="text-gray-400 group-hover:text-primary transition-colors"
                />
              </button>

              <button
                onClick={() => navigate("/admin/reports")}
                className="w-full flex items-center justify-between p-4 rounded-xl border border-gray-200 dark:border-white/10 hover:border-primary dark:hover:border-primary bg-gray-50 dark:bg-white/5 transition-colors group"
              >
                <span className="font-medium text-gray-700 dark:text-gray-300 group-hover:text-primary dark:group-hover:text-primary-accent transition-colors">
                  Generate Report
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
