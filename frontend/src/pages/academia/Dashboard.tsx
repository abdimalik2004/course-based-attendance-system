import {
  Building2,
  Network,
  BookOpen,
  Users,
  GraduationCap,
  ArrowUpRight,
  TrendingUp,
  Plus,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/academia/Table";
import { Badge } from "@/components/ui/Badge";
import { useAcademiaStore } from "@/store/useAcademiaStore";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { CourseModal } from "./components/CourseModal";
import { ClassModal } from "./components/ClassModal";

export default function AcademiaDashboard() {
  const {
    faculties,
    departments,
    courses,
    classes,
    classAssignments,
    fetchData,
    openModal,
    isLoading,
    error,
  } = useAcademiaStore();
  const navigate = useNavigate();

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const recentClassAssignments = classAssignments
    .slice(0, 5)
    .map((assignment) => {
      const matchedClass = classes.find((cls) => cls.id === assignment.classId);
      const matchedCourse = courses.find(
        (course) => course.id === assignment.courseId,
      );

      return {
        id: assignment.id,
        code: matchedCourse?.code ?? "N/A",
        name: matchedClass?.name ?? "Unassigned Class",
        instructor: matchedCourse ? matchedCourse.title : "Unassigned",
        capacity: matchedClass ? `Year ${matchedClass.year}` : "N/A",
        status: matchedClass ? "Assigned" : "Pending",
      };
    });

  const METRICS = [
    {
      id: 1,
      title: "Total Faculties",
      value: faculties.length,
      icon: Building2,
      color: "text-blue-500",
      trend: "Growing",
      path: "/academia/faculties",
    },
    {
      id: 2,
      title: "Active Departments",
      value: departments.length,
      icon: Network,
      color: "text-emerald-500",
      trend: "Stable",
      path: "/academia/departments",
    },
    {
      id: 3,
      title: "Live Courses",
      value: courses.length,
      icon: BookOpen,
      color: "text-indigo-500",
      trend: "Active",
      path: "/academia/courses",
    },
    {
      id: 4,
      title: "Enrolled Classes",
      value: classes.length,
      icon: Users,
      color: "text-purple-500",
      trend: "In-Session",
      path: "/academia/classes",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
          <GraduationCap className="text-primary" size={32} />
          Academia Overview
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Real-time metrics and academic structure status across all faculties.
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center rounded-2xl border border-gray-200 dark:border-white/10 bg-white/70 dark:bg-white/5 p-8 text-sm text-gray-500 dark:text-gray-400">
          Loading real academic structure data...
        </div>
      ) : error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-6 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      ) : null}

      {/* Metrics Row */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {METRICS.map((metric) => (
          <Card
            key={metric.id}
            className="relative overflow-hidden group glass-card hover:shadow-xl hover:border-primary/50 cursor-pointer transition-all duration-300 border-gray-200 dark:border-white/10 hover:-translate-y-1"
            onClick={() => navigate(metric.path)}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary opacity-5 rounded-full blur-[40px] -mr-10 -mt-10 group-hover:opacity-20 transition-opacity duration-500" />
            <CardContent className="p-6 relative z-10">
              <div className="flex items-start justify-between mb-4">
                <div className="p-3 rounded-2xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 group-hover:scale-110 transition-transform duration-300">
                  <metric.icon size={24} className={metric.color} />
                </div>
                <Badge
                  variant="default"
                  className="bg-gray-50 dark:bg-white/5 border-gray-200 dark:border-white/10 text-xs text-gray-500 dark:text-gray-400"
                >
                  <TrendingUp size={12} className="mr-1 inline" />
                  {metric.trend}
                </Badge>
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white tracking-tight mb-1">
                  {metric.value}
                </h3>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  {metric.title}
                </p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content Area */}
      <div className="grid gap-6 md:grid-cols-7">
        {/* Table Area (Left 5 columns) */}
        <Card className="md:col-span-4 lg:col-span-5 glass-card border-gray-200 dark:border-white/10">
          <div className="p-6 border-b border-gray-200 dark:border-white/10 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Live Classes Overview
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Current semester active course sessions
              </p>
            </div>
            <button
              onClick={() => navigate("/academia/classes")}
              className="text-sm text-primary flex items-center gap-1 hover:underline"
            >
              View all <ArrowUpRight size={16} />
            </button>
          </div>
          <div className="p-0 overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-gray-200 dark:border-white/10">
                  <TableHead className="text-gray-900 dark:text-gray-100 pl-6">
                    Course Code
                  </TableHead>
                  <TableHead className="text-gray-900 dark:text-gray-100">
                    Section Name
                  </TableHead>
                  <TableHead className="text-gray-900 dark:text-gray-100">
                    Instructor
                  </TableHead>
                  <TableHead className="text-gray-900 dark:text-gray-100">
                    Capacity
                  </TableHead>
                  <TableHead className="text-gray-900 dark:text-gray-100 pr-6 text-right">
                    Status
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentClassAssignments.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="py-12 text-center text-sm text-gray-500 dark:text-gray-400"
                    >
                      No academic class assignments found in the database.
                    </TableCell>
                  </TableRow>
                ) : (
                  recentClassAssignments.map((cls) => (
                    <TableRow
                      key={cls.id}
                      className="border-gray-100 dark:border-white/5 hover:bg-gray-50 dark:hover:bg-white/5"
                    >
                      <TableCell className="font-semibold text-gray-900 dark:text-white pl-6">
                        {cls.code}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {cls.name}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {cls.instructor}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300 font-medium">
                        {cls.capacity}
                      </TableCell>
                      <TableCell className="pr-6 text-right">
                        <Badge variant="success">{cls.status}</Badge>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* Quick Actions (Right 2 columns) */}
        <div className="md:col-span-3 lg:col-span-2 space-y-6">
          <Card className="glass-card border-gray-200 dark:border-white/10 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-full h-1 bg-gradient-brand" />
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Academic Actions
              </h3>
              <div className="space-y-3">
                <button
                  onClick={() => openModal("course", "create")}
                  className="w-full flex items-center justify-between p-4 rounded-xl border border-gray-200 dark:border-white/10 hover:border-primary dark:hover:border-primary bg-white dark:bg-white/5 hover:bg-gray-50 dark:hover:bg-white/10 shadow-sm transition-all group"
                >
                  <span className="font-medium text-gray-700 dark:text-gray-200 group-hover:text-primary transition-colors flex gap-2 items-center">
                    <Plus size={16} /> New Course
                  </span>
                  <BookOpen
                    size={18}
                    className="text-gray-400 group-hover:text-primary transition-colors"
                  />
                </button>

                <button
                  onClick={() => openModal("class", "create")}
                  className="w-full flex items-center justify-between p-4 rounded-xl border border-gray-200 dark:border-white/10 hover:border-emerald-500 dark:hover:border-emerald-500 bg-white dark:bg-white/5 hover:bg-gray-50 dark:hover:bg-white/10 shadow-sm transition-all group"
                >
                  <span className="font-medium text-gray-700 dark:text-gray-200 group-hover:text-emerald-500 transition-colors flex gap-2 items-center">
                    <Plus size={16} /> New Class
                  </span>
                  <Users
                    size={18}
                    className="text-gray-400 group-hover:text-emerald-500 transition-colors"
                  />
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>
      <CourseModal />
      <ClassModal />
    </div>
  );
}
