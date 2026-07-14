import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuthStore } from "@/store/useAuthStore";
import {
  Users,
  GraduationCap,
  Presentation,
  Activity,
  FileText,
  RefreshCw,
  Wifi,
  WifiOff,
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
import activityService, {
  formatTimeAgo,
  formatTime,
  formatFullDateTime,
} from "@/services/activityService";
import type { ActivityLog } from "@/services/activityService";

function AdminDashboard() {
  const navigate = useNavigate();
  const username = useAuthStore((s) => s.user?.username ?? "Admin");
  const [recentActivities, setRecentActivities] = useState<ActivityLog[]>([]);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["adminOverview"],
    queryFn: () => dashboardService.adminOverview(),
    staleTime: 1000 * 60 * 2,
  });

  const {
    data: initialActivities,
    isLoading: activityLoading,
    refetch: refetchActivity,
    isFetching: activityFetching,
  } = useQuery({
    queryKey: ["recentActivity"],
    queryFn: async () => {
      return await activityService.getRecentActivities(20, 2);
    },
    staleTime: 1000 * 30,
    refetchInterval: 1000 * 30, // Fallback polling every 30s
  });

  // Initialize activities from query result
  useEffect(() => {
    if (initialActivities) {
      setRecentActivities(initialActivities);
    }
  }, [initialActivities]);

  // WebSocket setup for real-time updates
  useEffect(() => {
    // Check if we have admin privileges and can connect to WebSocket
    // In a real app, you might check user roles here
    const handleNewActivity = (activity: ActivityLog) => {
      setRecentActivities((prev) => {
        // Prevent duplicates
        if (prev.some((a) => a.id === activity.id)) {
          return prev;
        }
        // Add to beginning and keep max 30
        return [activity, ...prev].slice(0, 30);
      });
    };

    const handleError = (_error: Event) => {
      setIsWebSocketConnected(false);
    };

    const handleClose = (_event: CloseEvent) => {
      setIsWebSocketConnected(false);
    };

    // Attempt to connect to WebSocket
    try {
      activityService.connectWebSocket(
        handleNewActivity,
        handleError,
        handleClose,
      );
      setIsWebSocketConnected(activityService.isWebSocketConnected());

      // Check connection status periodically
      const statusCheckInterval = setInterval(() => {
        setIsWebSocketConnected(activityService.isWebSocketConnected());
      }, 1000);

      return () => {
        clearInterval(statusCheckInterval);
        // Don't disconnect on unmount in case other components use it
      };
    } catch {
      // WebSocket setup failed — UI will reflect disconnected state
    }
  }, []);

  const metrics = [
    {
      id: 1,
      title: "Total Students",
      value: isLoading ? "—" : (data?.totalStudents ?? "—"),
      icon: GraduationCap,
      color: "text-green-500",
      path: "/admin/students",
    },
    {
      id: 2,
      title: "Total Teachers",
      value: isLoading ? "—" : (data?.totalTeachers ?? "—"),
      icon: Presentation,
      color: "text-purple-500",
      path: "/admin/teachers",
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
      path: "/admin/attendance-list",
    },
  ];

  // error is surfaced via the UI query error state below

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
          Overview
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Welcome back, <span className="font-medium text-gray-700 dark:text-gray-300">{username}</span>. Here's what's happening today.
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
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Recent Activity
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Recent activity · Live updates
                  {isWebSocketConnected && (
                    <span className="inline-flex items-center ml-2">
                      <Wifi
                        size={12}
                        className="text-green-500 mr-1 animate-pulse"
                      />
                      <span className="text-green-500">Connected</span>
                    </span>
                  )}
                  {!isWebSocketConnected && (
                    <span className="inline-flex items-center ml-2">
                      <WifiOff size={12} className="text-orange-500 mr-1" />
                      <span className="text-orange-500">Polling (30s)</span>
                    </span>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Link
                  to="/admin/activity-log"
                  className="text-xs text-primary hover:underline whitespace-nowrap"
                >
                  View all →
                </Link>
                <button
                  onClick={() => refetchActivity()}
                  disabled={activityFetching}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-white/10 transition-colors disabled:opacity-40"
                  title="Refresh activity"
                >
                  <RefreshCw
                    size={15}
                    className={`text-gray-400 ${activityFetching ? "animate-spin" : ""}`}
                  />
                </button>
              </div>
            </div>
            <div className="overflow-auto max-h-[340px] custom-scrollbar">
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
                  {activityLoading && !recentActivities.length ? (
                    Array.from({ length: 4 }).map((_, i) => (
                      <TableRow key={i}>
                        {Array.from({ length: 4 }).map((__, j) => (
                          <TableCell key={j}>
                            <div className="h-4 w-20 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : !recentActivities || recentActivities.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={4}
                        className="text-center text-gray-400 py-6"
                      >
                        No recent activity found.
                      </TableCell>
                    </TableRow>
                  ) : (
                    recentActivities.map((activity) => (
                      <TableRow
                        key={activity.id}
                        className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors"
                      >
                        <TableCell className="font-medium text-gray-900 dark:text-gray-100 whitespace-nowrap">
                          {activity.username}
                        </TableCell>
                        <TableCell
                          className="text-gray-600 dark:text-gray-400 max-w-[220px] truncate"
                          title={activity.action}
                        >
                          {activity.action}
                        </TableCell>
                        <TableCell>
                          <span
                            className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                              activity.status === "Success"
                                ? "bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400"
                                : activity.status === "Failed"
                                  ? "bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-400"
                                  : "bg-yellow-100 text-yellow-700 dark:bg-yellow-500/10 dark:text-yellow-400"
                            }`}
                          >
                            {activity.status}
                          </span>
                        </TableCell>
                        <TableCell
                          className="text-right text-gray-500 dark:text-gray-400 whitespace-nowrap"
                          title={formatFullDateTime(activity.created_at)}
                        >
                          <span className="text-xs">
                            {formatTimeAgo(activity.created_at)}
                          </span>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
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

export default AdminDashboard;
