import { useEffect } from 'react';
import { Users, Building2, BookOpen, GraduationCap, Calendar } from 'lucide-react';
import { StatCard } from '@/components/ui/StatCard';
import { useFacultyStore } from '@/store/useFacultyStore';

export default function FacultyDashboard() {
  const { stats, fetchData, isLoading } = useFacultyStore();

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Building2 className="text-primary" size={28} />
            Faculty Dashboard
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Overview of your department's academic activities and resources.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          title="Total Students"
          value={stats.totalStudents}
          icon={Users}
          iconColor="primary"
        />
        <StatCard
          title="Total Teachers"
          value={stats.totalTeachers}
          icon={GraduationCap}
          iconColor="success"
        />
        <StatCard
          title="Departments"
          value={stats.totalDepartments}
          icon={Building2}
          iconColor="warning"
        />
        <StatCard
          title="Classes"
          value={stats.totalClasses}
          icon={Calendar}
          iconColor="success"
        />
        <StatCard
          title="Courses"
          value={stats.totalCourses}
          icon={BookOpen}
          iconColor="danger"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <div className="glass-panel p-6 rounded-2xl border border-white/10 shadow-lg relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-6 bg-blue-500 rounded-full"></div>
            Quick Insights
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
              <span className="text-gray-400">Average Class Size</span>
              <span className="text-lg font-semibold text-white">39 Students</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
              <span className="text-gray-400">Active Teachers</span>
              <span className="text-lg font-semibold text-white">42 / 45</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
              <span className="text-gray-400">Unassigned Courses</span>
              <span className="text-lg font-semibold text-amber-400">3 Courses</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
              <span className="text-gray-400">Schedule Completion</span>
              <span className="text-lg font-semibold text-emerald-400">92%</span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl border border-white/10 shadow-lg relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-bl from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-6 bg-purple-500 rounded-full"></div>
            Recent Activities
          </h2>
          <div className="space-y-4">
            {[
              { text: "Dr. Smith assigned to Data Structures", time: "2 hours ago", type: "assignment" },
              { text: "Updated schedule for Algorithms CS301", time: "5 hours ago", type: "schedule" },
              { text: "New course 'Quantum Computing' added", time: "1 day ago", type: "course" },
              { text: "Dr. Johnson removed from Database Systems", time: "2 days ago", type: "assignment" },
            ].map((activity, i) => (
              <div key={i} className="flex items-start gap-4 p-3 rounded-lg hover:bg-white/5 transition-colors">
                <div className="mt-1">
                  {activity.type === 'assignment' ? (
                    <div className="w-2 h-2 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)]"></div>
                  ) : activity.type === 'schedule' ? (
                    <div className="w-2 h-2 rounded-full bg-purple-400 shadow-[0_0_8px_rgba(192,132,252,0.8)]"></div>
                  ) : (
                    <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"></div>
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-200">{activity.text}</p>
                  <p className="text-xs text-gray-500 mt-0.5">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
