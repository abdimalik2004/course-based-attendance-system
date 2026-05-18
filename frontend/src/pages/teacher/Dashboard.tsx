import { motion } from 'framer-motion';
import { 
  BookOpen, 
  Users, 
  Calendar, 
  Clock, 
  MoreVertical, 
  Play, 
  List, 
  ChevronRight,
  CheckCircle2
} from 'lucide-react';
import { 
  PieChart, 
  Pie, 
  Cell, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import attendanceService from "@/services/attendanceService";

export default function TeacherDashboard() {
  const navigate = useNavigate();
  const { data, isLoading, error } = useQuery({
    queryKey: ["teacherSessions"],
    queryFn: () => attendanceService.listSessions({ skip: 0, limit: 200 }),
  });

  const sessions = data?.items ?? data ?? [];

  const sessionCards = useMemo(() => sessions.slice(0, 5).map((session: any) => ({
    id: session.id,
    time: `${session.start_time ?? 'TBA'} - ${session.end_time ?? 'TBA'}`,
    course: session.course_title ?? `Course ${session.course_id ?? session.id}`,
    room: session.room_name ?? session.location ?? 'TBA',
    status: String(session.status ?? 'UPCOMING'),
  })), [sessions]);

  const statsData = useMemo(() => {
    const completed = sessions.filter((session: any) => String(session.status ?? '').toUpperCase() === 'ENDED').length;
    const active = sessions.filter((session: any) => String(session.status ?? '').toUpperCase() === 'ACTIVE').length;
    return [
      { title: "Today's Classes", value: String(sessionCards.length), subtitle: `${active} active`, icon: Clock, color: "text-blue-500", bg: "bg-blue-500/10" },
      { title: "My Courses", value: String(new Set(sessions.map((session: any) => session.course_id)).size), subtitle: "From live sessions", icon: BookOpen, color: "text-purple-500", bg: "bg-purple-500/10" },
      { title: "Attendance Sessions", value: String(completed), subtitle: "Ended sessions", icon: Users, color: "text-green-500", bg: "bg-green-500/10" },
      { title: "Upcoming Classes", value: String(active), subtitle: "Need attention", icon: Calendar, color: "text-orange-500", bg: "bg-orange-500/10" },
    ];
  }, [sessions, sessionCards.length]);

  const attendancePieData = useMemo(() => {
    const counts = sessions.reduce((acc: Record<string, number>, session: any) => {
      const key = String(session.status ?? 'UNKNOWN').toLowerCase();
      acc[key] = (acc[key] ?? 0) + 1;
      return acc;
    }, {});
    return [
      { name: 'Active', value: counts.active ?? 0, color: '#10B981' },
      { name: 'Ended', value: counts.ended ?? 0, color: '#EF4444' },
      { name: 'Upcoming', value: counts.upcoming ?? 0, color: '#F59E0B' },
    ].filter((entry) => entry.value > 0);
  }, [sessions]);

  const attendanceLineData = useMemo(() => {
    const byDay = new Map<string, number>();
    sessions.forEach((session: any) => {
      const day = new Date(session.start_time ?? session.created_at ?? Date.now()).toLocaleDateString('en-US', { weekday: 'short' });
      byDay.set(day, (byDay.get(day) ?? 0) + 1);
    });
    return Array.from(byDay.entries()).map(([day, present]) => ({ day, present }));
  }, [sessions]);

  const recentActivity = useMemo(() => sessionCards.slice(0, 3).map((session, index) => ({
    id: session.id,
    title: index === 0 ? 'Live session' : index === 1 ? 'Scheduled class' : 'Course session',
    desc: `${session.course} • ${session.time}`,
    time: index === 0 ? 'Recent' : `${index + 1} sessions back`,
    icon: index === 0 ? CheckCircle2 : index === 1 ? BookOpen : Calendar,
    color: index === 0 ? 'text-green-500' : index === 1 ? 'text-blue-500' : 'text-purple-500',
  })), [sessionCards]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard Overview</h2>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          Failed to load live teacher sessions.
        </div>
      ) : null}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
        {statsData.map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.1 }}
            className="glass-card p-6 rounded-2xl hover:shadow-lg hover:shadow-primary/5 transition-all duration-300 group"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{stat.title}</p>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mt-2 group-hover:text-primary transition-colors">{stat.value}</h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{stat.subtitle}</p>
              </div>
              <div className={`p-3 rounded-xl ${stat.bg} ${stat.color}`}>
                <stat.icon size={24} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content Area (Left 2 columns) */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Charts Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pie Chart */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="glass-card p-6 rounded-2xl"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Overall Attendance</h3>
              <div className="h-[250px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={attendancePieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {attendancePieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', border: 'none', borderRadius: '12px', color: '#fff' }}
                      itemStyle={{ color: '#fff' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-4 mt-2">
                {attendancePieData.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-xs text-gray-500 dark:text-gray-400">{item.name}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Line Chart */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="glass-card p-6 rounded-2xl"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Weekly Trend</h3>
              <div className="h-[250px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={attendanceLineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                    <XAxis dataKey="day" stroke="#6B7280" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#6B7280" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', border: 'none', borderRadius: '12px', color: '#fff' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="present" 
                      stroke="#3B82F6" 
                      strokeWidth={3}
                      dot={{ r: 4, fill: '#3B82F6', strokeWidth: 2, stroke: '#fff' }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          </div>

          {/* Today's Schedule Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="glass-card rounded-2xl overflow-hidden"
          >
            <div className="p-6 border-b border-gray-200 dark:border-white/10 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Today's Schedule</h3>
              <button onClick={() => navigate('/teacher/schedule')} className="text-sm text-primary hover:underline">View All</button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-gray-500 dark:text-gray-400">
                <thead className="bg-gray-50 dark:bg-white/5 text-xs uppercase text-gray-700 dark:text-gray-300">
                  <tr>
                    <th className="px-6 py-4 font-medium">Time</th>
                    <th className="px-6 py-4 font-medium">Course</th>
                    <th className="px-6 py-4 font-medium">Class / Room</th>
                    <th className="px-6 py-4 font-medium">Status</th>
                    <th className="px-6 py-4 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-white/10">
                  {isLoading ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400">Loading sessions...</td>
                    </tr>
                  ) : sessionCards.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400">No live sessions found in the database.</td>
                    </tr>
                  ) : sessionCards.map((item) => (
                    <tr key={item.id} className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group">
                      <td className="px-6 py-4 whitespace-nowrap">{item.time}</td>
                      <td className="px-6 py-4 font-medium text-gray-900 dark:text-white">{item.course}</td>
                      <td className="px-6 py-4">{item.room}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          item.status === 'ENDED' ? 'bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-400' :
                          item.status === 'ACTIVE' ? 'bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-400 animate-pulse' :
                          'bg-gray-100 text-gray-800 dark:bg-white/10 dark:text-gray-400'
                        }`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
                          <MoreVertical size={18} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>

        {/* Right Panel */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            className="glass-card p-6 rounded-2xl space-y-4"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Quick Actions</h3>
            
            <button 
              onClick={() => navigate('/teacher/attendance')}
              className="w-full group flex items-center justify-between p-4 rounded-xl bg-gradient-brand text-white hover:shadow-lg hover:shadow-primary/20 transition-all duration-300"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm group-hover:scale-110 transition-transform">
                  <Play size={20} className="fill-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold">Start Attendance</p>
                  <p className="text-xs text-white/80">Begin a new session</p>
                </div>
              </div>
              <ChevronRight size={20} className="text-white/70 group-hover:text-white group-hover:translate-x-1 transition-all" />
            </button>

            <button 
              onClick={() => navigate('/teacher/schedule')}
              className="w-full group flex items-center justify-between p-4 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-white/5 dark:hover:bg-white/10 border border-gray-200 dark:border-white/10 transition-all duration-300"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 bg-gray-200 dark:bg-white/10 rounded-lg text-gray-600 dark:text-gray-300 group-hover:scale-110 transition-transform">
                  <Calendar size={20} />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-gray-900 dark:text-white">View Schedule</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Check upcoming classes</p>
                </div>
              </div>
              <ChevronRight size={20} className="text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white group-hover:translate-x-1 transition-all" />
            </button>

            <button 
              onClick={() => navigate('/teacher/attendance')}
              className="w-full group flex items-center justify-between p-4 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-white/5 dark:hover:bg-white/10 border border-gray-200 dark:border-white/10 transition-all duration-300"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 bg-gray-200 dark:bg-white/10 rounded-lg text-gray-600 dark:text-gray-300 group-hover:scale-110 transition-transform">
                  <List size={20} />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-gray-900 dark:text-white">Attendance Records</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">View past sessions</p>
                </div>
              </div>
              <ChevronRight size={20} className="text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white group-hover:translate-x-1 transition-all" />
            </button>
          </motion.div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="glass-card p-6 rounded-2xl"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Recent Activity</h3>
            <div className="space-y-6">
              {recentActivity.map((activity, idx) => (
                <div key={activity.id} className="relative pl-6">
                  {/* Timeline Line */}
                  {idx !== recentActivity.length - 1 && (
                    <div className="absolute left-2.5 top-8 bottom-[-24px] w-[1px] bg-gray-200 dark:bg-white/10" />
                  )}
                  {/* Timeline Dot */}
                  <div className={`absolute left-0 top-1 w-5 h-5 rounded-full ${activity.color} bg-white dark:bg-dark-card flex items-center justify-center border-2 border-current shadow-sm`}>
                    <activity.icon size={10} className="text-current" />
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white">{activity.title}</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{activity.desc}</p>
                    <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2">{activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
