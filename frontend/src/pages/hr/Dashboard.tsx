import { useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Users, UserCheck, UserX, UserMinus, Activity } from 'lucide-react';
import { useHrStore } from '@/store/useHrStore';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell
} from 'recharts';

export default function Dashboard() {
  const { teachers, faculties, departments, fetchTeachers, fetchFaculties, fetchDepartments, isLoading, error } = useHrStore();

  useEffect(() => {
    fetchTeachers();
    fetchFaculties();
    fetchDepartments();
  }, [fetchTeachers, fetchFaculties, fetchDepartments]);

  const totalTeachers = teachers.length;
  const activeTeachers = teachers.filter((teacher) => teacher.status === 'Active').length;
  const onLeaveTeachers = teachers.filter((teacher) => teacher.status === 'On Leave').length;
  const inactiveTeachers = teachers.filter((teacher) => teacher.status === 'Inactive').length;

  const stats = [
    { label: 'Total Teachers', value: totalTeachers, icon: Users, color: 'text-white', bg: 'bg-gradient-to-br from-primary to-primary-accent shadow-lg shadow-primary/30' },
    { label: 'Active', value: activeTeachers, icon: UserCheck, color: 'text-white', bg: 'bg-gradient-to-br from-green-500 to-emerald-400 shadow-lg shadow-green-500/30' },
    { label: 'On Leave', value: onLeaveTeachers, icon: UserMinus, color: 'text-white', bg: 'bg-gradient-to-br from-amber-500 to-orange-400 shadow-lg shadow-amber-500/30' },
    { label: 'Inactive', value: inactiveTeachers, icon: UserX, color: 'text-white', bg: 'bg-gradient-to-br from-red-500 to-rose-400 shadow-lg shadow-red-500/30' },
  ];

  const facultyDistribution = useMemo(() => {
    return faculties.map((faculty) => ({
      name: faculty.name,
      value: teachers.filter((teacher) => teacher.facultyId === faculty.id).length,
    })).filter((entry) => entry.value > 0);
  }, [faculties, teachers]);

  const roleDistribution = useMemo(() => {
    const counts = new Map<string, number>();
    teachers.forEach((teacher) => {
      counts.set(teacher.role, (counts.get(teacher.role) ?? 0) + 1);
    });
    return Array.from(counts.entries()).map(([name, value]) => ({ name, value }));
  }, [teachers]);

  const COLORS = ['#2563EB', '#3B82F6', '#60A5FA', '#1E40AF', '#1D4ED8'];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">HR Overview</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Monitor staff metrics and distributions.</p>
        </div>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      ) : null}

      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      ) : (
        <>
          {/* Stats Grid */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.4 }}
                className="glass-card p-6 rounded-2xl border border-gray-200 dark:border-white/10"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{stat.label}</p>
                    <p className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{stat.value}</p>
                  </div>
                  <div className={`p-3 rounded-xl ${stat.bg}`}>
                    <stat.icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Faculty Distribution Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.4 }}
              className="lg:col-span-2 glass-card p-6 rounded-2xl border border-gray-200 dark:border-white/10"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Teacher Distribution by Faculty</h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={facultyDistribution} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#374151" opacity={0.2} />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6b7280', fontSize: 12 }} dy={10} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6b7280', fontSize: 12 }} />
                    <Tooltip 
                      cursor={{ fill: 'transparent' }}
                      contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.8)', borderColor: 'rgba(255,255,255,0.1)', color: '#fff', borderRadius: '8px' }}
                    />
                    <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={40} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Role Distribution Pie Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.4 }}
              className="glass-card p-6 rounded-2xl border border-gray-200 dark:border-white/10 flex flex-col"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Roles Distribution</h3>
              <div className="h-64 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={roleDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      stroke="none"
                    >
                      {roleDistribution.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.8)', borderColor: 'rgba(255,255,255,0.1)', color: '#fff', borderRadius: '8px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-4">
                {roleDistribution.map((role, idx) => (
                  <div key={role.name} className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></span>
                    <span className="text-xs text-gray-600 dark:text-gray-400">{role.name}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Recent Activity Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.4 }}
            className="glass-card p-6 rounded-2xl border border-gray-200 dark:border-white/10"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent HR Activity</h3>
              <button className="text-sm font-medium text-primary hover:text-primary-accent transition-colors flex items-center gap-1">
                View All <Activity size={16} />
              </button>
            </div>
            
            <div className="space-y-4">
              {(teachers.slice(0, 3).map((teacher, idx) => ({
                title: teacher.fullName,
                desc: `${teacher.role} in faculty ${teacher.facultyId || 'unassigned'}`,
                time: idx === 0 ? 'Recent' : `${idx + 1} entries back`,
                icon: idx === 0 ? UserCheck : idx === 1 ? Activity : Users,
                color: idx === 0 ? 'text-green-500' : idx === 1 ? 'text-blue-500' : 'text-purple-500',
                bg: idx === 0 ? 'bg-green-500/10' : idx === 1 ? 'bg-blue-500/10' : 'bg-purple-500/10',
              }))).map((item, idx) => (
                <div key={idx} className="flex items-start gap-4 p-3 hover:bg-gray-50 dark:hover:bg-white/5 rounded-xl transition-colors">
                  <div className={`p-2 rounded-lg ${item.bg}`}>
                    <item.icon className={`w-5 h-5 ${item.color}`} />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">{item.title}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{item.desc}</p>
                  </div>
                  <span className="text-xs font-medium text-gray-400">{item.time}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </div>
  );
}
