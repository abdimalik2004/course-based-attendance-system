import { useEffect, useState, useMemo } from 'react';
import {
  BarChart3, TrendingUp, BookOpen, AlertTriangle, ChevronRight, ChevronDown,
  Calendar, RefreshCw, ArrowUpDown, ArrowUp, ArrowDown, Building2, X,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { useAcademiaReportsStore, type DatePreset, type TrendPeriod } from '@/store/useAcademiaReportsStore';
import type { FacultyStat, DepartmentStat, CourseStat } from '@/services/academiaReportsService';

// ─── helpers ────────────────────────────────────────────────────────────────

function pctColor(pct: number): string {
  if (pct >= 75) return 'text-emerald-600 dark:text-emerald-400';
  if (pct >= 50) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400';
}

function pctBg(pct: number): string {
  if (pct >= 75) return 'bg-emerald-500';
  if (pct >= 50) return 'bg-amber-500';
  return 'bg-red-500';
}

function pctBadge(pct: number) {
  if (pct >= 75) return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300';
  if (pct >= 50) return 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300';
  return 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300';
}

function pctLabel(pct: number): string {
  if (pct >= 75) return 'Good';
  if (pct >= 50) return 'Warning';
  return 'Low';
}

const CHART_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#14b8a6',
];

function getPresetDates(preset: DatePreset): { start: string; end: string } {
  const today = new Date();
  const fmt = (d: Date) => d.toISOString().split('T')[0];
  if (preset === 'week') {
    const start = new Date(today); start.setDate(today.getDate() - 7);
    return { start: fmt(start), end: fmt(today) };
  }
  if (preset === 'month') {
    const start = new Date(today); start.setMonth(today.getMonth() - 1);
    return { start: fmt(start), end: fmt(today) };
  }
  if (preset === 'semester') {
    const start = new Date(today); start.setMonth(today.getMonth() - 6);
    return { start: fmt(start), end: fmt(today) };
  }
  return { start: '', end: '' };
}

// ─── Filter Bar ─────────────────────────────────────────────────────────────

function FilterBar() {
  const { filters, setFilters, fetchAll, fetchTrends, fetchCourseRanking, fetchComparison, loading } = useAcademiaReportsStore();
  const isLoading = Object.values(loading).some(Boolean);

  const presets: { key: DatePreset; label: string }[] = [
    { key: 'all', label: 'All Time' },
    { key: 'week', label: 'Last 7 days' },
    { key: 'month', label: 'Last 30 days' },
    { key: 'semester', label: 'Last 6 months' },
    { key: 'custom', label: 'Custom' },
  ];

  const handlePreset = (preset: DatePreset) => {
    const dates = getPresetDates(preset);
    setFilters({ preset, startDate: dates.start, endDate: dates.end });
  };

  const handleRefresh = () => {
    fetchAll();
  };

  return (
    <div className="flex flex-wrap items-center gap-3 mb-6">
      {/* Preset buttons */}
      <div className="flex items-center gap-1 rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-1 flex-wrap">
        {presets.map(p => (
          <button
            key={p.key}
            onClick={() => handlePreset(p.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              filters.preset === p.key
                ? 'bg-primary text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Custom date range */}
      {filters.preset === 'custom' && (
        <div className="flex items-center gap-2">
          <div className="relative">
            <Calendar size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            <input
              type="date"
              value={filters.startDate}
              onChange={e => setFilters({ startDate: e.target.value })}
              className="pl-8 pr-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <span className="text-xs text-gray-400">to</span>
          <div className="relative">
            <Calendar size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            <input
              type="date"
              value={filters.endDate}
              onChange={e => setFilters({ endDate: e.target.value })}
              className="pl-8 pr-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <button
            onClick={handleRefresh}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-primary text-white hover:bg-primary/90 transition-colors"
          >
            Apply
          </button>
        </div>
      )}

      {/* Refresh */}
      <button
        onClick={handleRefresh}
        disabled={isLoading}
        className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors disabled:opacity-50"
      >
        <RefreshCw size={13} className={isLoading ? 'animate-spin' : ''} />
        Refresh
      </button>
    </div>
  );
}

// ─── Section 1: Faculty Comparison ──────────────────────────────────────────

function FacultyComparisonSection() {
  const { comparison, loading, fetchDepartments, filters } = useAcademiaReportsStore();
  const isLoading = loading.comparison;

  if (isLoading) return <SectionSkeleton rows={4} />;
  if (!comparison) return null;

  const { faculties, institution_avg } = comparison;
  const best = faculties[0];
  const worst = faculties[faculties.length - 1];

  return (
    <div className="space-y-4">
      {/* Summary strip */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Institution Avg', value: `${institution_avg}%`, sub: 'overall attendance', color: pctColor(institution_avg) },
          { label: 'Best Faculty', value: best ? `${best.attendance_pct}%` : '—', sub: best?.faculty_name ?? '—', color: 'text-emerald-600 dark:text-emerald-400' },
          { label: 'Lowest Faculty', value: worst ? `${worst.attendance_pct}%` : '—', sub: worst?.faculty_name ?? '—', color: 'text-red-500 dark:text-red-400' },
        ].map(card => (
          <div key={card.label} className="rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400">{card.label}</p>
            <p className={`text-2xl font-bold mt-1 ${card.color}`}>{card.value}</p>
            <p className="text-xs text-gray-400 mt-0.5 truncate">{card.sub}</p>
          </div>
        ))}
      </div>

      {/* Faculty cards */}
      <div className="space-y-3">
        {faculties.map((f, idx) => (
          <FacultyCard key={f.faculty_id} faculty={f} rank={idx + 1} onDrill={() => fetchDepartments(f.faculty_id)} />
        ))}
        {faculties.length === 0 && (
          <EmptyState message="No attendance data for this period." />
        )}
      </div>
    </div>
  );
}

function FacultyCard({ faculty: f, rank, onDrill }: { faculty: FacultyStat; rank: number; onDrill: () => void }) {
  const { filters, departments, loading, setFilters } = useAcademiaReportsStore();
  const isExpanded = filters.drillFacultyId === f.faculty_id;
  const isDrillLoading = loading.departments && isExpanded;

  return (
    <div className="rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card overflow-hidden">
      {/* Main row */}
      <div className="flex items-center gap-4 px-5 py-4">
        {/* Rank badge */}
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gray-100 dark:bg-white/5 text-xs font-bold text-gray-500 dark:text-gray-400">
          {rank}
        </span>

        {/* Name + code */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-gray-900 dark:text-white">{f.faculty_name}</span>
            <span className="text-xs text-gray-400 bg-gray-100 dark:bg-white/5 px-2 py-0.5 rounded-full">{f.faculty_code}</span>
          </div>
          <div className="flex items-center gap-4 mt-1 text-xs text-gray-500 dark:text-gray-400 flex-wrap">
            <span>{f.total_students.toLocaleString()} students</span>
            <span>{f.total_sessions.toLocaleString()} sessions</span>
            <span>{f.total_records.toLocaleString()} records</span>
            {f.at_risk_students > 0 && (
              <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400">
                <AlertTriangle size={11} />
                {f.at_risk_students} at-risk
              </span>
            )}
          </div>
        </div>

        {/* Progress bar + pct */}
        <div className="w-32 shrink-0">
          <div className="flex justify-between items-center mb-1">
            <span className={`text-sm font-bold ${pctColor(f.attendance_pct)}`}>{f.attendance_pct}%</span>
            <span className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${pctBadge(f.attendance_pct)}`}>
              {pctLabel(f.attendance_pct)}
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-gray-100 dark:bg-white/10 overflow-hidden">
            <div className={`h-full rounded-full transition-all ${pctBg(f.attendance_pct)}`} style={{ width: `${f.attendance_pct}%` }} />
          </div>
        </div>

        {/* Drill button */}
        {f.total_records > 0 && (
          <button
            onClick={() => isExpanded ? setFilters({ drillFacultyId: null }) : onDrill()}
            className="shrink-0 flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium text-primary hover:bg-primary/10 transition-colors"
          >
            Departments
            {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          </button>
        )}
      </div>

      {/* Department drill-down */}
      {isExpanded && (
        <div className="border-t border-gray-100 dark:border-white/5 px-5 py-4 bg-gray-50 dark:bg-dark-bg">
          {isDrillLoading ? (
            <div className="text-xs text-gray-400 animate-pulse">Loading departments…</div>
          ) : departments && departments.faculty_id === f.faculty_id ? (
            <DepartmentDrilldown />
          ) : null}
        </div>
      )}
    </div>
  );
}

// ─── Section 2: Department Drill-down (rendered inside FacultyCard) ──────────

function DepartmentDrilldown() {
  const { departments } = useAcademiaReportsStore();
  if (!departments) return null;

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
        Departments — {departments.faculty_name}
      </p>
      {departments.departments.map(dept => (
        <DeptRow key={dept.department_id} dept={dept} />
      ))}
      {departments.departments.length === 0 && (
        <p className="text-xs text-gray-400">No department data for this period.</p>
      )}
    </div>
  );
}

function DeptRow({ dept: d }: { dept: DepartmentStat }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card px-4 py-3">
      <div className="min-w-0 flex-1">
        <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{d.department_name}</span>
        <span className="ml-2 text-xs text-gray-400">{d.department_code}</span>
        <div className="text-xs text-gray-400 mt-0.5">
          {d.total_students} students · {d.total_records} records
        </div>
      </div>
      <div className="w-28 shrink-0">
        <div className="flex justify-between items-center mb-1">
          <span className={`text-sm font-bold ${pctColor(d.attendance_pct)}`}>{d.attendance_pct}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-gray-100 dark:bg-white/10 overflow-hidden">
          <div className={`h-full rounded-full ${pctBg(d.attendance_pct)}`} style={{ width: `${d.attendance_pct}%` }} />
        </div>
      </div>
    </div>
  );
}

// ─── Section 3: Trend Chart ──────────────────────────────────────────────────

function TrendSection() {
  const { trends, loading, filters, setFilters, fetchTrends } = useAcademiaReportsStore();
  const isLoading = loading.trends;

  const chartData = useMemo(() => {
    if (!trends) return [];
    return trends.series.map(point => {
      const row: Record<string, string | number | null> = { period: point.period };
      point.faculties.forEach(f => { row[f.faculty_name] = f.pct; });
      return row;
    });
  }, [trends]);

  return (
    <div className="rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-5">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <TrendingUp size={18} className="text-primary" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Attendance Trends</h3>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-gray-200 dark:border-white/10 p-0.5">
          {(['weekly', 'monthly'] as TrendPeriod[]).map(p => (
            <button
              key={p}
              onClick={() => { setFilters({ trendPeriod: p }); fetchTrends(); }}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors capitalize ${
                filters.trendPeriod === p
                  ? 'bg-primary text-white'
                  : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="h-64 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      ) : !trends || chartData.length === 0 ? (
        <EmptyState message="Not enough data to show a trend for this period." height="h-48" />
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
            <XAxis dataKey="period" tick={{ fontSize: 11 }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} tickLine={false} tickFormatter={v => `${v}%`} />
            <Tooltip
              formatter={(value: number | null, name: string) => [value !== null ? `${value}%` : 'No data', name]}
              contentStyle={{
                background: 'var(--color-dark-card, #1e2433)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {trends.faculty_names.map((name, i) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ─── Section 4: Course Ranking ───────────────────────────────────────────────

type SortKey = 'attendance_pct' | 'total_records' | 'course_title' | 'faculty_name';
type SortDir = 'asc' | 'desc';

function CourseRankingSection() {
  const { courseRanking, loading, comparison, filters, setFilters, fetchCourseRanking } = useAcademiaReportsStore();
  const isLoading = loading.courses;
  const [sortKey, setSortKey] = useState<SortKey>('attendance_pct');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [search, setSearch] = useState('');

  const faculties = comparison?.faculties ?? [];

  const sorted = useMemo(() => {
    if (!courseRanking) return [];
    let rows = [...courseRanking.courses];
    if (search) {
      const s = search.toLowerCase();
      rows = rows.filter(r =>
        r.course_title.toLowerCase().includes(s) ||
        r.course_code.toLowerCase().includes(s) ||
        r.faculty_name.toLowerCase().includes(s)
      );
    }
    rows.sort((a, b) => {
      const av = a[sortKey] as string | number;
      const bv = b[sortKey] as string | number;
      if (typeof av === 'string' && typeof bv === 'string') {
        return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sortDir === 'asc' ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
    return rows;
  }, [courseRanking, sortKey, sortDir, search]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (sortKey !== k) return <ArrowUpDown size={12} className="opacity-30" />;
    return sortDir === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />;
  };

  return (
    <div className="rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 dark:border-white/5">
        <div className="flex items-center gap-2">
          <BookOpen size={18} className="text-primary" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Course Attendance Ranking</h3>
          {courseRanking && (
            <span className="text-xs text-gray-400 bg-gray-100 dark:bg-white/5 px-2 py-0.5 rounded-full">
              {courseRanking.total} courses
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Faculty filter */}
          <select
            value={filters.courseFilterFacultyId ?? ''}
            onChange={e => {
              const v = e.target.value;
              setFilters({ courseFilterFacultyId: v ? Number(v) : null });
              fetchCourseRanking();
            }}
            className="text-xs rounded-lg border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card text-gray-700 dark:text-gray-300 px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="">All Faculties</option>
            {faculties.map(f => (
              <option key={f.faculty_id} value={f.faculty_id}>{f.faculty_name}</option>
            ))}
          </select>
          {/* Search */}
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="text-xs rounded-lg border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card text-gray-700 dark:text-gray-300 px-3 py-1.5 w-36 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
      </div>

      {/* Table */}
      {isLoading ? (
        <SectionSkeleton rows={6} />
      ) : sorted.length === 0 ? (
        <EmptyState message="No course data for this period." />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 dark:border-white/5 bg-gray-50 dark:bg-white/5">
                <th className="text-left px-5 py-3 text-xs font-medium text-gray-500 dark:text-gray-400 w-8">#</th>
                <ColHeader label="Course" sortKey="course_title" active={sortKey} dir={sortDir} onSort={toggleSort} />
                <ColHeader label="Faculty" sortKey="faculty_name" active={sortKey} dir={sortDir} onSort={toggleSort} />
                <ColHeader label="Records" sortKey="total_records" active={sortKey} dir={sortDir} onSort={toggleSort} />
                <ColHeader label="Attendance" sortKey="attendance_pct" active={sortKey} dir={sortDir} onSort={toggleSort} />
                <th className="px-5 py-3 text-xs font-medium text-gray-500 dark:text-gray-400">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50 dark:divide-white/5">
              {sorted.map((c, i) => <CourseRow key={c.course_id} course={c} rank={i + 1} />)}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ColHeader({ label, sortKey, active, dir, onSort }: {
  label: string; sortKey: SortKey; active: SortKey; dir: SortDir; onSort: (k: SortKey) => void;
}) {
  return (
    <th
      className="text-left px-5 py-3 text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-200"
      onClick={() => onSort(sortKey)}
    >
      <div className="flex items-center gap-1">
        {label}
        {active === sortKey
          ? (dir === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)
          : <ArrowUpDown size={12} className="opacity-30" />
        }
      </div>
    </th>
  );
}

function CourseRow({ course: c, rank }: { course: CourseStat; rank: number }) {
  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
      <td className="px-5 py-3 text-xs text-gray-400">{rank}</td>
      <td className="px-5 py-3">
        <div className="font-medium text-gray-800 dark:text-gray-200">{c.course_title}</div>
        <div className="text-xs text-gray-400">{c.course_code}</div>
      </td>
      <td className="px-5 py-3 text-xs text-gray-500 dark:text-gray-400">{c.faculty_name}</td>
      <td className="px-5 py-3 text-xs text-gray-500 dark:text-gray-400">{c.total_records.toLocaleString()}</td>
      <td className="px-5 py-3">
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 rounded-full bg-gray-100 dark:bg-white/10 overflow-hidden">
            <div className={`h-full rounded-full ${pctBg(c.attendance_pct)}`} style={{ width: `${c.attendance_pct}%` }} />
          </div>
          <span className={`text-sm font-semibold ${pctColor(c.attendance_pct)}`}>{c.attendance_pct}%</span>
        </div>
      </td>
      <td className="px-5 py-3">
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${pctBadge(c.attendance_pct)}`}>
          {pctLabel(c.attendance_pct)}
        </span>
      </td>
    </tr>
  );
}

// ─── Section 5: At-Risk Summary ──────────────────────────────────────────────

function AtRiskSection() {
  const { comparison, loading } = useAcademiaReportsStore();
  const isLoading = loading.comparison;

  const atRiskFaculties = useMemo(() => {
    if (!comparison) return [];
    return comparison.faculties
      .filter(f => f.at_risk_students > 0)
      .sort((a, b) => b.at_risk_students - a.at_risk_students);
  }, [comparison]);

  const totalAtRisk = atRiskFaculties.reduce((s, f) => s + f.at_risk_students, 0);
  const totalStudents = comparison?.faculties.reduce((s, f) => s + f.total_students, 0) ?? 0;

  return (
    <div className="rounded-xl border border-gray-200 dark:border-white/10 bg-white dark:bg-dark-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle size={18} className="text-amber-500" />
        <h3 className="font-semibold text-gray-900 dark:text-white">At-Risk Students</h3>
        <span className="text-xs text-gray-400">(below 75% attendance)</span>
      </div>

      {isLoading ? (
        <SectionSkeleton rows={3} />
      ) : !comparison ? null : atRiskFaculties.length === 0 ? (
        <div className="flex items-center gap-3 py-4 text-sm text-emerald-600 dark:text-emerald-400">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-500/20">
            <TrendingUp size={16} />
          </div>
          No at-risk students — all faculties are above the 75% threshold.
        </div>
      ) : (
        <>
          <div className="flex items-baseline gap-2 mb-4">
            <span className="text-3xl font-bold text-amber-600 dark:text-amber-400">{totalAtRisk}</span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              of {totalStudents.toLocaleString()} students institution-wide
            </span>
            <span className="text-xs text-gray-400">
              ({totalStudents > 0 ? ((totalAtRisk / totalStudents) * 100).toFixed(1) : 0}%)
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {atRiskFaculties.map(f => (
              <div key={f.faculty_id} className="rounded-lg border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/5 p-3">
                <p className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate">{f.faculty_name}</p>
                <p className="text-xl font-bold text-amber-600 dark:text-amber-400 mt-1">{f.at_risk_students}</p>
                <p className="text-xs text-gray-400">
                  of {f.total_students} ({f.total_students > 0 ? ((f.at_risk_students / f.total_students) * 100).toFixed(1) : 0}%)
                </p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Shared UI helpers ───────────────────────────────────────────────────────

function SectionSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3 p-4 animate-pulse">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-10 rounded-lg bg-gray-100 dark:bg-white/5" />
      ))}
    </div>
  );
}

function EmptyState({ message, height = 'h-24' }: { message: string; height?: string }) {
  return (
    <div className={`${height} flex items-center justify-center text-sm text-gray-400 dark:text-gray-500 p-4`}>
      {message}
    </div>
  );
}

function SectionHeader({ icon: Icon, title, children }: {
  icon: React.ElementType; title: string; children?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <Icon size={18} className="text-primary" />
        <h2 className="text-base font-semibold text-gray-900 dark:text-white">{title}</h2>
      </div>
      {children}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function AcademiaReports() {
  const { fetchAll, error } = useAcademiaReportsStore();

  useEffect(() => {
    fetchAll();
  }, []);

  return (
    <div className="space-y-6">
      {/* Page heading */}
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-white">Attendance Reports</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
          Cross-faculty attendance comparison, trends, and at-risk analysis
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20 px-4 py-3 text-sm text-red-700 dark:text-red-400">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Filters */}
      <FilterBar />

      {/* Section 1 — Faculty Comparison */}
      <section>
        <SectionHeader icon={BarChart3} title="Faculty Comparison" />
        <FacultyComparisonSection />
      </section>

      {/* Section 2 — Trend Chart */}
      <TrendSection />

      {/* Section 3 — Course Ranking */}
      <CourseRankingSection />

      {/* Section 4 — At-Risk */}
      <AtRiskSection />
    </div>
  );
}
