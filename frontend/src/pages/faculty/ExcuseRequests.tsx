import { useState } from 'react';
import { motion } from 'framer-motion';
import { FilePen, CheckCircle2, XCircle, AlertCircle, User, CalendarDays, BookOpen } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { cn } from '@/utils/cn';
import { facultyService } from '@/services/facultyService';

interface ExcuseRequest {
  id: number;
  student_name: string | null;
  student_number: string | null;
  course_id: number | null;
  course_name: string | null;
  course_code: string | null;
  request_date: string;
  reason: string | null;
  status: 'PENDING' | 'APPROVED' | 'DENIED';
  created_at: string;
  reviewed_at: string | null;
}

type TabFilter = 'PENDING' | 'APPROVED' | 'DENIED' | 'ALL';

const TAB_OPTIONS: { key: TabFilter; label: string }[] = [
  { key: 'PENDING', label: 'Pending' },
  { key: 'APPROVED', label: 'Approved' },
  { key: 'DENIED', label: 'Denied' },
  { key: 'ALL', label: 'All' },
];

const STATUS_CONFIG = {
  PENDING: { label: 'Pending', Icon: AlertCircle, color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-500/20' },
  APPROVED: { label: 'Approved', Icon: CheckCircle2, color: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/20' },
  DENIED: { label: 'Denied', Icon: XCircle, color: 'text-rose-600 dark:text-rose-400', bg: 'bg-rose-50 dark:bg-rose-500/10 border-rose-200 dark:border-rose-500/20' },
} as const;

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
}

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const h = Math.floor(diff / 3_600_000);
  if (h < 1) return 'Just now';
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

export default function ExcuseRequests() {
  const [tab, setTab] = useState<TabFilter>('PENDING');
  const [reviewError, setReviewError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const { data: requests = [], isLoading, isError, refetch } = useQuery<ExcuseRequest[]>({
    queryKey: ['facultyExcuseRequests'],
    queryFn: () => facultyService.listExcuseRequests(),
    staleTime: 1000 * 60,
    refetchInterval: 1000 * 60 * 2,
  });

  const reviewMutation = useMutation({
    mutationFn: ({ id, action }: { id: number; action: 'approve' | 'deny' }) =>
      facultyService.reviewExcuseRequest(id, action),
    onSuccess: () => {
      setReviewError(null);
      void queryClient.invalidateQueries({ queryKey: ['facultyExcuseRequests'] });
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : "Failed to update request. Please try again.";
      setReviewError(msg);
    },
  });

  const filtered = tab === 'ALL' ? requests : requests.filter((r) => r.status === tab);
  const pendingCount = requests.filter((r) => r.status === 'PENDING').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">Excuse Requests</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Review and approve student excuse requests for your faculty.
          </p>
        </div>
        {pendingCount > 0 && (
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20">
            <AlertCircle size={16} className="text-amber-600 dark:text-amber-400" />
            <span className="text-sm font-semibold text-amber-700 dark:text-amber-300">
              {pendingCount} pending {pendingCount === 1 ? 'request' : 'requests'}
            </span>
          </div>
        )}
      </div>

      {/* Load error */}
      {isError && (
        <div className="flex items-center justify-between rounded-xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          <span>Failed to load excuse requests.</span>
          <button onClick={() => void refetch()} className="ml-4 text-xs font-semibold underline hover:no-underline">Retry</button>
        </div>
      )}

      {/* Review action error */}
      {reviewError && (
        <div className="flex items-center justify-between rounded-xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          <span>{reviewError}</span>
          <button onClick={() => setReviewError(null)} className="ml-4 text-xs font-semibold underline hover:no-underline">Dismiss</button>
        </div>
      )}

      {/* Tab filter */}
      <div className="flex gap-2 flex-wrap">
        {TAB_OPTIONS.map(({ key, label }) => {
          const count = key === 'ALL' ? requests.length : requests.filter((r) => r.status === key).length;
          return (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all',
                tab === key
                  ? 'bg-primary text-white shadow-[0_0_15px_rgba(37,99,235,0.35)]'
                  : 'bg-white dark:bg-white/5 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-white/10 hover:border-primary/40',
              )}
            >
              {label}
              <span className={cn(
                'text-[11px] font-bold px-1.5 py-0.5 rounded-full',
                tab === key ? 'bg-white/20 text-white' : 'bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-400',
              )}>
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {/* Cards */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 rounded-2xl bg-gray-100 dark:bg-white/5 animate-pulse" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <Card className="glass-card border-gray-200 dark:border-white/5">
            <CardContent className="flex flex-col items-center gap-3 py-16 text-center text-gray-400 dark:text-gray-500">
              <FilePen size={40} className="opacity-20" />
              <p className="text-sm">
                {tab === 'PENDING' ? 'No pending requests — all caught up!' : `No ${tab.toLowerCase()} requests.`}
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {filtered.map((req) => {
              const cfg = STATUS_CONFIG[req.status];
              const StatusIcon = cfg.Icon;
              const isPending = req.status === 'PENDING';
              const isActing = reviewMutation.isPending && (reviewMutation.variables as { id: number })?.id === req.id;

              return (
                <motion.div
                  key={req.id}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="glass-card border border-gray-200 dark:border-white/5 rounded-2xl p-5 flex flex-col sm:flex-row sm:items-center gap-4"
                >
                  {/* Status icon */}
                  <div className={cn('p-2.5 rounded-xl border shrink-0 self-start', cfg.bg)}>
                    <StatusIcon size={18} className={cfg.color} />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0 space-y-2">
                    {/* Student */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className="flex items-center gap-1.5">
                        <User size={13} className="text-gray-400" />
                        <span className="font-semibold text-gray-900 dark:text-white text-sm">
                          {req.student_name ?? 'Unknown student'}
                        </span>
                      </div>
                      {req.student_number && (
                        <span className="text-xs text-gray-400 font-mono dark:text-gray-500">
                          #{req.student_number}
                        </span>
                      )}
                      <span className="text-xs text-gray-400 dark:text-gray-500 ml-auto sm:ml-0">
                        {timeAgo(req.created_at)}
                      </span>
                    </div>

                    {/* Date + course */}
                    <div className="flex items-center gap-3 flex-wrap text-sm text-gray-600 dark:text-gray-300">
                      <div className="flex items-center gap-1.5">
                        <CalendarDays size={13} className="text-gray-400" />
                        <span className="font-medium">{formatDate(req.request_date)}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <BookOpen size={13} className="text-gray-400" />
                        <span>
                          {req.course_name
                            ? <><span className="text-xs font-medium text-primary dark:text-primary-accent bg-primary/5 dark:bg-primary/10 px-1.5 py-0.5 rounded mr-1">{req.course_code}</span>{req.course_name}</>
                            : <span className="italic text-gray-400">All courses that day</span>
                          }
                        </span>
                      </div>
                    </div>

                    {/* Reason */}
                    {req.reason && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                        "{req.reason}"
                      </p>
                    )}

                    {/* Reviewed at */}
                    {req.reviewed_at && (
                      <p className="text-xs text-gray-400 dark:text-gray-500">
                        Reviewed {formatDate(req.reviewed_at)}
                      </p>
                    )}
                  </div>

                  {/* Actions / status */}
                  <div className="flex items-center gap-2 shrink-0">
                    {isPending ? (
                      <>
                        <Button
                          size="sm"
                          variant="secondary"
                          className="gap-1.5 text-rose-600 dark:text-rose-400 border-rose-200 dark:border-rose-500/30 hover:bg-rose-50 dark:hover:bg-rose-500/10"
                          disabled={isActing}
                          onClick={() => reviewMutation.mutate({ id: req.id, action: 'deny' })}
                        >
                          <XCircle size={14} />
                          Deny
                        </Button>
                        <Button
                          size="sm"
                          className="gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white border-0"
                          disabled={isActing}
                          onClick={() => reviewMutation.mutate({ id: req.id, action: 'approve' })}
                        >
                          <CheckCircle2 size={14} />
                          {isActing ? '…' : 'Approve'}
                        </Button>
                      </>
                    ) : (
                      <div className={cn('flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-semibold', cfg.bg, cfg.color)}>
                        <StatusIcon size={13} />
                        {cfg.label}
                      </div>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </motion.div>
    </div>
  );
}
