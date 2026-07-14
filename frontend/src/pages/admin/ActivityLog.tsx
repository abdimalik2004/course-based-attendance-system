import { useEffect, useState, useCallback } from "react";
import {
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ScrollText,
  CheckCircle2,
  XCircle,
  Clock,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { api } from "@/services/api";

interface LogEntry {
  id: number;
  username: string;
  action: string;
  status: string;
  created_at: string;
}

interface LogsResponse {
  total: number;
  skip: number;
  limit: number;
  items: LogEntry[];
}

const PAGE_SIZE = 50;

const STATUS_OPTIONS = [
  { value: "", label: "All Statuses" },
  { value: "Success", label: "Success" },
  { value: "Failed", label: "Failed" },
  { value: "Pending", label: "Pending" },
];

function StatusBadge({ status }: { status: string }) {
  if (status === "Success") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400">
        <CheckCircle2 size={13} />
        Success
      </span>
    );
  }
  if (status === "Failed") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-red-400">
        <XCircle size={13} />
        Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-400">
      <Clock size={13} />
      {status || "Pending"}
    </span>
  );
}

function formatDate(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function ActivityLog() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [usernameFilter, setUsernameFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [actionFilter, setActionFilter] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  // Debounced fetch
  const fetchLogs = useCallback(
    async (currentPage: number) => {
      setIsLoading(true);
      setError(null);
      try {
        const params: Record<string, string | number> = {
          skip: (currentPage - 1) * PAGE_SIZE,
          limit: PAGE_SIZE,
        };
        if (usernameFilter.trim()) params.username = usernameFilter.trim();
        if (statusFilter) params.status = statusFilter;
        if (actionFilter.trim()) params.action = actionFilter.trim();
        if (dateFrom) params.date_from = dateFrom;
        if (dateTo) params.date_to = dateTo;

        const res = await api.get<LogsResponse>("/activity/logs", { params });
        setLogs(res.data.items);
        setTotal(res.data.total);
      } catch (err: any) {
        const detail = err?.response?.data?.error?.message ?? err?.message ?? "Failed to load activity logs";
        setError(detail);
      } finally {
        setIsLoading(false);
      }
    },
    [usernameFilter, statusFilter, actionFilter, dateFrom, dateTo]
  );

  useEffect(() => {
    setPage(1);
  }, [usernameFilter, statusFilter, actionFilter, dateFrom, dateTo]);

  useEffect(() => {
    fetchLogs(page);
  }, [fetchLogs, page]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const handleClearFilters = () => {
    setUsernameFilter("");
    setStatusFilter("");
    setActionFilter("");
    setDateFrom("");
    setDateTo("");
  };

  const hasActiveFilters =
    usernameFilter || statusFilter || actionFilter || dateFrom || dateTo;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <ScrollText className="text-primary" size={28} />
            Activity Log
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Full audit trail of all system actions — {total.toLocaleString()} records
          </p>
        </div>
        <Button
          variant="secondary"
          onClick={() => fetchLogs(page)}
          disabled={isLoading}
          className="flex items-center gap-2"
        >
          <RefreshCw size={15} className={isLoading ? "animate-spin" : ""} />
          Refresh
        </Button>
      </div>

      {/* Filters */}
      <Card className="glass-card shadow-2xl shadow-primary/5">
        <CardContent className="p-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
              <Input
                placeholder="Filter by user…"
                value={usernameFilter}
                onChange={(e) => setUsernameFilter(e.target.value)}
                className="pl-8"
              />
            </div>
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
              <Input
                placeholder="Search action…"
                value={actionFilter}
                onChange={(e) => setActionFilter(e.target.value)}
                className="pl-8"
              />
            </div>
            <Select
              options={STATUS_OPTIONS}
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            />
            <div className="space-y-1">
              <label className="text-xs text-gray-400 ml-1">From date</label>
              <Input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-400 ml-1">To date</label>
              <Input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
              />
            </div>
          </div>
          {hasActiveFilters && (
            <div className="mt-3 flex justify-end">
              <button
                onClick={handleClearFilters}
                className="text-xs text-primary hover:underline"
              >
                Clear all filters
              </button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Table */}
      <Card className="glass-card shadow-2xl shadow-primary/5">
        <CardContent className="p-0">
          {error ? (
            <div className="flex items-center justify-center h-32 text-red-400 text-sm gap-2">
              <XCircle size={16} />
              {error}
            </div>
          ) : (
            <div className="overflow-x-auto custom-scrollbar w-full">
              <Table className="w-full whitespace-nowrap min-w-max">
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Timestamp</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoading ? (
                    Array.from({ length: 10 }).map((_, i) => (
                      <TableRow key={`sk-${i}`}>
                        {Array.from({ length: 5 }).map((_, j) => (
                          <TableCell key={j}>
                            <div className="h-4 bg-gray-200 dark:bg-white/10 rounded animate-pulse" style={{ width: j === 2 ? "200px" : "80px" }} />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : logs.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} className="h-24 text-center text-gray-500">
                        {hasActiveFilters
                          ? "No log entries match the current filters."
                          : "No activity logs found."}
                      </TableCell>
                    </TableRow>
                  ) : (
                    logs.map((log, idx) => (
                      <TableRow key={log.id}>
                        <TableCell className="text-gray-400 dark:text-gray-500 text-xs font-mono">
                          {(page - 1) * PAGE_SIZE + idx + 1}
                        </TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                          {log.username || <span className="text-gray-400 italic">system</span>}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300 max-w-xs truncate" title={log.action}>
                          {log.action}
                        </TableCell>
                        <TableCell>
                          <StatusBadge status={log.status} />
                        </TableCell>
                        <TableCell className="text-gray-500 dark:text-gray-400 text-sm font-mono">
                          {formatDate(log.created_at)}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pagination */}
      {!error && (
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>
            Showing{" "}
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, total)}
            </span>{" "}
            of{" "}
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {total.toLocaleString()}
            </span>{" "}
            entries
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1 || isLoading}
              className="p-2 rounded-lg hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              // Show pages around current
              let pageNum: number;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (page <= 3) {
                pageNum = i + 1;
              } else if (page >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = page - 2 + i;
              }
              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  disabled={isLoading}
                  className={`w-8 h-8 rounded-lg text-sm font-medium transition-colors ${
                    page === pageNum
                      ? "bg-primary text-white shadow-sm shadow-primary/30"
                      : "hover:bg-white/10 text-gray-500 dark:text-gray-400"
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages || isLoading}
              className="p-2 rounded-lg hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
