import { useEffect, useState } from "react";
import { Search, Link2, Link2Off } from "lucide-react";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { hrService, type AvailableUser } from "@/services/hrService";

interface LinkUserModalProps {
  isOpen: boolean;
  onClose: () => void;
  teacherName: string;
  currentLinkedUsername: string | null;
  onLink: (userId: string) => Promise<void>;
  onUnlink: () => Promise<void>;
}

export function LinkUserModal({
  isOpen,
  onClose,
  teacherName,
  currentLinkedUsername,
  onLink,
  onUnlink,
}: LinkUserModalProps) {
  const [users, setUsers] = useState<AvailableUser[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setSearch("");
    setError(null);
    setIsLoading(true);
    hrService
      .getAvailableUsers()
      .then(setUsers)
      .catch(() => setError("Failed to load available users"))
      .finally(() => setIsLoading(false));
  }, [isOpen]);

  const filtered = users.filter(
    (u) =>
      u.username.toLowerCase().includes(search.toLowerCase()) ||
      (u.email ?? "").toLowerCase().includes(search.toLowerCase()),
  );

  const handleLink = async (userId: number) => {
    setIsSubmitting(true);
    setError(null);
    try {
      await onLink(String(userId));
      onClose();
    } catch (err: any) {
      setError(
        err?.response?.data?.detail ??
          err?.response?.data?.error?.message ??
          "Failed to link account",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUnlink = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      await onUnlink();
      onClose();
    } catch (err: any) {
      setError(
        err?.response?.data?.detail ??
          err?.response?.data?.error?.message ??
          "Failed to unlink account",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Link Login Account"
      className="md:max-w-md"
    >
      <div className="space-y-4">
        {/* Current link */}
        {currentLinkedUsername && (
          <div className="flex items-center justify-between rounded-xl border border-blue-200 bg-blue-50 dark:border-blue-500/20 dark:bg-blue-500/10 px-4 py-3">
            <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-300">
              <Link2 size={15} />
              <span>
                Currently linked to{" "}
                <span className="font-semibold">@{currentLinkedUsername}</span>
              </span>
            </div>
            <button
              onClick={handleUnlink}
              disabled={isSubmitting}
              className="ml-4 flex items-center gap-1.5 text-xs font-medium text-rose-600 hover:text-rose-700 dark:text-rose-400 disabled:opacity-50"
            >
              <Link2Off size={13} />
              Unlink
            </button>
          </div>
        )}

        {!currentLinkedUsername && (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Select a login account to link to{" "}
            <span className="font-medium text-gray-700 dark:text-gray-300">
              {teacherName}
            </span>
            . Only users with the <span className="font-mono text-xs">TEACHER</span> role
            that aren't already linked are shown.
          </p>
        )}

        {error && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
            {error}
          </div>
        )}

        {/* Search */}
        <div className="relative">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">
            <Search size={15} />
          </div>
          <input
            type="text"
            placeholder="Search by username or email…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-white/5 pl-9 pr-4 py-2.5 text-sm text-gray-900 dark:text-white placeholder-gray-400 outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
          />
        </div>

        {/* User list */}
        <div className="max-h-60 overflow-y-auto custom-scrollbar rounded-xl border border-gray-200 dark:border-white/10">
          {isLoading ? (
            <div className="space-y-1 p-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-12 rounded-lg bg-gray-100 dark:bg-white/5 animate-pulse" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-8 text-center text-sm text-gray-400 dark:text-gray-500">
              {users.length === 0
                ? "No unlinked TEACHER accounts found."
                : "No results match your search."}
            </div>
          ) : (
            <div className="p-1.5 space-y-0.5">
              {filtered.map((u) => (
                <button
                  key={u.id}
                  onClick={() => handleLink(u.id)}
                  disabled={isSubmitting}
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left hover:bg-gray-50 dark:hover:bg-white/5 transition-colors disabled:opacity-50 group"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-semibold">
                    {u.username[0].toUpperCase()}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-gray-900 dark:text-white">
                      @{u.username}
                    </p>
                    {u.email && (
                      <p className="truncate text-xs text-gray-400 dark:text-gray-500">
                        {u.email}
                      </p>
                    )}
                  </div>
                  <Link2
                    size={14}
                    className="shrink-0 text-gray-300 dark:text-gray-600 group-hover:text-primary transition-colors"
                  />
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex justify-end pt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </div>
    </Modal>
  );
}
