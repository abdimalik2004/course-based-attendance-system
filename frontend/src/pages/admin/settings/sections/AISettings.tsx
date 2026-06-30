import { useState, useEffect, useRef } from "react";
import { Brain, RefreshCw, CheckCircle2, XCircle, Loader2, AlertTriangle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { api } from "@/services/api";
import admissionService from "@/services/admissionService";

type JobStatus = "idle" | "queued" | "running" | "succeeded" | "failed";

export function AISettings() {
  const [status, setStatus] = useState<JobStatus>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const [finishedAt, setFinishedAt] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll job status until terminal
  useEffect(() => {
    if (!jobId || status === "succeeded" || status === "failed" || status === "idle") {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const job = await admissionService.getTrainingJob(jobId);
        setStatus(job.status);
        if (job.started_at) setStartedAt(job.started_at);
        if (job.finished_at) setFinishedAt(job.finished_at);
        if (job.status === "failed") {
          setErrorMsg(job.error ?? "Training failed. Check backend logs.");
        }
      } catch {
        // Network blip — keep polling
      }
    }, 2000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId, status]);

  const handleRetrain = async () => {
    setStatus("queued");
    setJobId(null);
    setErrorMsg("");
    setStartedAt(null);
    setFinishedAt(null);

    try {
      const res = await api.post<{ job_id: string; status: string }>("/training/retrain");
      setJobId(res.data.job_id);
      setStatus("queued");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setErrorMsg(typeof detail === "string" ? detail : "Failed to start retraining.");
      setStatus("failed");
    }
  };

  const isRunning = status === "queued" || status === "running";

  const statusBadge = () => {
    switch (status) {
      case "queued":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2.5 py-1 rounded-full">
            <Loader2 size={11} className="animate-spin" />
            Queued
          </span>
        );
      case "running":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-full">
            <Loader2 size={11} className="animate-spin" />
            Training…
          </span>
        );
      case "succeeded":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full">
            <CheckCircle2 size={11} />
            Succeeded
          </span>
        );
      case "failed":
        return (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2.5 py-1 rounded-full">
            <XCircle size={11} />
            Failed
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Face Recognition Model</h2>
        <p className="text-sm text-gray-400">
          Manage the AI model that recognises students during attendance sessions.
        </p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-6">

          {/* What this does */}
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 space-y-2">
            <div className="flex items-center gap-2 text-white font-medium">
              <Brain size={16} className="text-primary" />
              Retrain All Students
            </div>
            <p className="text-sm text-gray-400 leading-relaxed">
              Rebuilds the face recognition model using every student's captured
              images on disk. Run this after:
            </p>
            <ul className="text-sm text-gray-400 space-y-1 ml-4 list-disc">
              <li>New students have been registered and face images captured</li>
              <li>A student's face images were replaced or updated</li>
              <li>The recognition accuracy settings were changed</li>
              <li>Any recognition accuracy issues are observed</li>
            </ul>
            <p className="text-xs text-amber-400 mt-1">
              Training runs in the background — attendance sessions continue
              working with the old model until training completes.
            </p>
          </div>

          {/* Status row */}
          {status !== "idle" && (
            <div className="rounded-xl border border-white/10 bg-white/5 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-300">Training Job</span>
                {statusBadge()}
              </div>

              {jobId && (
                <div className="text-xs text-gray-500 font-mono">
                  Job ID: {jobId}
                </div>
              )}

              {startedAt && (
                <div className="text-xs text-gray-500">
                  Started: {new Date(startedAt).toLocaleTimeString()}
                </div>
              )}

              {finishedAt && (
                <div className="text-xs text-gray-500">
                  Finished: {new Date(finishedAt).toLocaleTimeString()}
                </div>
              )}

              {status === "succeeded" && (
                <div className="flex items-start gap-2 text-sm text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
                  <CheckCircle2 size={15} className="shrink-0 mt-0.5" />
                  <span>
                    Training complete. The new model is now active — all
                    students will be recognised using the updated embeddings.
                  </span>
                </div>
              )}

              {status === "failed" && errorMsg && (
                <div className="flex items-start gap-2 text-sm text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-lg p-3">
                  <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                  <span>{errorMsg}</span>
                </div>
              )}
            </div>
          )}

          {/* Action button */}
          <div className="flex items-center gap-4 pt-1">
            <Button
              onClick={handleRetrain}
              disabled={isRunning}
              isLoading={isRunning}
              className="min-w-[200px] flex items-center gap-2"
            >
              {!isRunning && <RefreshCw size={15} />}
              {isRunning ? "Training in progress…" : "Retrain All Students"}
            </Button>

            {status === "succeeded" && (
              <Button
                variant="ghost"
                onClick={() => {
                  setStatus("idle");
                  setJobId(null);
                }}
              >
                Dismiss
              </Button>
            )}
          </div>

        </CardContent>
      </Card>
    </div>
  );
}
