import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, XCircle, Info, AlertTriangle, Sun } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Modal } from "@/components/ui/Modal";
import { useAttendanceStore } from "@/store/useAttendanceStore";
import attendanceService from "@/services/attendanceService";
import { attendanceKeys } from "@/hooks/queries/useAttendance";
import { cn } from "@/utils/cn";

// Average pixel brightness considered "too dark" to scan (0–255 scale)
const LOW_LIGHT_THRESHOLD = 45;

function getAverageBrightness(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
): number {
  const { data } = ctx.getImageData(0, 0, w, h);
  let total = 0;
  let count = 0;
  // Sample every 8th pixel for performance (step = 4 channels × 8 pixels = 32)
  for (let i = 0; i < data.length; i += 32) {
    total += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    count++;
  }
  return count > 0 ? total / count : 128;
}

export default function ScannerInterface({ cameraIndex }: { cameraIndex?: number }) {
  const {
    sessionState,
    recognizedUser,
    activeSessionId,
    activeCourseName,
    waitForFace,
    setRecognitionResult,
    resetSession,
  } = useAttendanceStore();

  const queryClient = useQueryClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string>("");
  const [showEndConfirm, setShowEndConfirm] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const consecutiveNoMatchRef = useRef(0);
  // Mutex: prevents two async scan calls from overlapping inside the interval
  const scanInProgressRef = useRef(false);
  // Native browser face detector (Chrome / Edge — Shape Detection API)
  const faceDetectorRef = useRef<any>(null);

  useEffect(() => {
    if ("FaceDetector" in window) {
      try {
        faceDetectorRef.current = new (window as any).FaceDetector({
          fastMode: true,
          maxDetectedFaces: 1,
        });
      } catch {
        faceDetectorRef.current = null;
      }
    }
  }, []);

  // ── Camera start / stop ─────────────────────────────────────────────────
  useEffect(() => {
    if (sessionState !== "idle") {
      if (!stream && !cameraError) {
        (async () => {
          try {
            let constraints: MediaStreamConstraints = { video: true };
            if (typeof cameraIndex === "number" && cameraIndex > 0) {
              try {
                // enumerateDevices() only returns real deviceIds AFTER the browser
                // has been granted camera permission.  Request a temporary stream
                // first to unlock the permission gate, then enumerate properly.
                const permStream = await navigator.mediaDevices.getUserMedia({ video: true });
                permStream.getTracks().forEach((t) => t.stop());

                const list = await navigator.mediaDevices.enumerateDevices();
                const cams = list.filter((d) => d.kind === "videoinput");
                const device = cams[cameraIndex];
                if (device?.deviceId) {
                  constraints = { video: { deviceId: { exact: device.deviceId } } };
                }
              } catch {
                // If enumeration fails, fall back to the default camera
              }
            }
            const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            setStream(mediaStream);
          } catch {
            setCameraError("Camera permission denied or not found");
          }
        })();
      }
    } else {
      stream?.getTracks().forEach((t) => t.stop());
      setStream(null);
      setCameraError("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionState]);

  useEffect(() => {
    if (videoRef.current && stream && videoRef.current.srcObject !== stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream, sessionState]);

  // ── Capture one video frame ─────────────────────────────────────────────
  const captureFrame = (): { imageUrl: string; brightness: number } | null => {
    const video = videoRef.current;
    if (!video || !activeSessionId) return null;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const brightness = getAverageBrightness(ctx, canvas.width, canvas.height);
    return { imageUrl: canvas.toDataURL("image/jpeg", 0.9), brightness };
  };

  // ── End session ─────────────────────────────────────────────────────────
  const handleEndSession = async () => {
    stream?.getTracks().forEach((t) => t.stop());
    if (activeSessionId) {
      await attendanceService.endSession(activeSessionId).catch(() => null);
    }
    // Invalidate all attendance list queries so every attendance-list page
    // (admin, teacher, faculty) automatically re-fetches fresh data.
    queryClient.invalidateQueries({ queryKey: attendanceKeys.all });
    resetSession();
  };

  // ── Continuous scanning loop ────────────────────────────────────────────
  // Uses setInterval so the cleanup can reliably cancel ALL pending ticks
  // with a single clearInterval call — unlike recursive setTimeout where only
  // the initial timer is tracked and subsequent queued callbacks leak.
  //
  // scanInProgressRef acts as a mutex: while one async scan is running, any
  // interval tick that fires before it finishes is a no-op. When a result is
  // obtained (success / failed / already_marked / partial_face / low_light)
  // the mutex is left locked — the next tick will no-op — and the state
  // change triggers the cleanup, which clears the interval and unlocks for
  // the next student's loop.
  useEffect(() => {
    if (sessionState !== "waiting_for_face") return;
    if (!activeSessionId) return;

    consecutiveNoMatchRef.current = 0;
    scanInProgressRef.current = false;

    const doScan = async () => {
      // Mutex guard — skip if a previous scan hasn't finished yet
      if (scanInProgressRef.current) return;
      scanInProgressRef.current = true;

      // ── Capture frame ────────────────────────────────────────────────
      const captured = captureFrame();
      if (!captured) {
        // Video not ready yet — unlock and retry next tick
        scanInProgressRef.current = false;
        return;
      }

      // ── Low-light check (frontend — no backend call needed) ──────────
      if (captured.brightness < LOW_LIGHT_THRESHOLD) {
        // Leave mutex locked; cleanup will unlock when state resets
        setRecognitionResult("low_light");
        return;
      }

      // ── Native face-presence gate (Chrome/Edge FaceDetector API) ─────
      // Only hit the backend when the browser confirms a face is in frame.
      // Falls through silently on browsers that don't support the API.
      if (faceDetectorRef.current && videoRef.current) {
        try {
          const faces: any[] = await faceDetectorRef.current.detect(
            videoRef.current,
          );
          if (faces.length === 0) {
            // No face — unlock and quietly retry next tick
            consecutiveNoMatchRef.current = 0;
            scanInProgressRef.current = false;
            return;
          }
        } catch {
          // FaceDetector threw (video not ready, API unavailable, etc.) — fall through
        }
      }

      // ── Send frame to backend ────────────────────────────────────────
      setIsProcessing(true);
      const result = await attendanceService
        .processFrame(activeSessionId, captured.imageUrl)
        .catch(() => null);
      setIsProcessing(false);

      if (!result) {
        // Network / server error — unlock and retry next tick
        scanInProgressRef.current = false;
        return;
      }

      // Normalise the status field (new backend sends result.status;
      // fall back to message string for backward compatibility)
      const status: string =
        result.status ??
        (result.message === "Attendance recorded"
          ? "success"
          : result.message === "Attendance already marked"
            ? "already_marked"
            : "not_recognized");

      switch (status) {
        case "no_face":
          // Backend confirmed no face — unlock and retry next tick
          consecutiveNoMatchRef.current = 0;
          scanInProgressRef.current = false;
          break;

        case "partial_face":
          // Face occluded (mask / sunglasses / hand) — leave mutex locked;
          // cleanup unlocks when auto-reset restores waiting_for_face
          setRecognitionResult("partial_face");
          break;

        case "success":
          consecutiveNoMatchRef.current = 0;
          // Leave mutex locked — cleanup unlocks for next student's loop
          setRecognitionResult("success", {
            name: result.full_name ?? result.student_number ?? `Session ${activeSessionId}`,
            courseName: activeCourseName || "Live attendance session",
            sessionId: `SES-${activeSessionId}`,
            status: result.attendance_status ?? "Present",
            confidence: result.confidence ?? 0,
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
          });
          break;

        case "already_marked":
          consecutiveNoMatchRef.current = 0;
          // Leave mutex locked — cleanup unlocks for next student's loop
          setRecognitionResult("already_marked", {
            name: result.full_name ?? result.student_number ?? `Session ${activeSessionId}`,
            courseName: activeCourseName || "Live attendance session",
            sessionId: `SES-${activeSessionId}`,
            status: "Already Present",
            confidence: result.confidence ?? 0,
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
          });
          break;

        case "not_approved":
          // Unapproved student — treat exactly like an unknown face.
          // The backend already masks this as "not_recognized" but handle
          // the old status string here too for safety.
          consecutiveNoMatchRef.current += 1;
          if (consecutiveNoMatchRef.current >= 4) {
            consecutiveNoMatchRef.current = 0;
            setRecognitionResult("failed");
          } else {
            scanInProgressRef.current = false;
          }
          break;

        default:
          // not_recognized — face is visible but student is unknown.
          // After 4 consecutive misses show "failed"; otherwise unlock
          // and let the next tick try again.
          consecutiveNoMatchRef.current += 1;
          if (consecutiveNoMatchRef.current >= 4) {
            consecutiveNoMatchRef.current = 0;
            // Leave mutex locked — cleanup unlocks for next student's loop
            setRecognitionResult("failed");
          } else {
            scanInProgressRef.current = false;
          }
          break;
      }
    };

    // Brief startup delay so the camera can fully initialise before we send
    // the first frame, then tick every 900 ms.
    const startTimer = setTimeout(doScan, 600);
    const intervalId = setInterval(doScan, 900);

    return () => {
      clearTimeout(startTimer);
      clearInterval(intervalId);
      // Reset mutex and processing indicator for the next loop
      scanInProgressRef.current = false;
      setIsProcessing(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionState, activeSessionId]);

  // ── Auto-reset to waiting_for_face after showing any result ────────────
  useEffect(() => {
    const resultStates = [
      "success",
      "failed",
      "already_marked",
      "low_light",
      "partial_face",
    ];
    if (!resultStates.includes(sessionState)) return;
    const timer = setTimeout(() => waitForFace(), 2500);
    return () => clearTimeout(timer);
  }, [sessionState, waitForFace]);

  // ── Voice announcement on successful recognition ─────────────────────────
  useEffect(() => {
    if (sessionState !== "success" || !recognizedUser) return;
    if (!("speechSynthesis" in window)) return;

    // Extract first name from the full name stored in recognizedUser.name
    const firstName = recognizedUser.name.split(" ")[0] || recognizedUser.name;
    const utterance = new SpeechSynthesisUtterance(`Thank you, ${firstName}`);

    // ── Pick a male voice ────────────────────────────────────────────────
    // Prefer a deep male English voice. Voices load asynchronously so we
    // check twice: once from what's already cached, then after the list
    // populates (onvoiceschanged). The first call is enough on most systems;
    // the fallback fires on Chrome where voices aren't ready immediately.
    const applyVoice = (u: SpeechSynthesisUtterance) => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length === 0) return; // not yet populated
      // Ranked preference: Google UK English Male first (deep, clear Chrome voice),
      // then platform fallbacks (David on Windows, Alex on macOS), then any
      // other identifiably male English voice.
      const male =
        voices.find((v) => /google uk english male/i.test(v.name)) ??
        voices.find((v) => /david/i.test(v.name)) ??
        voices.find((v) => /google us english/i.test(v.name)) ??
        voices.find((v) => /alex/i.test(v.name)) ??
        voices.find((v) => /male/i.test(v.name) && !/female/i.test(v.name)) ??
        voices.find((v) => v.lang.startsWith("en") && !/female|zira|susan|samantha|victoria|karen|moira|tessa|fiona|veena|allison|ava|nicky/i.test(v.name)) ??
        null;
      if (male) u.voice = male;
    };

    applyVoice(utterance);
    // Fallback: re-try once voices have loaded (Chrome defers this)
    if (!utterance.voice) {
      const onReady = () => {
        applyVoice(utterance);
        window.speechSynthesis.removeEventListener("voiceschanged", onReady);
      };
      window.speechSynthesis.addEventListener("voiceschanged", onReady);
    }

    // ── Acoustic settings ────────────────────────────────────────────────
    // pitch < 1.0  → deeper / more masculine tone
    // rate  < 1.0  → slightly slower so the name is clearly heard
    // volume = 1.0 → maximum the browser API allows (system volume controls
    //                the rest — turn speakers up for extra loudness)
    utterance.rate   = 0.92;
    utterance.pitch  = 0.75; // 0.75 gives a noticeably deep male voice
    utterance.volume = 1.0;

    // Cancel any ongoing speech before starting the new one
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);

    // Do NOT cancel in cleanup — the state changes to waiting_for_face after
    // 2500ms (auto-reset), which would fire this cleanup and cut the voice off
    // mid-sentence before the name is spoken. Let the utterance play to its
    // natural end; the cancel() above ensures no overlap with the next person.
  }, [sessionState, recognizedUser]);

  // ── Derived booleans ────────────────────────────────────────────────────
  const isResult = [
    "success",
    "failed",
    "already_marked",
    "low_light",
    "partial_face",
  ].includes(sessionState);

  // ── Frame border + glow — each outcome has a distinct colour ───────────
  const borderClass =
    sessionState === "success"
      ? "border-emerald-500 shadow-[0_0_50px_rgba(16,185,129,0.45)]"
      : sessionState === "failed"
        ? "border-rose-500 shadow-[0_0_50px_rgba(244,63,94,0.45)]"
        : sessionState === "already_marked"
          ? "border-yellow-400 shadow-[0_0_50px_rgba(250,204,21,0.45)]"
          : sessionState === "partial_face"
            ? "border-orange-500 shadow-[0_0_50px_rgba(249,115,22,0.45)]"
            : sessionState === "low_light"
              ? "border-amber-400 shadow-[0_0_50px_rgba(251,191,36,0.45)]"
              : // waiting_for_face (scanning)
                "border-primary shadow-[0_0_50px_rgba(37,99,235,0.4)]";

  return (
    <motion.div
      key="scanning-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full h-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-center z-10"
    >
      {/* ── LEFT PANEL — Instructions ──────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="lg:col-span-3 order-2 lg:order-1"
      >
        <Card className="glass-card shadow-xl shadow-primary/5 bg-white dark:bg-white/5 border-gray-200 dark:border-white/5">
          <CardContent className="p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Face Recognition
            </h3>
            <ul className="space-y-4 text-gray-600 dark:text-gray-400 text-sm">
              <li className="flex items-start gap-3">
                <Info className="shrink-0 text-primary mt-0.5" size={16} />
                Look directly into the camera
              </li>
              <li className="flex items-start gap-3">
                <Info className="shrink-0 text-primary mt-0.5" size={16} />
                Keep your full face visible — no masks, sunglasses, or hands covering your face
              </li>
              <li className="flex items-start gap-3">
                <Info className="shrink-0 text-primary mt-0.5" size={16} />
                Stand in a well-lit area
              </li>
              <li className="flex items-start gap-3">
                <Info className="shrink-0 text-primary mt-0.5" size={16} />
                Hold still while scanning
              </li>
            </ul>

            {/* Colour legend */}
            <div className="mt-6 space-y-2 border-t border-gray-200 dark:border-white/10 pt-4">
              <p className="text-xs uppercase tracking-wider text-gray-400 mb-3">Status colours</p>
              {[
                { colour: "bg-emerald-500", label: "Recognized" },
                { colour: "bg-rose-500", label: "Not recognized" },
                { colour: "bg-yellow-400", label: "Already marked" },
                { colour: "bg-orange-500", label: "Face obscured" },
                { colour: "bg-amber-400", label: "Low light" },
              ].map(({ colour, label }) => (
                <div key={label} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                  <span className={cn("w-2.5 h-2.5 rounded-full shrink-0", colour)} />
                  {label}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── CENTER PANEL — Scanner frame ───────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1, type: "spring", stiffness: 200 }}
        className="lg:col-span-6 order-1 flex flex-col justify-center items-center gap-6"
      >
        <div
          className={cn(
            "relative w-full max-w-sm aspect-[3/4] rounded-[2.5rem] flex flex-col items-center justify-center overflow-hidden transition-all duration-500",
            "bg-[#0B0F19]/80 backdrop-blur-xl border-4",
            borderClass,
          )}
        >
          {/* Laser scan line — visible while processing a frame */}
          {isProcessing && (
            <motion.div
              animate={{ top: ["0%", "100%", "0%"] }}
              transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
              className="absolute left-0 w-full h-1 bg-primary shadow-[0_0_20px_rgba(37,99,235,1)] z-20"
            />
          )}

          {/* Inner target ring — w-48 h-48 = 192px × 192px */}
          <div className="relative z-10 w-48 h-48 rounded-full border border-white/20 flex items-center justify-center p-2">
            <div className="w-full h-full rounded-full overflow-hidden relative bg-[#111827]">
              {/* Camera feed */}
              {cameraError ? (
                <div className="flex items-center justify-center h-full w-full bg-gray-900 text-xs text-red-400 text-center px-4">
                  {cameraError}
                </div>
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={cn(
                    "w-full h-full object-cover transition-all duration-500",
                    isResult && "grayscale opacity-80",
                  )}
                />
              )}

              {/* Grid overlay while waiting */}
              <AnimatePresence>
                {!isResult && !cameraError && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.4 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 mix-blend-screen pointer-events-none"
                    style={{
                      backgroundImage:
                        "linear-gradient(rgba(37,99,235,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(37,99,235,0.2) 1px, transparent 1px)",
                      backgroundSize: "24px 24px",
                    }}
                  />
                )}
              </AnimatePresence>
            </div>

            {/* Animated progress ring — sized for 192px (cx/cy=96, r=94) */}
            <svg className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none">
              <circle cx="96" cy="96" r="94" fill="none" strokeWidth="4" className="stroke-gray-200 dark:stroke-white/10" />
              {isProcessing && (
                <circle
                  cx="96"
                  cy="96"
                  r="94"
                  fill="none"
                  strokeWidth="4"
                  strokeLinecap="round"
                  className="stroke-primary"
                  strokeDasharray="590"
                  strokeDashoffset="295"
                >
                  <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from="90 96 96"
                    to="450 96 96"
                    dur="1s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </svg>

            {/* Result overlay icon */}
            <AnimatePresence>
              {isResult && (
                <motion.div
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.5, opacity: 0 }}
                  className="absolute inset-0 rounded-full flex items-center justify-center bg-black/60 backdrop-blur-sm z-30"
                >
                  {sessionState === "success" && (
                    <CheckCircle2 className="text-emerald-500 drop-shadow-[0_0_15px_rgba(16,185,129,0.8)]" size={80} />
                  )}
                  {sessionState === "failed" && (
                    <XCircle className="text-rose-500 drop-shadow-[0_0_15px_rgba(244,63,94,0.8)]" size={80} />
                  )}
                  {sessionState === "already_marked" && (
                    <Info className="text-yellow-400 drop-shadow-[0_0_15px_rgba(250,204,21,0.8)]" size={80} />
                  )}
                  {sessionState === "partial_face" && (
                    <AlertTriangle className="text-orange-500 drop-shadow-[0_0_15px_rgba(249,115,22,0.8)]" size={80} />
                  )}
                  {sessionState === "low_light" && (
                    <Sun className="text-amber-400 drop-shadow-[0_0_15px_rgba(251,191,36,0.8)]" size={80} />
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Status text + countdown bar */}
          <div className="absolute bottom-12 text-center w-full px-6">
            <div className="bg-white dark:bg-white/5 backdrop-blur-md px-6 py-2 rounded-full inline-flex border border-gray-200 dark:border-white/10 shadow-lg mb-4">
              <span className="font-medium text-gray-900 dark:text-white tracking-wide">
                {sessionState === "waiting_for_face" && (isProcessing ? "Scanning face…" : "Waiting for face…")}
                {sessionState === "success" && "Recognized ✓"}
                {sessionState === "failed" && "Not Recognized"}
                {sessionState === "already_marked" && "Already Marked"}
                {sessionState === "partial_face" && "Full Face Required"}
                {sessionState === "low_light" && "Improve Lighting"}
              </span>
            </div>

            {/* Auto-reset countdown bar for result states */}
            {isResult && (
              <motion.div
                initial={{ width: "100%" }}
                animate={{ width: "0%" }}
                transition={{ duration: 2.5, ease: "linear" }}
                className={cn(
                  "h-0.5 rounded-full mx-auto max-w-[120px]",
                  sessionState === "success" && "bg-emerald-500",
                  sessionState === "failed" && "bg-rose-500",
                  sessionState === "already_marked" && "bg-yellow-400",
                  sessionState === "partial_face" && "bg-orange-500",
                  sessionState === "low_light" && "bg-amber-400",
                )}
              />
            )}
          </div>
        </div>

        {/* End Session button */}
        <Button
          onClick={() => setShowEndConfirm(true)}
          variant="secondary"
          className="bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 px-8 py-2.5 rounded-full font-semibold flex items-center gap-2 transition-all"
        >
          <XCircle size={18} />
          End Session
        </Button>
      </motion.div>

      {/* ── RIGHT PANEL — Status & Results ─────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="lg:col-span-3 order-3 self-stretch"
      >
        <Card className="glass-card shadow-xl shadow-primary/5 bg-white dark:bg-white/5 border-gray-200 dark:border-white/5 h-full min-h-[320px]">
          <CardContent className="p-6 flex flex-col h-full">
            <div className="mb-8">
              <h3 className="text-sm text-gray-600 dark:text-gray-400 font-medium mb-4">
                Live Status
              </h3>

              <div className="space-y-3">
                {/* Waiting / scanning */}
                {!isResult && (
                  <div className="flex items-center gap-3 text-primary">
                    <div className={cn("w-1.5 h-1.5 rounded-full bg-current", isProcessing && "animate-pulse")} />
                    <span className="text-sm font-medium">
                      {isProcessing ? "Processing frame…" : "Waiting for face…"}
                    </span>
                  </div>
                )}

                {/* Success */}
                {sessionState === "success" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 text-emerald-500 bg-emerald-500/10 px-3 py-2 rounded-lg border border-emerald-500/20"
                  >
                    <CheckCircle2 size={16} />
                    <span className="text-sm font-medium">Recognized Successfully</span>
                  </motion.div>
                )}

                {/* Failed */}
                {sessionState === "failed" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 text-rose-500 bg-rose-500/10 px-3 py-2 rounded-lg border border-rose-500/20"
                  >
                    <XCircle size={16} />
                    <span className="text-sm font-medium">Recognition Failed</span>
                  </motion.div>
                )}

                {/* Already marked */}
                {sessionState === "already_marked" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 text-yellow-500 bg-yellow-400/10 px-3 py-2 rounded-lg border border-yellow-400/20"
                  >
                    <Info size={16} />
                    <span className="text-sm font-medium">Attendance Already Marked</span>
                  </motion.div>
                )}

                {/* Partial face */}
                {sessionState === "partial_face" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-1 text-orange-500 bg-orange-500/10 px-3 py-2 rounded-lg border border-orange-500/20"
                  >
                    <div className="flex items-center gap-2">
                      <AlertTriangle size={16} />
                      <span className="text-sm font-medium">Full Face Required</span>
                    </div>
                    <span className="text-xs opacity-80">
                      Remove mask, sunglasses, or hands from face
                    </span>
                  </motion.div>
                )}

                {/* Low light */}
                {sessionState === "low_light" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-1 text-amber-500 bg-amber-400/10 px-3 py-2 rounded-lg border border-amber-400/20"
                  >
                    <div className="flex items-center gap-2">
                      <Sun size={16} />
                      <span className="text-sm font-medium">Poor Lighting</span>
                    </div>
                    <span className="text-xs opacity-80">
                      Stand in a well-lit area
                    </span>
                  </motion.div>
                )}
              </div>
            </div>

            {/* Recognized user details */}
            <div className="mt-auto border-t border-gray-200 dark:border-white/5 pt-6 overflow-hidden">
              <h3 className="text-sm text-gray-600 dark:text-gray-400 font-medium mb-4">
                Last Scan
              </h3>

              {recognizedUser ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-500 text-sm whitespace-nowrap">Name</span>
                    <span className="text-gray-900 dark:text-white font-medium text-right truncate" title={recognizedUser.name}>
                      {recognizedUser.name}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-500 text-sm whitespace-nowrap">Course</span>
                    <span className="text-gray-900 dark:text-gray-300 text-right truncate text-sm" title={recognizedUser.courseName}>
                      {recognizedUser.courseName}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-500 text-sm whitespace-nowrap">Session</span>
                    <span className="text-gray-900 dark:text-gray-300 text-right text-sm">
                      {recognizedUser.sessionId}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-500 text-sm whitespace-nowrap">Confidence</span>
                    <span className="text-gray-900 dark:text-gray-300 text-right text-sm">
                      {(recognizedUser.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-500 text-sm whitespace-nowrap">Time</span>
                    <span className="text-gray-900 dark:text-gray-300 text-sm text-right">
                      {recognizedUser.time}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4 mt-2">
                    <span className="text-gray-500 text-sm whitespace-nowrap">Status</span>
                    <Badge
                      variant={sessionState === "already_marked" ? "warning" : "success"}
                      className="ml-auto"
                    >
                      {recognizedUser.status}
                    </Badge>
                  </div>
                </motion.div>
              ) : (
                <div className="text-center py-4 text-gray-500 text-sm">
                  Waiting for scan…
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── End Session Confirmation Modal ─────────────────────────────── */}
      <Modal
        isOpen={showEndConfirm}
        onClose={() => setShowEndConfirm(false)}
        title="End Session"
        className="max-w-sm"
      >
        <div className="space-y-6 pt-2">
          <p className="text-gray-600 dark:text-gray-400">
            Ending the session will automatically mark all students who haven't
            scanned as <strong>Absent</strong>. Continue?
          </p>
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-100 dark:border-white/10">
            <Button variant="ghost" onClick={() => setShowEndConfirm(false)}>
              Cancel
            </Button>
            <Button
              className="bg-red-600 hover:bg-red-700 text-white"
              onClick={handleEndSession}
            >
              End Session
            </Button>
          </div>
        </div>
      </Modal>
    </motion.div>
  );
}
