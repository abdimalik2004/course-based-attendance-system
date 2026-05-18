import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, XCircle, Info } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Modal } from "@/components/ui/Modal";
import { useAttendanceStore } from "@/store/useAttendanceStore";
import attendanceService from "@/services/attendanceService";
import { cn } from "@/utils/cn";

export default function ScannerInterface() {
  const {
    sessionState,
    scanningProgress,
    recognizedUser,
    activeSessionId,
    activeCourseName,
    waitForFace,
    startScanning,
    setFaceDetected,
    setScanningProgress,
    setRecognitionResult,
    resetSession,
  } = useAttendanceStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string>("");
  const [showEndConfirm, setShowEndConfirm] = useState(false);

  const captureFrame = async () => {
    if (!videoRef.current || !activeSessionId) return null;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9);
  };

  const handleEndSession = async () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (activeSessionId) {
      await attendanceService.endSession(activeSessionId).catch(() => null);
    }
    resetSession();
  };

  // Start/Stop Camera
  useEffect(() => {
    if (sessionState !== "idle") {
      if (!stream && !cameraError) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((mediaStream) => {
            setStream(mediaStream);
          })
          .catch((err) => {
            console.error("Error accessing camera:", err);
            setCameraError("Camera permission denied or not found");
          });
      }
    } else {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }
      setCameraError("");
    }
  }, [sessionState, stream, cameraError]);

  // Ensure video element gets the stream when it mounts/updates
  useEffect(() => {
    if (videoRef.current && stream && videoRef.current.srcObject !== stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream, sessionState]);

  // Backend-driven scanning logic
  useEffect(() => {
    if (sessionState === "scanning") {
      let cancelled = false;
      let progress = 0;
      let attempts = 0;

      const interval = setInterval(async () => {
        if (cancelled || !activeSessionId) return;

        progress = Math.min(95, progress + 20);
        setScanningProgress(progress);
        attempts += 1;

        const image = await captureFrame();
        if (!image) {
          if (attempts >= 5) {
            clearInterval(interval);
            setRecognitionResult("failed");
          }
          return;
        }

        const result = await attendanceService
          .processFrame(activeSessionId, image)
          .catch(() => null);
        if (!result) {
          if (attempts >= 5) {
            clearInterval(interval);
            setRecognitionResult("failed");
          }
          return;
        }

        if (result.ok && result.message === "Attendance recorded") {
          clearInterval(interval);
          setScanningProgress(100);
          setRecognitionResult("success", {
            name: result.student_number ?? `Session ${activeSessionId}`,
            courseName: activeCourseName || "Live attendance session",
            sessionId: `SES-${activeSessionId}`,
            status: "Present",
            confidence: result.confidence ?? 0,
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
          });
          return;
        }

        if (result.ok && result.message === "Attendance already marked") {
          clearInterval(interval);
          setScanningProgress(100);
          setRecognitionResult("already_marked", {
            name: result.student_number ?? `Session ${activeSessionId}`,
            courseName: activeCourseName || "Live attendance session",
            sessionId: `SES-${activeSessionId}`,
            status: "Already Present",
            confidence: result.confidence ?? 0,
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
          });
          return;
        }

        if (attempts >= 5) {
          clearInterval(interval);
          setRecognitionResult("failed");
        }
      }, 1200);

      return () => {
        cancelled = true;
        clearInterval(interval);
      };
    }
  }, [
    sessionState,
    activeSessionId,
    setScanningProgress,
    setRecognitionResult,
  ]);

  // Handle transition from starting to waiting_for_face
  useEffect(() => {
    if (sessionState === "starting") {
      const timer = setTimeout(() => {
        waitForFace();
      }, 1500); // 1.5s buffer to show the UI
      return () => clearTimeout(timer);
    }
  }, [sessionState, waitForFace]);

  // Simulate waiting for a face before backend scanning begins
  useEffect(() => {
    if (sessionState === "waiting_for_face") {
      const waitTimer = setTimeout(
        () => {
          setFaceDetected();
        },
        Math.floor(Math.random() * 3000) + 2000,
      );
      return () => clearTimeout(waitTimer);
    }
  }, [sessionState, setFaceDetected]);

  // Simulate face_detected -> scanning
  useEffect(() => {
    if (sessionState === "face_detected") {
      const timer = setTimeout(() => {
        startScanning();
      }, 1200);
      return () => clearTimeout(timer);
    }
  }, [sessionState, startScanning]);

  // Auto-reset for the next student
  useEffect(() => {
    const finalStates = [
      "success",
      "failed",
      "already_marked",
      "low_light",
      "partial_face",
    ];
    if (finalStates.includes(sessionState)) {
      const resetTimer = setTimeout(() => {
        waitForFace();
      }, 6000);

      return () => clearTimeout(resetTimer);
    }
  }, [sessionState, waitForFace]);

  const isScanningActive =
    sessionState === "starting" ||
    sessionState === "waiting_for_face" ||
    sessionState === "face_detected" ||
    sessionState === "scanning";
  const isComplete = [
    "success",
    "failed",
    "already_marked",
    "low_light",
    "partial_face",
  ].includes(sessionState);
  const showFace =
    sessionState === "face_detected" ||
    sessionState === "scanning" ||
    isComplete;

  return (
    <motion.div
      key="scanning-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full h-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-center z-10"
    >
      {/* LEFT PANEL - Instructions */}
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
                Hold still while scanning
              </li>
              <li className="flex items-start gap-3">
                <Info className="shrink-0 text-primary mt-0.5" size={16} />
                Ensure proper lighting on your face
              </li>
            </ul>
          </CardContent>
        </Card>
      </motion.div>

      {/* CENTER PANEL - The Scanner */}
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
            isScanningActive
              ? "border-primary shadow-[0_0_50px_rgba(37,99,235,0.4)]"
              : sessionState === "success"
                ? "border-emerald-500 shadow-[0_0_50px_rgba(16,185,129,0.4)]"
                : sessionState === "low_light" ||
                    sessionState === "partial_face" ||
                    sessionState === "already_marked"
                  ? "border-yellow-500 shadow-[0_0_50px_rgba(234,179,8,0.4)]"
                  : "border-rose-500 shadow-[0_0_50px_rgba(244,63,94,0.4)]",
          )}
        >
          {/* Laser scan line effect */}
          {sessionState === "scanning" && (
            <motion.div
              animate={{ top: ["0%", "100%", "0%"] }}
              transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
              className="absolute left-0 w-full h-1 bg-primary shadow-[0_0_20px_rgba(37,99,235,1)] z-20"
            />
          )}

          {/* Inner Target Frame */}
          <div className="relative z-10 w-48 h-48 rounded-full border border-white/20 flex items-center justify-center p-2">
            <div className="w-full h-full rounded-full overflow-hidden relative bg-[#111827]">
              {/* Camera Feed */}
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
                    isComplete && "grayscale opacity-80",
                    !showFace && "opacity-40 blur-sm",
                  )}
                />
              )}

              <AnimatePresence>
                {!showFace && !cameraError && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className={cn(
                      "absolute inset-0 transition-all duration-500 mix-blend-screen",
                      sessionState === "waiting_for_face" && "opacity-60",
                    )}
                    style={{
                      backgroundImage:
                        "linear-gradient(rgba(37,99,235,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(37,99,235,0.2) 1px, transparent 1px)",
                      backgroundSize: "20px 20px",
                    }}
                  />
                )}
              </AnimatePresence>
            </div>

            {/* Progress Ring Overlay */}
            <svg className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none">
              <circle
                cx="96"
                cy="96"
                r="94"
                fill="none"
                strokeWidth="4"
                className="stroke-gray-200 dark:stroke-white/10"
              />
              {isScanningActive && (
                <circle
                  cx="96"
                  cy="96"
                  r="94"
                  fill="none"
                  strokeWidth="4"
                  strokeLinecap="round"
                  className="stroke-primary transition-all duration-300"
                  strokeDasharray="590"
                  strokeDashoffset={590 - (590 * scanningProgress) / 100}
                />
              )}
            </svg>

            {/* Result Overlays */}
            <AnimatePresence>
              {isComplete && (
                <motion.div
                  initial={{ scale: 0.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className={cn(
                    "absolute inset-0 rounded-full flex items-center justify-center bg-black/60 backdrop-blur-sm shadow-[inset_0_0_30px_rgba(0,0,0,0.5)] z-30",
                  )}
                >
                  {sessionState === "success" ? (
                    <CheckCircle2
                      className="text-emerald-500 drop-shadow-[0_0_15px_rgba(16,185,129,0.8)]"
                      size={64}
                    />
                  ) : sessionState === "already_marked" ||
                    sessionState === "low_light" ||
                    sessionState === "partial_face" ? (
                    <Info
                      className="text-yellow-500 drop-shadow-[0_0_15px_rgba(234,179,8,0.8)]"
                      size={64}
                    />
                  ) : (
                    <XCircle
                      className="text-rose-500 drop-shadow-[0_0_15px_rgba(244,63,94,0.8)]"
                      size={64}
                    />
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Status Text Area */}
          <div className="absolute bottom-12 text-center w-full px-6">
            <div className="bg-white dark:bg-white/5 backdrop-blur-md px-6 py-2 rounded-full inline-flex border border-gray-200 dark:border-white/10 shadow-lg mb-4">
              <span className="font-medium text-gray-900 dark:text-white tracking-wide">
                {sessionState === "starting"
                  ? "Initializing..."
                  : sessionState === "waiting_for_face"
                    ? "Waiting for face..."
                    : sessionState === "face_detected"
                      ? "Face detected"
                      : sessionState === "scanning"
                        ? "Scanning face..."
                        : sessionState === "success"
                          ? "Recognized"
                          : sessionState === "already_marked"
                            ? "Already Marked"
                            : sessionState === "low_light"
                              ? "Low Light"
                              : sessionState === "partial_face"
                                ? "Partial Face"
                                : "Failed"}
              </span>
            </div>
            {isScanningActive && (
              <div className="flex items-center justify-center gap-4 text-xs font-mono text-primary animate-pulse">
                <span className="w-8 text-right">{scanningProgress}%</span>
                <div className="h-[1px] w-12 bg-primary/50" />
                <span>PROCESSING</span>
              </div>
            )}
          </div>
        </div>

        {/* End Session Button */}
        <Button
          onClick={() => setShowEndConfirm(true)}
          variant="secondary"
          className="bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20 px-8 py-2.5 rounded-full font-semibold flex items-center gap-2 transition-all"
        >
          <XCircle size={18} />
          End Session
        </Button>
      </motion.div>

      {/* RIGHT PANEL - Status & Results */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="lg:col-span-3 order-3"
      >
        <Card className="glass-card shadow-xl shadow-primary/5 bg-white dark:bg-white/5 border-gray-200 dark:border-white/5 h-full min-h-[320px]">
          <CardContent className="p-6 flex flex-col h-full">
            <div className="mb-8">
              <h3 className="text-sm text-gray-600 dark:text-gray-400 font-medium mb-4">
                Status
              </h3>

              <div className="space-y-3">
                <div
                  className={cn(
                    "flex items-center gap-3 transition-colors text-primary",
                  )}
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-current" />
                  <span className="text-sm font-medium">
                    {sessionState === "waiting_for_face"
                      ? "Waiting across camera frame..."
                      : sessionState === "face_detected"
                        ? "Locking on face..."
                        : "Scanning..."}
                  </span>
                </div>

                {sessionState === "success" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 text-emerald-500 bg-emerald-500/10 px-3 py-2 rounded-lg border border-emerald-500/20"
                  >
                    <CheckCircle2 size={16} />
                    <span className="text-sm font-medium">
                      Recognized Successfully
                    </span>
                  </motion.div>
                )}

                {sessionState === "failed" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 text-rose-500 bg-rose-500/10 px-3 py-2 rounded-lg border border-rose-500/20"
                  >
                    <XCircle size={16} />
                    <span className="text-sm font-medium">
                      Recognition Failed
                    </span>
                  </motion.div>
                )}

                {sessionState === "already_marked" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-start gap-1 text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 px-3 py-2 rounded-lg border border-yellow-500/20"
                  >
                    <div className="flex items-center gap-2">
                      <Info size={16} />
                      <span className="text-sm font-medium">
                        Already Marked
                      </span>
                    </div>
                  </motion.div>
                )}

                {sessionState === "low_light" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-start gap-1 text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 px-3 py-2 rounded-lg border border-yellow-500/20"
                  >
                    <div className="flex items-center gap-2">
                      <Info size={16} />
                      <span className="text-sm font-medium">Poor Lighting</span>
                    </div>
                    <span className="text-xs opacity-80 mt-1">
                      Stand probably in a good place
                    </span>
                  </motion.div>
                )}

                {sessionState === "partial_face" && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-start gap-1 text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 px-3 py-2 rounded-lg border border-yellow-500/20"
                  >
                    <div className="flex items-center gap-2">
                      <Info size={16} />
                      <span className="text-sm font-medium">
                        Partial Face Detected
                      </span>
                    </div>
                    <span className="text-xs opacity-80 mt-1">
                      Full Face Required
                    </span>
                  </motion.div>
                )}
              </div>
            </div>

            <div className="mt-auto border-t border-gray-200 dark:border-white/5 pt-6 overflow-hidden">
              <h3 className="text-sm text-gray-600 dark:text-gray-400 font-medium mb-4">
                User Info
              </h3>

              {recognizedUser ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-3"
                >
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-600 dark:text-gray-500 text-sm whitespace-nowrap">
                      Name
                    </span>
                    <span
                      className="text-gray-900 dark:text-white font-medium text-right truncate"
                      title={recognizedUser.name}
                    >
                      {recognizedUser.name}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-600 dark:text-gray-500 text-sm whitespace-nowrap">
                      Course
                    </span>
                    <span
                      className="text-gray-900 dark:text-gray-300 text-right truncate text-sm"
                      title={recognizedUser.courseName}
                    >
                      {recognizedUser.courseName}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-600 dark:text-gray-500 text-sm whitespace-nowrap">
                      Session ID
                    </span>
                    <span className="text-gray-900 dark:text-gray-300 text-right text-sm">
                      {recognizedUser.sessionId}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-600 dark:text-gray-500 text-sm whitespace-nowrap">
                      Confidence
                    </span>
                    <span className="text-gray-900 dark:text-gray-300 text-right text-sm">
                      {(recognizedUser.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4">
                    <span className="text-gray-600 dark:text-gray-500 text-sm whitespace-nowrap">
                      Time
                    </span>
                    <span className="text-gray-900 dark:text-gray-300 text-sm text-right">
                      {recognizedUser.time}
                    </span>
                  </div>
                  <div className="flex justify-between items-center gap-4 mt-2">
                    <span className="text-gray-600 dark:text-gray-500 text-sm whitespace-nowrap">
                      Status
                    </span>
                    <Badge
                      variant={
                        sessionState === "already_marked"
                          ? "warning"
                          : "success"
                      }
                      className="ml-auto"
                    >
                      {recognizedUser.status}
                    </Badge>
                  </div>
                </motion.div>
              ) : (
                <div className="text-center py-4 text-gray-600 text-sm">
                  Waiting for scan...
                </div>
              )}
            </div>

            {/* Reset Button (only shown on complete) */}
            <AnimatePresence>
              {isComplete && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto", marginTop: 24 }}
                >
                  <Button
                    variant="secondary"
                    className="w-full"
                    onClick={startSession}
                  >
                    Scan Next User
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </motion.div>

      {/* End Session Confirmation Modal */}
      <Modal
        isOpen={showEndConfirm}
        onClose={() => setShowEndConfirm(false)}
        title="End Session"
        className="max-w-sm"
      >
        <div className="space-y-6 pt-2">
          <p className="text-gray-600 dark:text-gray-400">
            Do you want to end the session?
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
