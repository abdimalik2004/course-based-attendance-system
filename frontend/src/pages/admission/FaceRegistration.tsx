import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Camera,
  StopCircle,
  RefreshCw,
  Save,
  CheckCircle2,
  AlertTriangle,
  Play,
  X,
  UserX,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Modal } from "@/components/ui/Modal";
import { useAdmissionStore } from "@/store/useAdmissionStore";
import admissionService, { type FacultyDto } from "@/services/admissionService";
import { cn } from "@/utils/cn";

// Minimum cosine similarity to consider a face the same person as the anchor
const PERSON_TRACKING_THRESHOLD = 0.72;

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // vectors are already unit-normalised by the backend
}

type CaptureStatus =
  | "ready"
  | "initializing"
  | "camera_ready"
  | "capturing"
  | "paused"
  | "completed";

// Person tracking state:
//  "none"         — no anchor set yet (before first face)
//  "anchor_set"   — anchor person is in the frame (OK to capture)
//  "wrong_person" — anchor person left; a different person is now alone in the frame
type PersonTracking = "none" | "anchor_set" | "wrong_person";

export default function FaceRegistration() {
  const { fetchAdmissionData } = useAdmissionStore();

  useEffect(() => {
    void fetchAdmissionData();
  }, [fetchAdmissionData]);

  useEffect(() => {
    const loadFacultyCodes = async () => {
      try {
        const data = await admissionService.listFaculties();
        setFacultyOptions(data);
      } catch {
        // Faculty dropdown will stay empty; user will see no options
      }
    };

    void loadFacultyCodes();
  }, []);

  // Step 1 Form State
  const [faculty, setFaculty] = useState("");
  const [studentId, setStudentId] = useState("");
  const [photoCount, setPhotoCount] = useState(30);
  const [cameraIndex, setCameraIndex] = useState(0);
  const [facultyOptions, setFacultyOptions] = useState<FacultyDto[]>([]);

  // Modals & Process State
  const [isCaptureModalOpen, setIsCaptureModalOpen] = useState(false);
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);
  const [status, setStatus] = useState<CaptureStatus>("ready");
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [capturedCount, setCapturedCount] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  // Save & Train result state
  const [formError, setFormError] = useState<string | null>(null);
  const [saveErrorModal, setSaveErrorModal] = useState<{ title: string; message: string } | null>(null);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<
    "queued" | "running" | "succeeded" | "failed" | null
  >(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  // Set to true when backend returns 409 (student already has images)
  const [needsOverwrite, setNeedsOverwrite] = useState(false);

  // Video & Capture Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const imagesRef = useRef<string[]>([]);

  // Face Detection State (driven by real backend calls)
  const [multipleFaces, setMultipleFaces] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [personTracking, setPersonTracking] = useState<PersonTracking>("none");

  // Refs for person-identity tracking (must survive re-renders without triggering effects)
  const anchorEmbeddingRef = useRef<number[] | null>(null);
  const hadMultipleFacesRef = useRef(false);       // true while 2+ faces were in frame
  const detectInFlightRef = useRef(false);         // guard against overlapping API calls
  // Mirror of personTracking state as a ref so runFaceDetection stays stable (no deps churn)
  const personTrackingRef = useRef<PersonTracking>("none");
  // Ref mirrors of face-detection booleans — used in captureFrame to avoid stale closures
  const multipleFacesRef = useRef(false);
  const faceDetectedRef = useRef(false);
  // "No Face Detected" is only shown after 10 s of continuous no-face (avoids flashing at start)
  const noFaceStartTimeRef = useRef<number | null>(null);
  const [noFaceOverlayVisible, setNoFaceOverlayVisible] = useState(false);

  // Poll training job status until it reaches a terminal state
  useEffect(() => {
    if (!trainingJobId) return;

    const poll = setInterval(async () => {
      try {
        const job = await admissionService.getTrainingJob(trainingJobId);
        setTrainingStatus(job.status);
        if (job.status === "succeeded" || job.status === "failed") {
          clearInterval(poll);
          if (job.status === "failed") {
            setTrainingError(job.error ?? "Training failed for an unknown reason.");
          }
        }
      } catch {
        // Swallow polling errors — next tick will retry
      }
    }, 3000);

    return () => clearInterval(poll);
  }, [trainingJobId]);

  // Handle WebRTC Stream
  const startCamera = async () => {
    try {
      let videoConstraint: MediaTrackConstraints | boolean = true;

      if (cameraIndex > 0) {
        // enumerateDevices() only returns real deviceIds after the browser has been
        // granted camera permission. Open a temporary stream first to unlock the
        // permission gate, then enumerate to find the right deviceId.
        try {
          const permStream = await navigator.mediaDevices.getUserMedia({ video: true });
          permStream.getTracks().forEach((t) => t.stop());

          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter((d) => d.kind === "videoinput");
          const deviceId = videoDevices[cameraIndex]?.deviceId;
          if (deviceId) {
            videoConstraint = { deviceId: { exact: deviceId } };
          }
        } catch {
          // Permission or enumeration failed — fall back to default camera
        }
      }

      const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch {
      setFormError("Could not access camera. Please check browser permissions and try again.");
      handleStopCapture();
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Capture Logic
  // Reads detection state from refs (not state) so the callback never has stale values —
  // this means auto-resume works the moment the original person returns, and multiple-face
  // detection stops a capture within the same tick it is detected.
  const captureFrame = useCallback(() => {
    if (
      videoRef.current &&
      canvasRef.current &&
      status === "capturing" &&
      !multipleFacesRef.current &&        // always fresh — no stale closure
      faceDetectedRef.current &&          // always fresh
      personTrackingRef.current !== "wrong_person"  // always fresh
    ) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL("image/png", 1.0);
        imagesRef.current.push(imageData);
        setCapturedCount(imagesRef.current.length);

        if (imagesRef.current.length >= photoCount) {
          setStatus("completed");
        }
      }
    }
  }, [status, photoCount]); // only status/photoCount can change the logic; face state is ref-driven

  // Stable setters — keep both state (for UI renders) and ref (for capture reads) in sync
  const updatePersonTracking = useCallback((next: PersonTracking) => {
    personTrackingRef.current = next;
    setPersonTracking(next);
  }, []);

  const updateFaceDetected = useCallback((val: boolean) => {
    faceDetectedRef.current = val;
    setFaceDetected(val);
  }, []);

  const updateMultipleFaces = useCallback((val: boolean) => {
    multipleFacesRef.current = val;
    setMultipleFaces(val);
  }, []);

  // Real face detection via backend — runs every 200ms while camera is live.
  // Uses personTrackingRef (not state) so the callback stays stable and the
  // detection interval is never torn-down/recreated on state changes.
  const runFaceDetection = useCallback(async () => {
    if (detectInFlightRef.current) return; // skip if previous call still in-flight
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    if (video.readyState < 2 || video.videoWidth === 0) return; // stream not ready

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw raw (unmirrored) frame — the backend model doesn't need the CSS mirror
    ctx.drawImage(video, 0, 0);

    // Higher quality for more reliable face detection
    const imageData = canvas.toDataURL("image/jpeg", 0.75);

    detectInFlightRef.current = true;
    try {
      const { face_count, embedding } = await admissionService.detectFaces(imageData);

      if (face_count === 0) {
        // ── No face in frame ──────────────────────────────────────────────────
        updateFaceDetected(false);
        updateMultipleFaces(false);
        // Start (or continue) the 10-second no-face timer
        if (noFaceStartTimeRef.current === null) {
          noFaceStartTimeRef.current = Date.now();
        } else if (Date.now() - noFaceStartTimeRef.current >= 10_000) {
          setNoFaceOverlayVisible(true);
        }
        // Don't touch personTracking — a blink / brief look-away shouldn't reset the anchor

      } else {
        // Face(s) present → cancel the no-face timer and hide the overlay immediately
        noFaceStartTimeRef.current = null;
        setNoFaceOverlayVisible(false);

        if (face_count > 1) {
          // ── Multiple faces ────────────────────────────────────────────────────
          // Update ref FIRST so captureFrame blocks on the very next tick
          updateFaceDetected(true);
          updateMultipleFaces(true);
          hadMultipleFacesRef.current = true;

        } else {
          // ── Exactly one face ──────────────────────────────────────────────────
          updateMultipleFaces(false);
          updateFaceDetected(true);

          if (anchorEmbeddingRef.current === null) {
            // First face ever — make them the anchor
            if (embedding) {
              anchorEmbeddingRef.current = embedding;
              updatePersonTracking("anchor_set");
            }
          } else if (embedding) {
            const similarity = cosineSimilarity(anchorEmbeddingRef.current, embedding);
            const current = personTrackingRef.current;

            if (hadMultipleFacesRef.current) {
              // Returning from multi-face: verify it's the anchor before clearing the flag
              if (similarity >= PERSON_TRACKING_THRESHOLD) {
                hadMultipleFacesRef.current = false;
                updatePersonTracking("anchor_set"); // ← resumes capture automatically
              } else {
                updatePersonTracking("wrong_person");
              }
            } else if (current === "wrong_person") {
              // Waiting for anchor to return — only clear when similarity is high enough
              if (similarity >= PERSON_TRACKING_THRESHOLD) {
                updatePersonTracking("anchor_set"); // ← resumes capture automatically
              }
            } else {
              // Normal single-face tracking
              if (similarity >= PERSON_TRACKING_THRESHOLD) {
                updatePersonTracking("anchor_set");
              } else {
                // Quick swap without a multi-face interval
                updatePersonTracking("wrong_person");
              }
            }
          }
        }
      }
    } catch {
      // Silently ignore detection errors — last known state is kept
    } finally {
      detectInFlightRef.current = false;
    }
  }, [updatePersonTracking, updateFaceDetected, updateMultipleFaces]);

  // Main capture loop + detection loop
  useEffect(() => {
    let captureInterval: ReturnType<typeof setInterval>;
    let detectionInterval: ReturnType<typeof setInterval>;

    if (status === "capturing") {
      captureInterval = setInterval(() => {
        captureFrame();
      }, 800); // 800 ms interval ≈ 24 s for 30 photos
    }

    if (status === "camera_ready" || status === "capturing") {
      // Poll backend for face detection every 100 ms — as fast as the backend can respond.
      // detectInFlightRef prevents overlapping calls, so the effective rate is
      // max(100ms, backend latency). Multiple-face detection blocks the next capture tick.
      detectionInterval = setInterval(() => {
        void runFaceDetection();
      }, 100);
    }

    if (status === "completed") {
      setCapturedImages([...imagesRef.current]);
      stopCamera();
      setIsCaptureModalOpen(false);
      setIsReviewModalOpen(true);
    }

    return () => {
      clearInterval(captureInterval);
      clearInterval(detectionInterval);
    };
  }, [status, captureFrame, stopCamera, runFaceDetection]);

  const resetDetectionState = () => {
    faceDetectedRef.current = false;
    setFaceDetected(false);
    multipleFacesRef.current = false;
    setMultipleFaces(false);
    personTrackingRef.current = "none";
    setPersonTracking("none");
    anchorEmbeddingRef.current = null;
    hadMultipleFacesRef.current = false;
    detectInFlightRef.current = false;
    noFaceStartTimeRef.current = null;
    setNoFaceOverlayVisible(false);
  };

  const handleStartCapture = () => {
    if (!faculty || !studentId) {
      setFormError("Please select a faculty code and enter the student ID before starting.");
      return;
    }
    setFormError(null);
    // Reset any previous Save & Train state so a fresh capture starts clean
    setTrainingJobId(null);
    setTrainingStatus(null);
    setTrainingError(null);
    setNeedsOverwrite(false);
    imagesRef.current = [];
    setCapturedCount(0);
    setCapturedImages([]);
    resetDetectionState();
    setStatus("initializing");
    setIsCaptureModalOpen(true);
    startCamera();
  };

  const handleStopCapture = () => {
    resetDetectionState();
    setStatus("ready");
    stopCamera();
    setIsCaptureModalOpen(false);
  };

  const handleRetake = () => {
    setIsReviewModalOpen(false);
    imagesRef.current = [];
    setCapturedCount(0);
    setCapturedImages([]);
    resetDetectionState();
    setStatus("ready");
    setTrainingJobId(null);
    setTrainingStatus(null);
    setTrainingError(null);
    setNeedsOverwrite(false);
  };

  const handleSaveAndTrain = async (overwrite = false) => {
    if (!faculty || !studentId) {
      setFormError("Missing faculty code or student ID.");
      return;
    }
    if (capturedImages.length === 0) {
      setFormError("No captured images to save.");
      return;
    }

    setIsUploading(true);
    setFormError(null);
    setSaveErrorModal(null);
    setTrainingJobId(null);
    setTrainingStatus(null);
    setTrainingError(null);
    setNeedsOverwrite(false);

    try {
      const result = await admissionService.uploadStudentCapturedImages({
        faculty_code: faculty,
        student_number: studentId,
        images: capturedImages,
        overwrite,
      });

      void fetchAdmissionData();

      // Images saved — now track the training job
      if (result.job_id) {
        setTrainingJobId(result.job_id);
        setTrainingStatus("queued");
      }
    } catch (err: unknown) {
      const response = (err as { response?: { status?: number; data?: { detail?: string } } })?.response;
      const httpStatus = response?.status;
      const detail = response?.data?.detail;
      if (httpStatus === 409) {
        // Student already has images on disk — ask user to confirm overwrite
        setNeedsOverwrite(true);
      } else if (httpStatus === 404) {
        // Student number does not exist in the database
        setSaveErrorModal({
          title: "Student Not Found",
          message: detail ?? `Student number "${studentId}" was not found in the system. Please register the student first before capturing their face.`,
        });
      } else if (httpStatus === 400) {
        // Validation error — e.g. faculty code doesn't match the student's faculty
        setSaveErrorModal({
          title: "Faculty Mismatch",
          message: detail ?? "The selected faculty code does not match the student's registered faculty. Please select the correct faculty and try again.",
        });
      } else {
        setFormError(detail ?? "Failed to save images or start training. Please try again.");
      }
    } finally {
      setIsUploading(false);
    }
  };

  // Determine if capture is actively allowed
  const captureAllowed = faceDetected && !multipleFaces && personTracking !== "wrong_person";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
            Students Face Registration
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Capture and register student faces for the AI attendance system.
          </p>
        </div>
      </div>

      <div className="max-w-2xl mx-auto">
        <Card className="glass-card shadow-lg border-primary/20">
          <CardHeader className="border-b border-gray-200 dark:border-white/10 pb-6">
            <CardTitle className="flex items-center gap-2">
              <Camera className="text-primary" />
              Registration Setup
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Faculty Code
                </label>
                <Select
                  value={faculty}
                  onChange={(e) => setFaculty(e.target.value)}
                  options={[
                    { value: "", label: "Select Faculty Code" },
                    ...facultyOptions.map((facultyOption) => ({
                      value: facultyOption.code,
                      label: facultyOption.code,
                    })),
                  ]}
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Student ID / Number
                </label>
                <Input
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                  placeholder="e.g. CIS26001"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Photo Count
                </label>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setPhotoCount(Math.max(1, photoCount - 5))}
                  >
                    -
                  </Button>
                  <Input
                    type="number"
                    value={photoCount}
                    onChange={(e) =>
                      setPhotoCount(parseInt(e.target.value) || 0)
                    }
                    className="text-center"
                    min={1}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setPhotoCount(photoCount + 5)}
                  >
                    +
                  </Button>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Camera Index
                </label>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setCameraIndex(Math.max(0, cameraIndex - 1))}
                  >
                    -
                  </Button>
                  <Input
                    type="number"
                    value={cameraIndex}
                    onChange={(e) =>
                      setCameraIndex(parseInt(e.target.value) || 0)
                    }
                    className="text-center"
                    min={0}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setCameraIndex(cameraIndex + 1)}
                  >
                    +
                  </Button>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-white/5 rounded-xl p-4 flex items-center justify-between border border-gray-200 dark:border-white/10">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Status:{" "}
                  <span className="font-semibold text-gray-900 dark:text-white capitalize">
                    {status.replace("_", " ")}
                  </span>
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Captured:{" "}
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {status === "capturing"
                      ? capturedCount
                      : capturedImages.length}
                  </span>{" "}
                  / {photoCount}
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  onClick={handleStartCapture}
                  disabled={status === "capturing" || status === "initializing"}
                >
                  <Camera size={18} className="mr-2" />
                  Start Capture
                </Button>
              </div>
            </div>

            {/* Inline error message — replaces alert() */}
            {formError && (
              <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 dark:border-red-500/30 dark:bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-400">
                <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                <span>{formError}</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Live Capture Modal */}
      <Modal
        isOpen={isCaptureModalOpen}
        onClose={handleStopCapture}
        title="Live Face Capture"
        className="max-w-4xl"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 relative rounded-2xl overflow-hidden bg-black aspect-video flex items-center justify-center border-2 border-white/10">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover"
              style={{ transform: "scaleX(-1)" }} // Mirror effect
              onLoadedMetadata={() => {
                if (status === "initializing") {
                  // Controlled delay to ensure stream is fully stabilized and no corrupted frames
                  setTimeout(() => {
                    setStatus("capturing");
                  }, 1200);
                }
              }}
            />

            {status === "initializing" && (
              <div className="absolute inset-0 z-30 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center text-white">
                <RefreshCw
                  className="animate-spin mb-4 text-primary"
                  size={40}
                />
                <p className="text-lg font-medium">Initializing Camera...</p>
                <p className="text-sm opacity-70 mt-1">
                  Waiting for stream to stabilize
                </p>
              </div>
            )}

            {/* Capture-ready border ring — green when OK, red when blocked */}
            {(status === "camera_ready" || status === "capturing") && (
              <div
                className={cn(
                  "absolute inset-0 pointer-events-none z-10 rounded-2xl transition-colors duration-300",
                  captureAllowed
                    ? "ring-2 ring-inset ring-emerald-500/60"
                    : "ring-2 ring-inset ring-red-500/60",
                )}
              />
            )}

            <AnimatePresence>
              {/* Multiple faces detected */}
              {multipleFaces && status !== "initializing" && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-500/90 text-white px-4 py-2 rounded-lg backdrop-blur flex items-center gap-2 text-sm font-medium z-20 shadow-xl whitespace-nowrap"
                >
                  <AlertTriangle size={16} />
                  Multiple Faces Detected — please ask others to step away
                </motion.div>
              )}

              {/* Wrong person in frame */}
              {!multipleFaces && personTracking === "wrong_person" && status !== "initializing" && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="absolute top-4 left-1/2 -translate-x-1/2 bg-orange-500/90 text-white px-4 py-2 rounded-lg backdrop-blur flex items-center gap-2 text-sm font-medium z-20 shadow-xl whitespace-nowrap"
                >
                  <UserX size={16} />
                  Original person must return to continue
                </motion.div>
              )}

              {/* No face detected — only shown after 10 s of continuous no-face */}
              {noFaceOverlayVisible && !multipleFaces && status !== "initializing" && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="absolute top-4 left-1/2 -translate-x-1/2 bg-amber-500/90 text-white px-4 py-2 rounded-lg backdrop-blur flex items-center gap-2 text-sm font-medium z-20 shadow-xl whitespace-nowrap"
                >
                  <AlertTriangle size={16} />
                  No Face Detected — please step in front of the camera
                </motion.div>
              )}
            </AnimatePresence>

            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 text-white px-4 py-2 rounded-full backdrop-blur text-sm font-medium z-20">
              Captured: {capturedCount} / {photoCount}
            </div>
          </div>

          <div className="space-y-6 flex flex-col justify-between">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Capture Progress
              </h3>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500 dark:text-gray-400">
                    Progress
                  </span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {Math.round((capturedCount / photoCount) * 100)}%
                  </span>
                </div>
                <div className="h-2 w-full bg-gray-200 dark:bg-white/10 rounded-full overflow-hidden">
                  <motion.div
                    className={cn(
                      "h-full transition-colors duration-300",
                      !captureAllowed ? "bg-red-500" : "bg-primary",
                    )}
                    initial={{ width: 0 }}
                    animate={{
                      width: `${(capturedCount / photoCount) * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div className="p-4 bg-gray-50 dark:bg-white/5 rounded-xl border border-gray-200 dark:border-white/10">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                  Instructions
                </h4>
                <ul className="text-sm text-gray-500 dark:text-gray-400 space-y-2">
                  <li className="flex items-start gap-2">
                    <CheckCircle2
                      size={16}
                      className={cn(
                        "shrink-0 mt-0.5",
                        faceDetected ? "text-emerald-500" : "text-gray-400",
                      )}
                    />
                    Face visible in frame
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2
                      size={16}
                      className="text-emerald-500 shrink-0 mt-0.5"
                    />
                    Ensure good lighting
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2
                      size={16}
                      className={cn(
                        "shrink-0 mt-0.5",
                        !multipleFaces ? "text-emerald-500" : "text-red-500",
                      )}
                    />
                    One person in frame only
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2
                      size={16}
                      className={cn(
                        "shrink-0 mt-0.5",
                        personTracking !== "wrong_person"
                          ? "text-emerald-500"
                          : "text-orange-500",
                      )}
                    />
                    Same person throughout
                  </li>
                </ul>
              </div>
            </div>

            <div className="space-y-3">
              {status === "capturing" && (
                <Button
                  variant="secondary"
                  className="w-full text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10"
                  onClick={() => setStatus("paused")}
                >
                  <StopCircle size={18} className="mr-2" />
                  Pause Capture
                </Button>
              )}
              {status === "paused" && (
                <Button
                  className="w-full bg-emerald-500 hover:bg-emerald-600 text-white border-transparent shadow-lg shadow-emerald-500/20"
                  onClick={() => setStatus("capturing")}
                >
                  <Play size={18} className="mr-2" />
                  Resume Capture
                </Button>
              )}
              <Button
                variant="ghost"
                className="w-full text-gray-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10"
                onClick={handleStopCapture}
              >
                <X size={18} className="mr-2" />
                Close
              </Button>
            </div>
          </div>
        </div>
      </Modal>

      {/* Save Validation Error Modal (404 student not found / 400 faculty mismatch) */}
      <Modal
        isOpen={!!saveErrorModal}
        onClose={() => setSaveErrorModal(null)}
        title={saveErrorModal?.title ?? "Error"}
      >
        <div className="space-y-5">
          <div className="flex items-start gap-4 p-4 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-100 dark:border-red-500/20">
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-500/20 flex items-center justify-center shrink-0">
              <AlertTriangle className="text-red-600 dark:text-red-400" size={20} />
            </div>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              {saveErrorModal?.message}
            </p>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Go to the <span className="font-medium text-gray-700 dark:text-gray-300">Students</span> page to verify the student's details, then return here with the correct faculty code and student number.
          </p>
          <div className="flex justify-end">
            <Button onClick={() => setSaveErrorModal(null)}>
              Got It
            </Button>
          </div>
        </div>
      </Modal>

      {/* Review Modal */}
      <Modal
        isOpen={isReviewModalOpen}
        onClose={() => setIsReviewModalOpen(false)}
        title="Review Captured Images"
        className="max-w-5xl"
      >
        <div className="space-y-6">
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3 max-h-[50vh] overflow-y-auto custom-scrollbar p-1">
            {capturedImages.map((img, i) => (
              <div
                key={i}
                className="relative aspect-square rounded-lg overflow-hidden border border-gray-200 dark:border-white/10 group"
              >
                <img
                  src={img}
                  alt={`Capture ${i}`}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-1 right-1 bg-emerald-500 rounded-full p-0.5">
                  <CheckCircle2 size={12} className="text-white" />
                </div>
              </div>
            ))}
          </div>

          <div className="flex flex-col gap-4 pt-4 border-t border-gray-200 dark:border-white/10">

            {/* Overwrite confirmation — shown when backend returns 409 */}
            {needsOverwrite && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-500/30 dark:bg-amber-500/10 px-4 py-3 space-y-2">
                <p className="text-sm font-semibold text-amber-800 dark:text-amber-300 flex items-center gap-2">
                  <AlertTriangle size={16} />
                  Student already has registered images
                </p>
                <p className="text-sm text-amber-700 dark:text-amber-400">
                  This student already has face data saved. Do you want to replace the existing images with the new capture?
                </p>
                <div className="flex gap-3 pt-1">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setNeedsOverwrite(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    className="bg-amber-500 hover:bg-amber-600 text-white border-transparent"
                    onClick={() => void handleSaveAndTrain(true)}
                    disabled={isUploading}
                  >
                    {isUploading ? "Replacing..." : "Yes, Replace & Retrain"}
                  </Button>
                </div>
              </div>
            )}

            {/* Training progress — shown after images are saved */}
            {trainingJobId && !needsOverwrite && (
              <div
                className={cn(
                  "rounded-lg border px-4 py-3 space-y-1",
                  trainingStatus === "succeeded"
                    ? "border-emerald-200 bg-emerald-50 dark:border-emerald-500/30 dark:bg-emerald-500/10"
                    : trainingStatus === "failed"
                    ? "border-red-200 bg-red-50 dark:border-red-500/30 dark:bg-red-500/10"
                    : "border-blue-200 bg-blue-50 dark:border-blue-500/30 dark:bg-blue-500/10",
                )}
              >
                <p
                  className={cn(
                    "text-sm font-semibold flex items-center gap-2",
                    trainingStatus === "succeeded"
                      ? "text-emerald-700 dark:text-emerald-400"
                      : trainingStatus === "failed"
                      ? "text-red-700 dark:text-red-400"
                      : "text-blue-700 dark:text-blue-400",
                  )}
                >
                  {trainingStatus === "succeeded" ? (
                    <CheckCircle2 size={16} />
                  ) : trainingStatus === "failed" ? (
                    <AlertTriangle size={16} />
                  ) : (
                    <RefreshCw size={16} className="animate-spin" />
                  )}
                  {trainingStatus === "queued" && "Training queued — waiting to start…"}
                  {trainingStatus === "running" && "Training in progress — building face model…"}
                  {trainingStatus === "succeeded" && `Training complete! ${studentId} can now be recognised in attendance sessions.`}
                  {trainingStatus === "failed" && "Training failed"}
                </p>
                {trainingStatus === "failed" && trainingError && (
                  <p className="text-xs text-red-600 dark:text-red-400">{trainingError}</p>
                )}
                {trainingStatus === "succeeded" && (
                  <p className="text-xs text-emerald-600 dark:text-emerald-400">
                    Job ID: {trainingJobId}
                  </p>
                )}
              </div>
            )}

            {/* Bottom action row */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="bg-gray-50 dark:bg-white/5 px-4 py-2 rounded-lg border border-gray-200 dark:border-white/10">
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Total Images:{" "}
                  <span className="font-bold text-gray-900 dark:text-white">
                    {capturedImages.length}
                  </span>
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Valid Faces:{" "}
                  <span className="font-bold text-emerald-500">
                    {capturedImages.length} / {photoCount}
                  </span>
                </p>
              </div>

              <div className="flex gap-3 w-full sm:w-auto">
                <Button
                  variant="secondary"
                  className="flex-1 sm:flex-none"
                  onClick={handleRetake}
                  disabled={isUploading || trainingStatus === "running" || trainingStatus === "queued"}
                >
                  <RefreshCw size={18} className="mr-2" />
                  Retake
                </Button>
                {/* Hide Save & Train once training has started */}
                {!trainingJobId && (
                  <Button
                    className="flex-1 sm:flex-none bg-emerald-500 hover:bg-emerald-600 text-white border-transparent"
                    onClick={() => void handleSaveAndTrain(false)}
                    disabled={isUploading || needsOverwrite}
                  >
                    <Save size={18} className="mr-2" />
                    {isUploading ? "Saving..." : "Save & Train"}
                  </Button>
                )}
                {/* Close button shown after terminal state */}
                {(trainingStatus === "succeeded" || trainingStatus === "failed") && (
                  <Button
                    className="flex-1 sm:flex-none"
                    onClick={() => {
                      setIsReviewModalOpen(false);
                      setStatus("ready");
                      imagesRef.current = [];
                      setCapturedCount(0);
                      setCapturedImages([]);
                      setTrainingJobId(null);
                      setTrainingStatus(null);
                      setTrainingError(null);
                    }}
                  >
                    <CheckCircle2 size={18} className="mr-2" />
                    Done
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </Modal>
    </div>
  );
}
