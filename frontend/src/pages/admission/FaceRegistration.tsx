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
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Modal } from "@/components/ui/Modal";
import { useAdmissionStore } from "@/store/useAdmissionStore";
import admissionService, { type FacultyDto } from "@/services/admissionService";
import { cn } from "@/utils/cn";

type CaptureStatus =
  | "ready"
  | "initializing"
  | "camera_ready"
  | "capturing"
  | "paused"
  | "completed";

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
      } catch (error) {
        console.error("Failed to load faculty codes", error);
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

  // Video & Capture Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const imagesRef = useRef<string[]>([]);

  // Face Tracking Simulation State
  const [multipleFaces, setMultipleFaces] = useState(false);
  const [faceDetected, setFaceDetected] = useState(true);
  const [boxPosition, setBoxPosition] = useState({
    x: 30,
    y: 20,
    w: 40,
    h: 60,
  });

  // Handle WebRTC Stream
  const startCamera = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((d) => d.kind === "videoinput");

      const deviceId = videoDevices[cameraIndex]?.deviceId;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please check permissions.");
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
  const captureFrame = useCallback(() => {
    if (
      videoRef.current &&
      canvasRef.current &&
      status === "capturing" &&
      !multipleFaces &&
      faceDetected
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
  }, [status, multipleFaces, faceDetected, photoCount]);

  // Main capture loop and mock face detection simulation
  useEffect(() => {
    let captureInterval: ReturnType<typeof setInterval>;
    let mockDetectionInterval: ReturnType<typeof setInterval>;

    if (status === "capturing") {
      captureInterval = setInterval(() => {
        captureFrame();
      }, 800); // 800ms interval takes ~24 seconds for 30 photos
    }

    if (status === "camera_ready" || status === "capturing") {
      // Simulate highly realistic face detection tracking
      mockDetectionInterval = setInterval(() => {
        const hasMultiple = Math.random() < 0.05;
        const noFace = Math.random() < 0.08; // 8% chance to temporarily lose face

        setMultipleFaces(hasMultiple);
        setFaceDetected(!noFace);

        if (!noFace) {
          // Smoothly drift box to track simulated face
          setBoxPosition((prev) => {
            const dx = (Math.random() - 0.5) * 8;
            const dy = (Math.random() - 0.5) * 8;
            const dw = (Math.random() - 0.5) * 4; // slight size breathing
            return {
              x: Math.max(10, Math.min(50, prev.x + dx)),
              y: Math.max(10, Math.min(40, prev.y + dy)),
              w: Math.max(30, Math.min(50, prev.w + dw)),
              h: Math.max(45, Math.min(70, prev.h + dw * 1.5)),
            };
          });
        }
      }, 500);
    }

    if (status === "completed") {
      setCapturedImages([...imagesRef.current]);
      stopCamera();
      setIsCaptureModalOpen(false);
      setIsReviewModalOpen(true);
    }

    return () => {
      clearInterval(captureInterval);
      clearInterval(mockDetectionInterval);
    };
  }, [status, captureFrame, stopCamera]);

  const handleStartCapture = () => {
    if (!faculty || !studentId) {
      alert("Please select faculty and enter student ID.");
      return;
    }
    imagesRef.current = [];
    setCapturedCount(0);
    setCapturedImages([]);
    setStatus("initializing");
    setIsCaptureModalOpen(true);
    startCamera();
  };

  const handleStopCapture = () => {
    setStatus("ready");
    stopCamera();
    setIsCaptureModalOpen(false);
  };

  const handleRetake = () => {
    setIsReviewModalOpen(false);
    imagesRef.current = [];
    setCapturedCount(0);
    setCapturedImages([]);
    setStatus("ready");
  };

  const handleSaveAndTrain = async () => {
    setStatus("ready");
    setIsReviewModalOpen(false);
    imagesRef.current = [];
    setCapturedCount(0);
    setCapturedImages([]);

    alert(
      `Successfully saved ${photoCount} images and triggered training process for ${studentId}.`,
    );
  };

  // SVG Paths for corners
  const tl = `M ${boxPosition.x} ${boxPosition.y + 10} L ${boxPosition.x} ${boxPosition.y} L ${boxPosition.x + 10} ${boxPosition.y}`;
  const tr = `M ${boxPosition.x + boxPosition.w - 10} ${boxPosition.y} L ${boxPosition.x + boxPosition.w} ${boxPosition.y} L ${boxPosition.x + boxPosition.w} ${boxPosition.y + 10}`;
  const bl = `M ${boxPosition.x} ${boxPosition.y + boxPosition.h - 10} L ${boxPosition.x} ${boxPosition.y + boxPosition.h} L ${boxPosition.x + 10} ${boxPosition.y + boxPosition.h}`;
  const br = `M ${boxPosition.x + boxPosition.w - 10} ${boxPosition.y + boxPosition.h} L ${boxPosition.x + boxPosition.w} ${boxPosition.y + boxPosition.h} L ${boxPosition.x + boxPosition.w} ${boxPosition.y + boxPosition.h - 10}`;

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

            {/* Smooth SVG Overlay for Bounding Box */}
            <AnimatePresence>
              {(status === "camera_ready" || status === "capturing") &&
                faceDetected && (
                  <motion.svg
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 w-full h-full pointer-events-none z-10"
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                  >
                    <motion.rect
                      animate={{
                        x: boxPosition.x,
                        y: boxPosition.y,
                        width: boxPosition.w,
                        height: boxPosition.h,
                      }}
                      transition={{
                        type: "spring",
                        damping: 20,
                        stiffness: 100,
                        mass: 0.8,
                      }}
                      fill="none"
                      stroke={multipleFaces ? "#ef4444" : "#10b981"}
                      strokeWidth="0.5"
                      strokeDasharray="2,2"
                    />
                    {/* Corner brackets */}
                    <motion.path
                      animate={{ d: tl }}
                      transition={{
                        type: "spring",
                        damping: 20,
                        stiffness: 100,
                        mass: 0.8,
                      }}
                      fill="none"
                      stroke={multipleFaces ? "#ef4444" : "#10b981"}
                      strokeWidth="1"
                    />
                    <motion.path
                      animate={{ d: tr }}
                      transition={{
                        type: "spring",
                        damping: 20,
                        stiffness: 100,
                        mass: 0.8,
                      }}
                      fill="none"
                      stroke={multipleFaces ? "#ef4444" : "#10b981"}
                      strokeWidth="1"
                    />
                    <motion.path
                      animate={{ d: bl }}
                      transition={{
                        type: "spring",
                        damping: 20,
                        stiffness: 100,
                        mass: 0.8,
                      }}
                      fill="none"
                      stroke={multipleFaces ? "#ef4444" : "#10b981"}
                      strokeWidth="1"
                    />
                    <motion.path
                      animate={{ d: br }}
                      transition={{
                        type: "spring",
                        damping: 20,
                        stiffness: 100,
                        mass: 0.8,
                      }}
                      fill="none"
                      stroke={multipleFaces ? "#ef4444" : "#10b981"}
                      strokeWidth="1"
                    />
                  </motion.svg>
                )}
            </AnimatePresence>

            <AnimatePresence>
              {multipleFaces && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-500/90 text-white px-4 py-2 rounded-lg backdrop-blur flex items-center gap-2 text-sm font-medium z-20 shadow-xl"
                >
                  <AlertTriangle size={16} />
                  Multiple faces detected
                </motion.div>
              )}

              {!faceDetected && status !== "initializing" && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-amber-500/90 text-white px-6 py-3 rounded-2xl backdrop-blur flex flex-col items-center gap-2 text-sm font-medium z-20 shadow-2xl"
                >
                  <AlertTriangle size={24} />
                  No Face Detected
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
                      multipleFaces || !faceDetected
                        ? "bg-red-500"
                        : "bg-primary",
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
                      className={
                        faceDetected
                          ? "text-emerald-500 shrink-0 mt-0.5"
                          : "text-gray-400 shrink-0 mt-0.5"
                      }
                    />
                    Keep face centered in the box
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
                      className={
                        !multipleFaces
                          ? "text-emerald-500 shrink-0 mt-0.5"
                          : "text-gray-400 shrink-0 mt-0.5"
                      }
                    />
                    One person in frame
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

          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-gray-200 dark:border-white/10">
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
              >
                <RefreshCw size={18} className="mr-2" />
                Retake
              </Button>
              <Button
                className="flex-1 sm:flex-none bg-emerald-500 hover:bg-emerald-600 text-white border-transparent"
                onClick={handleSaveAndTrain}
              >
                <Save size={18} className="mr-2" />
                Save & Train
              </Button>
            </div>
          </div>
        </div>
      </Modal>
    </div>
  );
}
