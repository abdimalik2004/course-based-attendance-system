import { create } from "zustand";

export type SessionState =
  | "idle"
  | "starting"
  | "waiting_for_face"
  | "face_detected"
  | "scanning"
  | "success"
  | "failed"
  | "already_marked"
  | "low_light"
  | "partial_face";

export interface RecognizedUser {
  name: string;
  courseName: string;
  sessionId: string;
  status: string;
  confidence: number;
  time: string;
}

interface AttendanceState {
  sessionState: SessionState;
  scanningProgress: number;
  recognizedUser: RecognizedUser | null;
  activeSessionId: number | null;
  activeCourseName: string;
  /** Set to true when a session is ended via the scanner; cleared when a new session starts or the banner is dismissed */
  justEndedSession: boolean;
  lastEndedCourseName: string;

  startSession: (session?: { sessionId: number; courseName?: string }) => void;
  waitForFace: () => void;
  startScanning: () => void;
  setFaceDetected: () => void;
  setScanningProgress: (progress: number) => void;
  setRecognitionResult: (state: SessionState, user?: RecognizedUser) => void;
  setActiveSession: (sessionId: number | null, courseName?: string) => void;
  resetSession: () => void;
  dismissEndedBanner: () => void;
}

export const useAttendanceStore = create<AttendanceState>((set) => ({
  sessionState: "idle",
  scanningProgress: 0,
  recognizedUser: null,
  activeSessionId: null,
  activeCourseName: "",
  justEndedSession: false,
  lastEndedCourseName: "",

  startSession: (session) =>
    set({
      // Go straight to waiting_for_face so the scanner begins immediately
      sessionState: "waiting_for_face",
      scanningProgress: 0,
      recognizedUser: null,
      activeSessionId: session?.sessionId ?? null,
      activeCourseName: session?.courseName ?? "",
      justEndedSession: false,
    }),

  waitForFace: () =>
    set({
      sessionState: "waiting_for_face",
      scanningProgress: 0,
      recognizedUser: null,
    }),

  startScanning: () =>
    set({
      sessionState: "scanning",
      scanningProgress: 0,
      recognizedUser: null,
    }),

  setFaceDetected: () => set({ sessionState: "face_detected" }),

  setScanningProgress: (progress) =>
    set({ scanningProgress: Math.min(100, Math.max(0, progress)) }),

  setRecognitionResult: (state, user) =>
    set({
      sessionState: state,
      recognizedUser: user || null,
      scanningProgress: 100,
    }),

  setActiveSession: (sessionId, courseName = "") =>
    set({
      activeSessionId: sessionId,
      activeCourseName: courseName,
    }),

  resetSession: () =>
    set((state) => ({
      sessionState: "idle",
      scanningProgress: 0,
      recognizedUser: null,
      // Capture the course name before clearing it so the banner can show it
      justEndedSession: state.activeSessionId != null,
      lastEndedCourseName: state.activeCourseName,
      activeSessionId: null,
      activeCourseName: "",
    })),

  dismissEndedBanner: () =>
    set({ justEndedSession: false, lastEndedCourseName: "" }),
}));
