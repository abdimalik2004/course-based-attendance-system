import { useState, useEffect, useRef, useCallback } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { motion, AnimatePresence } from "framer-motion";
import { User, Lock, Eye, EyeOff, ShieldAlert, Timer, Mail, ArrowLeft, CheckCircle2, KeyRound } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useAuthStore } from "@/store/useAuthStore";
import { authService } from "@/services/authService";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import logoUrl from "@/assets/logo.png";
import lightLogoUrl from "@/assets/light-logo.png";

const loginSchema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});

type LoginForm = z.infer<typeof loginSchema>;

// ── Forgot-password step machine ─────────────────────────────────────────────
type FpStep = "login" | "fp_email" | "fp_code" | "fp_newpw" | "fp_success";

// Shared slide animation for step transitions
const slideVariants = {
  enter: { opacity: 0, x: 32 },
  center: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -32 },
};

// ── 6-box OTP input component ─────────────────────────────────────────────────
function OtpInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  const handleChange = useCallback(
    (idx: number, char: string) => {
      const digit = char.replace(/\D/g, "").slice(-1);
      const arr = value.padEnd(6, " ").split("");
      arr[idx] = digit || " ";
      const next = arr.join("").trimEnd();
      onChange(next);
      if (digit && idx < 5) refs.current[idx + 1]?.focus();
    },
    [value, onChange],
  );

  const handleKeyDown = useCallback(
    (idx: number, e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Backspace") {
        const arr = value.padEnd(6, " ").split("");
        if (arr[idx].trim()) {
          arr[idx] = " ";
          onChange(arr.join("").trimEnd());
        } else if (idx > 0) {
          refs.current[idx - 1]?.focus();
          const arr2 = value.padEnd(6, " ").split("");
          arr2[idx - 1] = " ";
          onChange(arr2.join("").trimEnd());
        }
      }
    },
    [value, onChange],
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      e.preventDefault();
      const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
      onChange(pasted);
      const focusIdx = Math.min(pasted.length, 5);
      refs.current[focusIdx]?.focus();
    },
    [onChange],
  );

  return (
    <div className="flex gap-2.5 justify-center">
      {Array.from({ length: 6 }).map((_, idx) => (
        <input
          key={idx}
          ref={(el) => { refs.current[idx] = el; }}
          type="text"
          inputMode="numeric"
          maxLength={1}
          value={(value[idx] ?? "").trim()}
          onChange={(e) => handleChange(idx, e.target.value)}
          onKeyDown={(e) => handleKeyDown(idx, e)}
          onPaste={handlePaste}
          onFocus={(e) => e.target.select()}
          className={`
            w-11 h-14 text-center text-xl font-bold rounded-xl border-2 outline-none
            transition-all duration-150 bg-white dark:bg-white/5
            text-gray-900 dark:text-white caret-primary
            ${(value[idx] ?? "").trim()
              ? "border-primary shadow-[0_0_0_3px_rgba(37,99,235,0.15)]"
              : "border-gray-200 dark:border-white/10 focus:border-primary focus:shadow-[0_0_0_3px_rgba(37,99,235,0.15)]"
            }
          `}
        />
      ))}
    </div>
  );
}

export default function Login() {
  // ── Login form state ─────────────────────────────────────────────────────
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [lockoutSeconds, setLockoutSeconds] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const { login } = useAuthStore();

  const {
    register,
    handleSubmit,
    setError,
    clearErrors,
    formState: { errors },
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
  });

  // Countdown timer: tick every second until lockout expires
  useEffect(() => {
    if (lockoutSeconds <= 0) {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      return;
    }
    timerRef.current = setInterval(() => {
      setLockoutSeconds((s) => {
        if (s <= 1) { clearErrors("root"); return 0; }
        return s - 1;
      });
    }, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [lockoutSeconds > 0]);

  const isLocked = lockoutSeconds > 0;

  const onSubmit = async (data: LoginForm) => {
    if (isLocked) return;
    setIsLoading(true);
    try {
      const user = await authService.login(data.username, data.password);
      const roleSource = Array.isArray(user.role_names) && user.role_names.length > 0
        ? user.role_names[0] : user.role;
      const role = (roleSource || "").toString().toUpperCase();
      let destination = "/";
      switch (role) {
        case "SUPER_ADMIN": case "ADMIN": destination = "/admin/dashboard"; break;
        case "ACADEMIA": destination = "/academia/dashboard"; break;
        case "HR": destination = "/hr/dashboard"; break;
        case "ADMISSIONS": destination = "/admission/dashboard"; break;
        case "FACULTY": destination = "/faculty/dashboard"; break;
        case "TEACHER": destination = "/teacher/dashboard"; break;
        case "STUDENT": destination = "/student/dashboard"; break;
        default: destination = "/";
      }
      window.location.replace(destination);
    } catch (err: any) {
      const status = err?.response?.status;
      const details = err?.response?.data?.error?.details;
      if (status === 429) {
        const retryAfter: number = details?.retry_after ?? 30;
        setLockoutSeconds(retryAfter);
        setError("root", { type: "lockout", message: details?.message ?? `Too many failed attempts. Please wait ${retryAfter} seconds.` });
      } else {
        const message = err?.response?.data?.error?.message || err?.response?.data?.detail || err.message || "Login failed";
        setError("root", { type: "manual", message });
      }
    } finally {
      setIsLoading(false);
    }
  };

  // ── Forgot-password state ────────────────────────────────────────────────
  const [fpStep, setFpStep] = useState<FpStep>("login");
  const [fpEmail, setFpEmail] = useState("");
  const [fpEmailError, setFpEmailError] = useState("");
  const [fpOtp, setFpOtp] = useState("");
  const [fpOtpError, setFpOtpError] = useState("");
  const [fpResetToken, setFpResetToken] = useState("");
  const [fpNewPw, setFpNewPw] = useState("");
  const [fpConfirmPw, setFpConfirmPw] = useState("");
  const [fpPwError, setFpPwError] = useState("");
  const [fpLoading, setFpLoading] = useState(false);
  const [showFpPw, setShowFpPw] = useState(false);
  const [showFpConfirm, setShowFpConfirm] = useState(false);
  const [resendSeconds, setResendSeconds] = useState(0);
  const resendTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [fpDevMode, setFpDevMode] = useState(false);

  // Resend countdown
  useEffect(() => {
    if (resendSeconds <= 0) {
      if (resendTimerRef.current) { clearInterval(resendTimerRef.current); resendTimerRef.current = null; }
      return;
    }
    resendTimerRef.current = setInterval(() => {
      setResendSeconds((s) => (s <= 1 ? 0 : s - 1));
    }, 1000);
    return () => { if (resendTimerRef.current) clearInterval(resendTimerRef.current); };
  }, [resendSeconds > 0]);

  const startResendTimer = () => setResendSeconds(60);

  const handleFpEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = fpEmail.trim().toLowerCase();
    if (!trimmed || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed)) {
      setFpEmailError("Please enter a valid email address");
      return;
    }
    setFpEmailError("");
    setFpLoading(true);
    try {
      const { devCode } = await authService.forgotPassword(trimmed);
      setFpStep("fp_code");
      startResendTimer();
      if (devCode) {
        // SMTP not configured — backend returned the code directly for dev testing
        setFpOtp(devCode);
        setFpDevMode(true);
      } else {
        setFpDevMode(false);
      }
    } catch {
      setFpEmailError("Something went wrong. Please try again.");
    } finally {
      setFpLoading(false);
    }
  };

  const handleResend = async () => {
    if (resendSeconds > 0) return;
    setFpOtp("");
    setFpOtpError("");
    setFpLoading(true);
    try {
      await authService.forgotPassword(fpEmail.trim().toLowerCase());
      startResendTimer();
    } catch {
      setFpOtpError("Failed to resend. Please try again.");
    } finally {
      setFpLoading(false);
    }
  };

  const handleFpCodeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const code = fpOtp.trim();
    if (code.length !== 6 || !/^\d{6}$/.test(code)) {
      setFpOtpError("Please enter the full 6-digit code");
      return;
    }
    setFpOtpError("");
    setFpLoading(true);
    try {
      const token = await authService.verifyResetCode(fpEmail.trim().toLowerCase(), code);
      setFpResetToken(token);
      setFpStep("fp_newpw");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setFpOtpError(detail || "Invalid code. Please check and try again.");
    } finally {
      setFpLoading(false);
    }
  };

  const handleFpPasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (fpNewPw.length < 6) {
      setFpPwError("Password must be at least 6 characters");
      return;
    }
    if (fpNewPw !== fpConfirmPw) {
      setFpPwError("Passwords do not match");
      return;
    }
    setFpPwError("");
    setFpLoading(true);
    try {
      await authService.resetPassword(fpResetToken, fpNewPw);
      setFpStep("fp_success");
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setFpPwError(detail || "Session expired. Please start over.");
    } finally {
      setFpLoading(false);
    }
  };

  const resetFpFlow = () => {
    setFpStep("login");
    setFpEmail(""); setFpEmailError("");
    setFpOtp(""); setFpOtpError("");
    setFpResetToken("");
    setFpNewPw(""); setFpConfirmPw(""); setFpPwError("");
    setResendSeconds(0);
    setFpDevMode(false);
  };

  // ── Logo header (shared) ─────────────────────────────────────────────────
  const LogoHeader = ({ title, subtitle }: { title: string; subtitle: string }) => (
    <div className="flex flex-col items-center mt-2 mb-8 text-center">
      <div className="w-20 h-20 bg-white dark:bg-white/5 rounded-3xl flex items-center justify-center mb-5 shadow-xl border border-gray-200 dark:border-white/10 p-2 overflow-hidden">
        <img src={lightLogoUrl} alt="Heegan Light" className="dark:hidden w-full h-full object-cover rounded-2xl" />
        <img src={logoUrl} alt="Heegan Dark" className="hidden dark:block w-full h-full object-cover rounded-2xl" />
      </div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1.5 tracking-tight">{title}</h1>
      <p className="text-gray-500 dark:text-gray-400 text-sm">{subtitle}</p>
    </div>
  );

  return (
    <div className="min-h-screen w-full bg-gray-50 dark:bg-dark-bg flex items-center justify-center py-16 relative overflow-hidden px-4 transition-colors duration-300">
      <div className="absolute top-4 right-4 z-50"><ThemeToggle /></div>
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-primary-accent/10 rounded-full blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-md my-8 -translate-y-10 lg:-translate-y-16"
      >
        <div className="glass-card rounded-[2.5rem] px-8 py-10 md:px-12 md:py-12 relative overflow-hidden shadow-2xl shadow-primary/5 dark:shadow-white/5 border border-white/20 dark:border-white/10">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-brand opacity-80" />

          <AnimatePresence mode="wait">

            {/* ── STEP: Login ──────────────────────────────────────────────── */}
            {fpStep === "login" && (
              <motion.div key="login" variants={slideVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.22 }}>
                <div className="flex flex-col items-center mt-2 mb-10 text-center">
                  <div className="w-28 h-28 bg-white dark:bg-white/5 rounded-3xl flex items-center justify-center mb-6 shadow-xl border border-gray-200 dark:border-white/10 p-2 overflow-hidden">
                    <img src={lightLogoUrl} alt="Heegan Light" className="dark:hidden w-full h-full object-cover rounded-2xl" />
                    <img src={logoUrl} alt="Heegan Dark" className="hidden dark:block w-full h-full object-cover rounded-2xl" />
                  </div>
                  <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 tracking-tight">Welcome back</h1>
                  <p className="text-gray-500 dark:text-gray-400 text-sm">Enter your credentials to access your dashboard</p>
                </div>

                <form onSubmit={handleSubmit(onSubmit)} className="space-y-7">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">Username</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                        <User size={18} />
                      </div>
                      <Input {...register("username")} type="text" placeholder="Enter your username" className="pl-11 transition-all" error={errors.username?.message} disabled={isLocked} />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between ml-1">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Password</label>
                    </div>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                        <Lock size={18} />
                      </div>
                      <Input {...register("password")} type={showPassword ? "text" : "password"} placeholder="Enter your password" className="pl-11 pr-11 transition-all" error={errors.password?.message} disabled={isLocked} />
                      <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 z-10 focus:outline-none transition-colors" tabIndex={-1}>
                        {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>

                  {errors.root && (
                    <div className={`p-3 rounded-lg border text-center space-y-1 ${isLocked ? "bg-amber-500/10 border-amber-500/20" : "bg-red-500/10 border-red-500/20"}`}>
                      <div className="flex items-center justify-center gap-2">
                        {isLocked && <ShieldAlert size={15} className="text-amber-400 shrink-0" />}
                        <p className={`text-sm font-medium ${isLocked ? "text-amber-400" : "text-red-500"}`}>{errors.root.message}</p>
                      </div>
                      {isLocked && (
                        <div className="flex items-center justify-center gap-1.5 text-amber-300/80 text-xs">
                          <Timer size={13} />
                          <span>You can try again in <span className="font-bold tabular-nums">{lockoutSeconds}s</span></span>
                        </div>
                      )}
                    </div>
                  )}

                  <Button type="submit" className="w-full mt-10 mb-2" size="lg" isLoading={isLoading} disabled={isLocked || isLoading}>
                    {isLocked ? `Locked · ${lockoutSeconds}s` : "Sign in to Heegan"}
                  </Button>
                </form>
              </motion.div>
            )}

            {/* ── STEP: Enter email ─────────────────────────────────────────── */}
            {fpStep === "fp_email" && (
              <motion.div key="fp_email" variants={slideVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.22 }}>
                <LogoHeader
                  title="Reset your password"
                  subtitle="Enter the email address linked to your account"
                />
                <form onSubmit={handleFpEmailSubmit} className="space-y-6">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">Email address</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                        <Mail size={18} />
                      </div>
                      <Input
                        type="email"
                        placeholder="your@email.com"
                        value={fpEmail}
                        onChange={(e) => { setFpEmail(e.target.value); setFpEmailError(""); }}
                        className="pl-11 transition-all"
                        error={fpEmailError}
                        autoFocus
                      />
                    </div>
                  </div>

                  <Button type="submit" className="w-full" size="lg" isLoading={fpLoading} disabled={fpLoading}>
                    Send Reset Code
                  </Button>

                  <button
                    type="button"
                    onClick={resetFpFlow}
                    className="flex items-center gap-1.5 mx-auto text-sm text-gray-500 dark:text-gray-400 hover:text-primary dark:hover:text-primary transition-colors"
                  >
                    <ArrowLeft size={15} /> Back to sign in
                  </button>
                </form>
              </motion.div>
            )}

            {/* ── STEP: Enter 6-digit code ──────────────────────────────────── */}
            {fpStep === "fp_code" && (
              <motion.div key="fp_code" variants={slideVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.22 }}>
                <LogoHeader
                  title="Check your email"
                  subtitle={`We sent a 6-digit code to ${fpEmail}`}
                />
                <form onSubmit={handleFpCodeSubmit} className="space-y-6">
                  {fpDevMode && (
                    <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3 text-center">
                      <p className="text-xs font-semibold text-amber-400 mb-0.5">Development mode</p>
                      <p className="text-xs text-amber-300/70">
                        SMTP not configured — code auto-filled below.<br />
                        Add <code className="font-mono">SMTP_EMAIL</code> &amp; <code className="font-mono">SMTP_PASSWORD</code> to your <code className="font-mono">.env</code> for real emails.
                      </p>
                    </div>
                  )}
                  <div className="space-y-4">
                    <OtpInput value={fpOtp} onChange={(v) => { setFpOtp(v); setFpOtpError(""); }} />
                    {fpOtpError && (
                      <p className="text-sm text-red-500 text-center">{fpOtpError}</p>
                    )}
                  </div>

                  <Button
                    type="submit"
                    className="w-full"
                    size="lg"
                    isLoading={fpLoading}
                    disabled={fpLoading || fpOtp.replace(/\s/g, "").length !== 6}
                  >
                    Verify Code
                  </Button>

                  {/* Resend link */}
                  <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                    Didn't receive it?{" "}
                    {resendSeconds > 0 ? (
                      <span className="text-gray-400 dark:text-gray-500">
                        Resend in <span className="font-bold tabular-nums">{resendSeconds}s</span>
                      </span>
                    ) : (
                      <button
                        type="button"
                        onClick={handleResend}
                        disabled={fpLoading}
                        className="text-primary-accent hover:text-primary transition-colors font-medium"
                      >
                        Resend code
                      </button>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={() => { setFpStep("fp_email"); setFpOtp(""); setFpOtpError(""); }}
                    className="flex items-center gap-1.5 mx-auto text-sm text-gray-500 dark:text-gray-400 hover:text-primary dark:hover:text-primary transition-colors"
                  >
                    <ArrowLeft size={15} /> Change email
                  </button>
                </form>
              </motion.div>
            )}

            {/* ── STEP: Set new password ────────────────────────────────────── */}
            {fpStep === "fp_newpw" && (
              <motion.div key="fp_newpw" variants={slideVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.22 }}>
                <LogoHeader
                  title="Set new password"
                  subtitle="Choose a strong password for your account"
                />
                <form onSubmit={handleFpPasswordSubmit} className="space-y-5">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">New password</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                        <KeyRound size={18} />
                      </div>
                      <Input
                        type={showFpPw ? "text" : "password"}
                        placeholder="At least 6 characters"
                        value={fpNewPw}
                        onChange={(e) => { setFpNewPw(e.target.value); setFpPwError(""); }}
                        className="pl-11 pr-11 transition-all"
                        autoFocus
                      />
                      <button type="button" onClick={() => setShowFpPw(!showFpPw)} className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 z-10 focus:outline-none transition-colors" tabIndex={-1}>
                        {showFpPw ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">Confirm password</label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                        <Lock size={18} />
                      </div>
                      <Input
                        type={showFpConfirm ? "text" : "password"}
                        placeholder="Repeat your password"
                        value={fpConfirmPw}
                        onChange={(e) => { setFpConfirmPw(e.target.value); setFpPwError(""); }}
                        className="pl-11 pr-11 transition-all"
                        error={fpPwError}
                      />
                      <button type="button" onClick={() => setShowFpConfirm(!showFpConfirm)} className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 z-10 focus:outline-none transition-colors" tabIndex={-1}>
                        {showFpConfirm ? <EyeOff size={18} /> : <Eye size={18} />}
                      </button>
                    </div>
                  </div>

                  <Button type="submit" className="w-full mt-2" size="lg" isLoading={fpLoading} disabled={fpLoading}>
                    Update Password
                  </Button>
                </form>
              </motion.div>
            )}

            {/* ── STEP: Success ─────────────────────────────────────────────── */}
            {fpStep === "fp_success" && (
              <motion.div key="fp_success" variants={slideVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.22 }}>
                <div className="flex flex-col items-center py-6 text-center gap-5">
                  <div className="w-20 h-20 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                    <CheckCircle2 size={40} className="text-emerald-500" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Password updated!</h2>
                    <p className="text-gray-500 dark:text-gray-400 text-sm">
                      Your password has been changed successfully.<br />You can now sign in with your new password.
                    </p>
                  </div>
                  <Button className="w-full mt-2" size="lg" onClick={resetFpFlow}>
                    Sign in to Heegan
                  </Button>
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}
