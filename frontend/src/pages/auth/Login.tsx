import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { motion } from "framer-motion";
import { User, Lock, Eye, EyeOff } from "lucide-react";
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

export default function Login() {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { login } = useAuthStore();

  const {
    register,
    handleSubmit,
    setError,
    formState: { errors },
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginForm) => {
    setIsLoading(true);
    try {
      const user = await authService.login(data.username, data.password);
      // user.role may be uppercase; normalize for routing
      const roleSource =
        Array.isArray(user.role_names) && user.role_names.length > 0
          ? user.role_names[0]
          : user.role;
      const role = (roleSource || "").toString().toUpperCase();
      let destination = "/";
      switch (role) {
        case "SUPER_ADMIN":
        case "ADMIN":
          destination = "/admin/dashboard";
          break;
        case "ACADEMIA":
          destination = "/academia/dashboard";
          break;
        case "HR":
          destination = "/hr/dashboard";
          break;
        case "ADMISSIONS":
          destination = "/admission/dashboard";
          break;
        case "FACULTY":
        case "FACULTY_ADMIN":
          destination = "/faculty/dashboard";
          break;
        case "TEACHER":
          destination = "/teacher/dashboard";
          break;
        case "STUDENT":
          destination = "/student/dashboard";
          break;
        default:
          destination = "/";
      }

      // Hard redirect so the dashboard loads cleanly after auth state updates.
      window.location.replace(destination);
    } catch (err: any) {
      const message =
        err?.response?.data?.detail || err.message || "Login failed";
      setError("root", { type: "manual", message });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gray-50 dark:bg-dark-bg flex items-center justify-center py-16 relative overflow-hidden px-4 transition-colors duration-300">
      {/* Theme Toggle Button */}
      <div className="absolute top-4 right-4 z-50">
        <ThemeToggle />
      </div>

      {/* Background ambient accents */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-primary-accent/10 rounded-full blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-md my-8 -translate-y-10 lg:-translate-y-16"
      >
        <div className="glass-card rounded-[2.5rem] px-8 py-10 md:px-12 md:py-14 relative overflow-hidden shadow-2xl shadow-primary/5 dark:shadow-white/5 border border-white/20 dark:border-white/10">
          {/* Top border gradient accent */}
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-brand opacity-80" />

          <div className="flex flex-col items-center mt-2 mb-10 text-center">
            <div className="w-28 h-28 bg-white dark:bg-white/5 rounded-3xl flex items-center justify-center mb-6 shadow-xl border border-gray-200 dark:border-white/10 p-2 overflow-hidden">
              <img
                src={lightLogoUrl}
                alt="Heegan Light"
                className="dark:hidden w-full h-full object-cover rounded-2xl"
              />
              <img
                src={logoUrl}
                alt="Heegan Dark"
                className="hidden dark:block w-full h-full object-cover rounded-2xl"
              />
            </div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 tracking-tight">
              Welcome back
            </h1>
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              Enter your credentials to access your dashboard
            </p>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-7">
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
                Username
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                  <User size={18} />
                </div>
                <Input
                  {...register("username")}
                  type="text"
                  placeholder="Enter your username"
                  className="pl-11 transition-all"
                  error={errors.username?.message}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between ml-1">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Password
                </label>
                <a
                  href="#"
                  className="text-xs text-primary-accent hover:text-primary transition-colors"
                >
                  Forgot Password?
                </a>
              </div>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 z-10 transition-colors group-focus-within:text-primary">
                  <Lock size={18} />
                </div>
                <Input
                  {...register("password")}
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  className="pl-11 pr-11 transition-all"
                  error={errors.password?.message}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 z-10 focus:outline-none transition-colors"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {/* Display root error from failed auth */}
            {errors.root && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-center">
                <p className="text-sm font-medium text-red-500">
                  {errors.root.message}
                </p>
              </div>
            )}

            <Button
              type="submit"
              className="w-full mt-10 mb-2"
              size="lg"
              isLoading={isLoading}
            >
              Sign in to Heegan
            </Button>
          </form>
        </div>
      </motion.div>
    </div>
  );
}
