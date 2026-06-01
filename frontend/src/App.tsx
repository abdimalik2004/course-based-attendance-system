import { useEffect, Suspense, useState } from "react";
import { RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { router } from "./app/router";
import { useThemeStore } from "./store/useThemeStore";
import { EditProfileModal } from "./components/ui/EditProfileModal";
import { ChangePasswordModal } from "./components/ui/ChangePasswordModal";
import { ToastContainer } from "./components/ui/ToastContainer";
import { authService } from "./services/authService";

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

// Fallback component for lazy-loaded routes
const FullPageLoader = () => (
  <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
    <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
  </div>
);

function App() {
  const { theme } = useThemeStore();
  const [authInitDone, setAuthInitDone] = useState(false);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
  }, [theme]);

  // Remove the hardcoded 'dark' class
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await authService.initialize();
      } finally {
        if (mounted) setAuthInitDone(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Suspense fallback={<FullPageLoader />}>
        {!authInitDone ? (
          <FullPageLoader />
        ) : (
          <RouterProvider router={router} />
        )}
      </Suspense>
      <EditProfileModal />
      <ChangePasswordModal />
      <ToastContainer />
    </QueryClientProvider>
  );
}

export default App;
