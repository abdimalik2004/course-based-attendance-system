import { useState, useRef, useEffect } from "react";
import { Camera, AlertCircle, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Modal } from "@/components/ui/Modal";
import { useAuthStore } from "@/store/useAuthStore";
import { useUIStore } from "@/store/useUIStore";
import { api } from "@/services/api";
import { useQueryClient } from "@tanstack/react-query";

const API_URL = import.meta.env.VITE_API_URL ?? "";

function resolveUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.startsWith("http") ? url : `${API_URL}${url}`;
}

export function EditProfileModal() {
  const { user, updateProfileImage, login } = useAuthStore();
  const { isEditProfileOpen, closeEditProfile } = useUIStore();
  const queryClient = useQueryClient();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync fields whenever the modal opens or user changes
  useEffect(() => {
    if (isEditProfileOpen && user) {
      setUsername(user.username ?? "");
      setEmail(user.email ?? "");
      setPreviewUrl(resolveUrl(user.profile_image_url));
      setError("");
      setSuccess("");
    }
  }, [isEditProfileOpen, user]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => setPreviewUrl(reader.result as string);
    reader.readAsDataURL(file);
    setError("");
    setSuccess("");
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError("");
    setSuccess("");
    try {
      let changed = false;

      // Upload new profile picture if selected
      if (fileInputRef.current?.files?.length) {
        const file = fileInputRef.current.files[0];
        const formData = new FormData();
        formData.append("file", file);
        const res = await api.post("/users/me/profile-image", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        const newUrl: string = res.data.profile_image_url;
        updateProfileImage(newUrl);
        setPreviewUrl(resolveUrl(newUrl));
        if (fileInputRef.current) fileInputRef.current.value = "";
        changed = true;
      }

      // Save username / email changes if they differ
      const trimmedUsername = username.trim();
      const trimmedEmail = email.trim();
      const usernameChanged = trimmedUsername && trimmedUsername !== user?.username;
      const emailChanged = trimmedEmail !== (user?.email ?? "");

      if (usernameChanged || emailChanged) {
        const payload: Record<string, string> = {};
        if (usernameChanged) payload.username = trimmedUsername;
        if (emailChanged) payload.email = trimmedEmail;
        const res = await api.patch("/users/me", payload);
        // Re-sync auth store with returned user data
        login(res.data);
        // Invalidate student profile cache so the Contact section refreshes
        queryClient.invalidateQueries({ queryKey: ["studentProfile"] });
        changed = true;
      }

      if (changed) {
        setSuccess("Profile updated successfully.");
        setTimeout(() => closeEditProfile(), 900);
      } else {
        closeEditProfile();
      }
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Failed to save changes. Please try again.",
      );
    } finally {
      setIsSaving(false);
    }
  };

  if (!user) return null;

  const initial = user.username?.charAt(0)?.toUpperCase() ?? "?";

  return (
    <Modal
      isOpen={isEditProfileOpen}
      onClose={closeEditProfile}
      title="Edit Profile"
      className="max-w-md"
    >
      <div className="space-y-6 pt-4">
        {/* Profile Photo */}
        <div className="flex flex-col items-center gap-3">
          <div className="relative group">
            <div className="w-24 h-24 rounded-full overflow-hidden bg-gray-100 dark:bg-white/5 border-2 border-dashed border-gray-300 dark:border-white/20 flex items-center justify-center">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Profile"
                  className="w-full h-full object-cover"
                  onError={() => setPreviewUrl(null)}
                />
              ) : (
                <span className="text-3xl text-gray-400 dark:text-gray-500 font-bold select-none">
                  {initial}
                </span>
              )}
            </div>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="absolute bottom-0 right-0 p-2 bg-primary text-white rounded-full hover:bg-primary/90 shadow-lg transition-transform transform group-hover:scale-110"
              title="Change profile picture"
            >
              <Camera size={16} />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Click the camera icon to change your photo
          </p>
        </div>

        {/* Status messages */}
        {error && (
          <div className="flex items-center gap-2 text-sm text-red-400">
            <AlertCircle size={14} className="shrink-0" /> {error}
          </div>
        )}
        {success && (
          <div className="flex items-center gap-2 text-sm text-emerald-400">
            <CheckCircle size={14} className="shrink-0" /> {success}
          </div>
        )}

        {/* Editable fields */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Username
            </label>
            <Input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Username"
              className="glass-input"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Email
            </label>
            <Input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email address"
              className="glass-input"
            />
          </div>

          {/* Role — read-only */}
          <div className="flex justify-between items-center py-2 border-b border-gray-100 dark:border-white/5">
            <span className="text-sm text-gray-500 dark:text-gray-400">Role</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {user.role ?? "—"}
            </span>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 pt-2 border-t border-gray-100 dark:border-white/10">
          <Button variant="ghost" onClick={closeEditProfile} disabled={isSaving}>
            Cancel
          </Button>
          <Button onClick={handleSave} isLoading={isSaving} disabled={isSaving}>
            Save Changes
          </Button>
        </div>
      </div>
    </Modal>
  );
}
