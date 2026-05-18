import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Camera, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Modal } from "@/components/ui/Modal";
import { useAuthStore } from "@/store/useAuthStore";
import { useUIStore } from "@/store/useUIStore";
import { fileService } from "@/services/fileService";

export function EditProfileModal() {
  const { user } = useAuthStore();
  const { isEditProfileOpen, closeEditProfile } = useUIStore();
  const [isSaving, setIsSaving] = useState(false);
  const [profileImage, setProfileImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setProfileImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      if (
        fileInputRef.current?.files &&
        fileInputRef.current.files.length > 0
      ) {
        const file = fileInputRef.current.files[0];
        await fileService.uploadProfileImage(file);
      }
      closeEditProfile();
    } catch (err) {
      console.error("Failed to upload profile image", err);
      alert("Failed to upload profile image.");
    } finally {
      setIsSaving(false);
    }
  };

  if (!user) return null;

  return (
    <Modal
      isOpen={isEditProfileOpen}
      onClose={closeEditProfile}
      title="Edit Profile"
      className="max-w-md"
    >
      <div className="space-y-6 pt-4">
        {/* Profile Photo Upload */}
        <div className="flex flex-col items-center gap-4">
          <div className="relative group">
            <div className="w-24 h-24 rounded-full overflow-hidden bg-gray-100 dark:bg-white/5 border-2 border-dashed border-gray-300 dark:border-white/20 flex items-center justify-center">
              {profileImage ? (
                <img
                  src={profileImage}
                  alt="Profile"
                  className="w-full h-full object-cover"
                />
              ) : (
                <span className="text-3xl text-gray-400 font-bold">
                  {user.username?.charAt(0)?.toUpperCase()}
                </span>
              )}
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="absolute bottom-0 right-0 p-2 bg-primary text-white rounded-full hover:bg-primary-hover shadow-lg transition-transform transform group-hover:scale-110"
            >
              <Camera size={16} />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleImageUpload}
            />
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Click the camera icon to update photo
          </p>
        </div>

        {/* Form Fields (Read-Only) */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
              Username / Name
            </label>
            <Input
              value={user.username}
              readOnly
              className="bg-gray-50 dark:bg-white/5 text-gray-500 cursor-not-allowed border-gray-200 dark:border-white/10"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
              Email
            </label>
            <Input
              value={`${user.username?.toLowerCase().replace(/\s+/g, "")}@heegan.edu.so`}
              readOnly
              className="bg-gray-50 dark:bg-white/5 text-gray-500 cursor-not-allowed border-gray-200 dark:border-white/10"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 ml-1">
              Password
            </label>
            <Input
              type="password"
              value="••••••••••••"
              readOnly
              className="bg-gray-50 dark:bg-white/5 text-gray-500 cursor-not-allowed border-gray-200 dark:border-white/10"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 ml-1">
              Password changes are disabled in this demo.
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-100 dark:border-white/10">
          <Button
            variant="ghost"
            onClick={closeEditProfile}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button onClick={handleSave} isLoading={isSaving}>
            Save Changes
          </Button>
        </div>
      </div>
    </Modal>
  );
}
