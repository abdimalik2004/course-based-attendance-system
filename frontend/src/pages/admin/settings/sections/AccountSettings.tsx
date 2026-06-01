import { useState, useEffect, useRef } from 'react';
import { CheckCircle, AlertCircle, Camera, Trash2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { api } from '@/services/api';
import placeholderFace from '@/assets/logo.png';

interface MeData {
  id: number;
  username: string;
  email: string | null;
  profile_image_url: string | null;
  role_names: string[];
}

export function AccountSettings() {
  const [me, setMe] = useState<MeData | null>(null);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState('');
  const [saveError, setSaveError] = useState('');

  // Image upload state
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');

  // Load current user on mount
  useEffect(() => {
    api.get('/auth/me').then((res) => {
      const data: MeData = res.data;
      setMe(data);
      setUsername(data.username ?? '');
      setEmail(data.email ?? '');
      setAvatarUrl(data.profile_image_url ?? null);
    }).catch(() => {});
  }, []);

  const handleFieldChange = (setter: (v: string) => void) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setter(e.target.value);
    setHasChanges(true);
    setSaveSuccess('');
    setSaveError('');
  };

  const handleSave = async () => {
    if (!me) return;
    setSaveError('');
    setSaveSuccess('');
    setIsSaving(true);
    try {
      await api.put(`/users/${me.id}`, { username, email: email || null });
      setSaveSuccess('Profile updated successfully.');
      setHasChanges(false);
      setTimeout(() => setSaveSuccess(''), 4000);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setSaveError(typeof detail === 'string' ? detail : 'Failed to update profile.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    if (!me) return;
    setUsername(me.username ?? '');
    setEmail(me.email ?? '');
    setHasChanges(false);
    setSaveError('');
    setSaveSuccess('');
  };

  const handleAvatarClick = () => fileInputRef.current?.click();

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError('');
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await api.post('/users/me/profile-image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setAvatarUrl(res.data.profile_image_url);
    } catch (err: any) {
      setUploadError('Image upload failed. Please try a different file.');
    } finally {
      setIsUploading(false);
      // Reset file input so the same file can be re-selected if needed
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemovePicture = async () => {
    // Optimistically clear avatar; no dedicated DELETE endpoint, so just clear locally
    setAvatarUrl(null);
    setUploadError('');
  };

  const displayAvatar = avatarUrl
    ? (avatarUrl.startsWith('http') ? avatarUrl : `${import.meta.env.VITE_API_URL ?? ''}${avatarUrl}`)
    : placeholderFace;

  return (
    <div className="space-y-6">
      <div className="border-b border-white/5 pb-4">
        <h2 className="text-xl font-bold text-white mb-1">Account Settings</h2>
        <p className="text-sm text-gray-400">Update your profile information and administrative details.</p>
      </div>

      <Card className="glass-card border-white/5">
        <CardContent className="p-6 space-y-8">

          {/* Avatar */}
          <div className="flex flex-col sm:flex-row items-center gap-6">
            <div className="relative shrink-0">
              <img
                src={displayAvatar}
                alt="Profile"
                className="w-24 h-24 rounded-full object-cover border-4 border-white/10"
              />
              <button
                onClick={handleAvatarClick}
                disabled={isUploading}
                className="absolute bottom-0 right-0 bg-primary w-8 h-8 rounded-full flex items-center justify-center text-white border-2 border-[#111827] hover:bg-primary-hover transition-colors disabled:opacity-50"
                title="Change profile picture"
              >
                <Camera size={13} />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>
            <div>
              <h3 className="text-white font-medium text-lg">Profile Picture</h3>
              <p className="text-gray-400 text-sm mb-3">PNG or JPG, up to 5 MB.</p>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleRemovePicture}
                disabled={!avatarUrl || isUploading}
                className="gap-2"
              >
                <Trash2 size={14} />
                Remove Picture
              </Button>
              {uploadError && (
                <p className="text-xs text-red-400 mt-2 flex items-center gap-1">
                  <AlertCircle size={12} /> {uploadError}
                </p>
              )}
              {isUploading && (
                <p className="text-xs text-gray-400 mt-2">Uploading…</p>
              )}
            </div>
          </div>

          {/* Role badge */}
          {me && me.role_names.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm text-gray-400">Roles:</span>
              {me.role_names.map((r) => (
                <span
                  key={r}
                  className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary border border-primary/20"
                >
                  {r}
                </span>
              ))}
            </div>
          )}

          {/* Fields */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Username</label>
              <Input
                value={username}
                onChange={handleFieldChange(setUsername)}
                placeholder="your_username"
                className="bg-white/5 border-white/10"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
              <Input
                type="email"
                value={email}
                onChange={handleFieldChange(setEmail)}
                placeholder="you@example.com"
                className="bg-white/5 border-white/10"
              />
            </div>
          </div>

          {saveError && (
            <div className="flex items-center gap-2 text-sm text-red-400">
              <AlertCircle size={14} className="shrink-0" />
              <span>{saveError}</span>
            </div>
          )}
          {saveSuccess && (
            <div className="flex items-center gap-2 text-sm text-emerald-400">
              <CheckCircle size={14} />
              <span>{saveSuccess}</span>
            </div>
          )}

          <div className="pt-4 flex items-center gap-3">
            <Button disabled={!hasChanges || isSaving} onClick={handleSave} className="min-w-[140px]">
              {isSaving ? 'Saving…' : 'Update Profile'}
            </Button>
            <Button variant="ghost" onClick={handleCancel} disabled={isSaving}>Cancel</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
