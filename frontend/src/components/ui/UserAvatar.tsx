/**
 * UserAvatar — shows the user's profile picture if one exists, otherwise
 * falls back to a coloured circle with the first letter of the username.
 *
 * Usage:
 *   <UserAvatar imageUrl={user?.profile_image_url} username={user?.username} size={36} />
 */

const API_URL = import.meta.env.VITE_API_URL ?? '';

interface UserAvatarProps {
  imageUrl?: string | null;
  username?: string | null;
  /** Diameter in pixels — applied as both width and height. Default: 36 */
  size?: number;
  className?: string;
}

export function UserAvatar({ imageUrl, username, size = 36, className = '' }: UserAvatarProps) {
  // Relative URLs (e.g. /static/uploads/...) must be prefixed with the API base URL
  const resolvedUrl = imageUrl
    ? imageUrl.startsWith('http')
      ? imageUrl
      : `${API_URL}${imageUrl}`
    : null;

  const initial = username ? username.charAt(0).toUpperCase() : '?';

  const style: React.CSSProperties = { width: size, height: size, minWidth: size, minHeight: size };

  if (resolvedUrl) {
    return (
      <img
        src={resolvedUrl}
        alt={username ?? 'User'}
        style={style}
        className={`rounded-full object-cover ${className}`}
        onError={(e) => {
          // If the image fails to load, hide it and show the initial instead
          (e.currentTarget as HTMLImageElement).style.display = 'none';
          const sibling = (e.currentTarget as HTMLImageElement).nextSibling as HTMLElement | null;
          if (sibling) sibling.style.display = 'flex';
        }}
      />
    );
  }

  return (
    <span
      style={style}
      className={`rounded-full flex items-center justify-center bg-primary/20 text-primary font-semibold text-sm select-none ${className}`}
    >
      {initial}
    </span>
  );
}
