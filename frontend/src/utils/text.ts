const AUDIO_EXTENSIONS = [".wav", ".m4a", ".mp3", ".flac", ".ogg"];

export function stripFileExtension(value: string | null | undefined): string {
  if (!value) {
    return "";
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  const lower = trimmed.toLowerCase();
  const match = AUDIO_EXTENSIONS.find((ext) => lower.endsWith(ext));
  if (!match) {
    return trimmed;
  }
  return trimmed.slice(0, -match.length) || trimmed;
}

