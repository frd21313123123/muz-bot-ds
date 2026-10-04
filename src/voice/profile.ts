import path from 'node:path';

export type VoiceProfile = 'light' | 'heavy';
export function voiceProfile(value = process.env.VOICE_PROFILE): VoiceProfile {
  const profile = value?.trim().toLowerCase() || 'light';
  if (profile !== 'light' && profile !== 'heavy') throw new Error('VOICE_PROFILE: light или heavy.');
  return profile;
}
export function voiceDirectory(profile: VoiceProfile): string {
  return path.resolve('.runtime', profile === 'light' ? 'voice-light' : 'voice');
}
