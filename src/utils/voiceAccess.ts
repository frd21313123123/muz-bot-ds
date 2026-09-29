import type { ChatInputCommandInteraction, ButtonInteraction, VoiceBasedChannel } from 'discord.js';

type VoiceInteraction = ChatInputCommandInteraction | ButtonInteraction;

export function memberVoiceChannel(interaction: VoiceInteraction): VoiceBasedChannel | null {
  const cached = interaction.guild?.voiceStates.cache.get(interaction.user.id)?.channel;
  if (cached) return cached;
  const member = interaction.member;
  if (!member || !('voice' in member)) return null;
  return member.voice.channel;
}

export function canControl(interaction: VoiceInteraction, botChannelId: string | null): boolean {
  return Boolean(botChannelId && memberVoiceChannel(interaction)?.id === botChannelId);
}
