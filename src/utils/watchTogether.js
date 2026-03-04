'use strict';

const { InviteTargetType, PermissionsBitField } = require('discord.js');

const WATCH_TOGETHER_APP_ID = process.env.WATCH_TOGETHER_APP_ID?.trim() || '880218394199220334';

/**
 * Проверяет, может ли бот создать invite для активности в voice-канале.
 * @param {import('discord.js').VoiceBasedChannel} voiceChannel
 * @param {import('discord.js').GuildMember|null} me
 */
function canCreateWatchTogetherInvite(voiceChannel, me) {
  if (!voiceChannel || !me) return false;
  const perms = voiceChannel.permissionsFor(me);
  return Boolean(
    perms?.has(PermissionsBitField.Flags.ViewChannel) &&
    perms?.has(PermissionsBitField.Flags.CreateInstantInvite) &&
    perms?.has(PermissionsBitField.Flags.UseEmbeddedActivities),
  );
}

/**
 * Создает invite на Discord Activity "Watch Together".
 * @param {import('discord.js').VoiceBasedChannel} voiceChannel
 * @param {string} requestedByTag
 */
async function createWatchTogetherInvite(voiceChannel, requestedByTag) {
  return voiceChannel.createInvite({
    maxAge: 0,
    maxUses: 0,
    temporary: false,
    targetType: InviteTargetType.EmbeddedApplication,
    targetApplication: WATCH_TOGETHER_APP_ID,
    reason: `Watch Together requested by ${requestedByTag}`,
  });
}

module.exports = {
  WATCH_TOGETHER_APP_ID,
  canCreateWatchTogetherInvite,
  createWatchTogetherInvite,
};
