import type { PlayerApi } from './player.js';
import type { SpeakerStore } from './store.js';
import { greeting, parsePairingCode, runVoiceCommand } from './commands.js';
import type { CommandResult } from './types.js';

export class AliceAuthError extends Error {
  constructor(message: string) { super(message); this.name = 'AliceAuthError'; }
}

export interface AliceDeps {
  store: SpeakerStore;
  player: PlayerApi;
  skillId: string | null;
  skillName: string;
  log: (kind: string, text: string, ok: boolean) => void;
}

interface AliceSession {
  new?: boolean;
  message_id?: unknown;
  session_id?: unknown;
  skill_id?: string;
  user_id?: string;
  user?: { user_id?: string };
  application?: { application_id?: string };
}

interface AliceBody {
  version?: unknown;
  session?: AliceSession;
  request?: { command?: string; original_utterance?: string; type?: string };
  meta?: { client_id?: string; user_id?: string };
}

export interface AliceResponse {
  version: '1.0';
  session: { message_id: unknown; session_id: unknown };
  response: { text: string; tts: string; end_session: boolean };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function bodyOf(value: unknown): AliceBody {
  if (!isObject(value)) throw new AliceAuthError('Некорректный запрос Диалогов.');
  return value as AliceBody;
}

function stringField(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function deviceIdOf(body: AliceBody): string | null {
  return stringField(body.session?.application?.application_id)
    ?? stringField(body.session?.user?.user_id)
    ?? stringField(body.session?.user_id)
    ?? stringField(body.meta?.user_id);
}

function textResponse(body: AliceBody, text: string, keepSession: boolean): AliceResponse {
  const safe = text.replace(/\s+/gu, ' ').trim().slice(0, 1024) || 'Готово.';
  return {
    version: '1.0',
    session: { message_id: body.session?.message_id ?? 0, session_id: body.session?.session_id ?? '' },
    response: { text: safe, tts: safe, end_session: !keepSession },
  };
}

function hint(skillName: string): string {
  return `Колонка ещё не подключена. Откройте сайт пульта ${skillName}, создайте код и скажите: подключи колонку 123456.`;
}

export function quickAliceResponse(value: unknown, text: string, keepSession = true): AliceResponse {
  return textResponse(bodyOf(value), text, keepSession);
}

/**
 * Handles one Yandex Dialogs request. The caller enforces the 4.5 second
 * response budget; all slow work is allowed to finish in the background after
 * a short acknowledgement has been returned.
 */
export async function handleAliceRequest(value: unknown, deps: AliceDeps): Promise<AliceResponse> {
  const body = bodyOf(value);
  if (deps.skillId && body.session?.skill_id !== deps.skillId) {
    throw new AliceAuthError('Запрос принадлежит другому навыку.');
  }
  const command = stringField(body.request?.command)
    ?? stringField(body.request?.original_utterance)
    ?? '';
  if (!command) return textResponse(body, greeting(deps.skillName), true);

  const code = parsePairingCode(command);
  if (code) {
    const deviceId = deviceIdOf(body);
    if (!deviceId) return textResponse(body, 'Не удалось определить устройство. Попробуйте ещё раз.', true);
    const pairing = deps.store.takeCode(code);
    if (!pairing) return textResponse(body, 'Код неверный или уже истёк. Создайте новый код на сайте.', true);
    const speaker = await deps.store.pairAlice(deviceId, pairing);
    deps.log('alice', `Подключена колонка «${speaker.name}»`, true);
    return textResponse(body, `Колонка «${speaker.name}» подключена. Теперь скажите: включи Кино.`, true);
  }

  const deviceId = deviceIdOf(body);
  const speaker = deviceId ? deps.store.findAliveAlice(deviceId) : undefined;
  if (!speaker) return textResponse(body, hint(deps.skillName), true);

  deps.store.touch(speaker.id);
  const target = { guildId: speaker.guildId, channelId: speaker.channelId };
  let result: CommandResult;
  try {
    result = await runVoiceCommand(deps.player, target, command, deps.skillName);
  } catch (error) {
    console.error('[Alice] command failed:', error);
    result = { text: 'Не получилось выполнить команду.', keepSession: true, ok: false };
  }
  deps.log('alice', `${speaker.name}: ${command}`, result.ok);
  return textResponse(body, result.text, result.keepSession);
}
