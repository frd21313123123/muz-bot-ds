import 'dotenv/config';
import { REST, Routes, InteractionContextType } from 'discord.js';
import { commands } from './commands.js';

async function main(): Promise<void> {
  const { DISCORD_TOKEN, CLIENT_ID, GUILD_ID } = process.env;
  if (!DISCORD_TOKEN || !CLIENT_ID) throw new Error('Укажите DISCORD_TOKEN и CLIENT_ID в .env');
  const route = GUILD_ID
    ? Routes.applicationGuildCommands(CLIENT_ID, GUILD_ID)
    : Routes.applicationCommands(CLIENT_ID);
  const payload = commands.map((command) => command.data.setContexts(InteractionContextType.Guild).toJSON());
  const result = await new REST().setToken(DISCORD_TOKEN).put(route, { body: payload });
  console.log(`✅ Зарегистрировано ${Array.isArray(result) ? result.length : payload.length} команд.`);
}

main().catch((error) => { console.error('Ошибка регистрации команд:', error); process.exitCode = 1; });
