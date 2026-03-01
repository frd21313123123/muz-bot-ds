'use strict';

require('dotenv').config();

const { REST, Routes } = require('discord.js');
const fs = require('fs');
const path = require('path');

const { DISCORD_TOKEN, CLIENT_ID, GUILD_ID } = process.env;

if (!DISCORD_TOKEN || !CLIENT_ID) {
  console.error('❌ Укажите DISCORD_TOKEN и CLIENT_ID в файле .env');
  process.exit(1);
}

const commands = [];
const commandsDir = path.join(__dirname, 'commands');

for (const file of fs.readdirSync(commandsDir).filter(f => f.endsWith('.js'))) {
  const { data } = require(path.join(commandsDir, file));
  if (data) commands.push(data.toJSON());
}

const rest = new REST().setToken(DISCORD_TOKEN);

(async () => {
  try {
    console.log(`Регистрация ${commands.length} слэш-команд...`);

    // Если указан GUILD_ID — регистрируем на конкретном сервере (мгновенно)
    // Если нет — глобально (до 1 часа на распространение)
    const route = GUILD_ID
      ? Routes.applicationGuildCommands(CLIENT_ID, GUILD_ID)
      : Routes.applicationCommands(CLIENT_ID);

    const data = await rest.put(route, { body: commands });
    console.log(`✅ Зарегистрировано ${data.length} команд${GUILD_ID ? ` на сервере ${GUILD_ID}` : ' глобально'}!`);
  } catch (error) {
    console.error('Ошибка регистрации команд:', error);
  }
})();
