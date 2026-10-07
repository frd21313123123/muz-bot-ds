# MUZ landing page

English, responsive landing page for the Discord music bot. No npm dependencies.

Production: https://wave-projects.com/ on the bot's server. Nginx configuration is tracked in `deployment/wave-projects.nginx.conf`. Build the page, copy only `landing/dist/` into `/var/www/muz-landing`, validate with `nginx -t`, and reload nginx. The existing certificate and ACME challenge route are retained. The independent application at `ocenochka.wave-projects.com` keeps its own configuration.

Build with `node landing/build.mjs` from the repository root. Serve `landing/dist` with any static web server. The player is an interactive visual demonstration and does not play audio or control the bot. Discord invitation links use the application's public client ID and request View Channels, Send Messages, Embed Links, Connect, Speak and Use Voice Activity permissions.

Features and command descriptions reflect the bot's README. Voice command examples remain Russian because the bot's voice controls support Russian requests. An administrator must keep the bot running for music playback to work.

Source files are tracked; generated `dist/` files remain ignored. Sites publishing uses a separate checkout so the bot's source and local configuration are never uploaded.
