## Starting up the tool sandbox 
Make sure Docker daemon is running and run:
` docker compose up -d `

## Start up the server
Run the following (assuming the virtual env is in /.venv):
` .venv/bin/python -m agent_bot.server  `
for the Tkinter GUI:
` .venv/bin/python gui/app.py `