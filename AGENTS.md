# Development workflow

- Change code only. Do not run application code, tests, builds, or code validation commands.
- Do not create Python or uv environments, install dependencies, or run code with Python or uv.
- The human runs and verifies changes in a Docker container. Leave execution and runtime validation to the human.
- Read files and inspect diffs as needed. Report what changed and which checks the human should run in Docker; clearly state that those checks were not run by the agent.
