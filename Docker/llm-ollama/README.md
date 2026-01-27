# Setup for Ollama and Open WebUI using Docker Compose
The [`docker-compose.yml`](./docker-compose.yml) is configured based on the following principles:
* **Service Separation for Flexible Management**: Defines the inference engine (Ollama) and the interface (Open WebUI) as separate containers. This ensures independent updates and simplifies troubleshooting for each component.
* **Data Persistence**: Utilizes Named Volumes to prevent the loss of model data, chat history, and user settings when containers are removed or updated.
* **GPU Acceleration Support**: Configured to access NVIDIA GPUs from within the container, enabling high-speed AI inference in a local environment.
* **Automatic Restart Policy**: Implements the `unless-stopped` policy to automatically recover services upon system reboot. The AI environment remains active in the background unless manually stopped.
* **Internal Network Integration**: Leverages Docker's service name resolution, allowing the UI to connect to the backend using a simple URL: `http://ollama:11434`.

### Getting Started
After configuring the settings, start the containers:
```sh
docker compose up -d --build
```

Pull the required models:
```sh
docker exec -it ollama ollama pull qwen2.5-coder:14b
docker exec -it ollama ollama pull gemma2:9b
docker exec -it ollama ollama pull qwen2.5-coder:1.5b
docker exec -it ollama ollama pull nomic-embed-text:latest
```

Check the status:
```sh
docker compose ps
```

Ensure that the `open-webui` status is **healthy** before proceeding.
If you encounter any issues, check the logs:
```sh
docker logs -f open-webui
```

To restart the containers:
```sh
docker compose restart
```

### Usage
Once Open WebUI is running, access it via your browser at `localhost:3000`.
On the first run, you will need to register an email and password. Since this is a local installation, any credentials will work.
To verify the setup, select a model from the UI and send a test prompt. If a response is generated, the setup is successful.

**Test Prompt:**

```txt
Explain the difference between Japanese curry rice and Hayashi rice. Specifically, explain the defining elements of Hayashi rice.
```

### Integration with VS Code + Continue
To use these models for coding assistance in Visual Studio Code, follow these steps to configure the [Continue](https://www.continue.dev/) extension using the provided [`config.yaml`](./vscode/config.yaml).

1. **Install the Continue Extension** in VS Code.
2. **Configure the Extension**:
   * Open the Continue config file (usually accessed via the gear icon in the Continue sidebar).
   * Map the settings from the provided `config.yaml` to your Continue configuration.
3. **API Endpoint Note**:
   * Ensure the `apiBase` is set to `http://localhost:11434`.
   * Since the containers are running on your local host, the VS Code extension can communicate directly with Ollama via this address.
4. **Model Roles**:
   * Use **Qwen2.5-Coder 14B** for complex chat, refactoring, and code generation.
   * Use **Qwen2.5-Coder 1.5B** for low-latency autocomplete (Tab-complete).
   * Use **Nomic Embed** for indexing your codebase to enable "Context-aware" answers.

**Verification:**
Highlight a block of code in VS Code and press `Cmd/Ctrl + L` to send it to the chat. Ask, "Refactor this function for better readability." If the 14B model responds, the integration is successful.