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
