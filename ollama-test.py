import unittest
import requests
import subprocess
import platform
import time
import os
import signal

# Ollama API endpoint and model name (can be overridden with env var)
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "smollm2:135m")

class TestOllamaServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Launch the Ollama server as a background subprocess
        cls.ollama_proc = subprocess.Popen(["ollama", "serve"])
        time.sleep(10)  # wait for server to boot

        # Pull the specified model before running tests
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)

    @classmethod
    def tearDownClass(cls):
        # Gracefully shut down the Ollama server
        if platform.system() == "Windows":
            cls.ollama_proc.terminate()     # Use terminate() for Windows
        else:
            cls.ollama_proc.send_signal(signal.SIGINT) # Use SIGINT for Unix-based systems
        cls.ollama_proc.wait()

    def test_server_responds(self):
        response = requests.get(f"{OLLAMA_URL}/api/tags")

        # -- Assertions for getting a list of available models -- #
        self.assertEqual(response.status_code, 200)
        self.assertIn("models", response.json())

    def test_generation(self):
        # -- Sample dummy request -- #
        payload = {
            "model": MODEL_NAME,
            "prompt": "Say hello",
            "stream": False
        }
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)

        # -- Assertions for obtaining a response back from the model -- #
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())

if __name__ == "__main__":
    unittest.main()
