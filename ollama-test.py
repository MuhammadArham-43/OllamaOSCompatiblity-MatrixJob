import unittest
import requests
import subprocess
import platform
import time
import os
import signal

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "smollm2:135m")

class TestOllamaServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Start Ollama server in background
        cls.ollama_proc = subprocess.Popen(["ollama", "serve"])
        time.sleep(10)  # wait for server to boot

        # Pull model
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)

    @classmethod
    def tearDownClass(cls):
        # Stop the Ollama server
        if platform.system() == "Windows":
            cls.ollama_proc.terminate()
        else:
            cls.ollama_proc.send_signal(signal.SIGINT)
        cls.ollama_proc.wait()

    def test_server_responds(self):
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        self.assertEqual(response.status_code, 200)
        self.assertIn("models", response.json())

    def test_generation(self):
        payload = {
            "model": MODEL_NAME,
            "prompt": "Say hello",
            "stream": False
        }
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())

if __name__ == "__main__":
    unittest.main()
