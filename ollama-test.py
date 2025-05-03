#!/usr/bin/env python3
"""
Ollama cross-platform integration test suite (pytest-compatible).
"""

import os
import time
import json
import platform
import subprocess
import requests
import pytest

BASE_URL = "http://localhost:11434/api"
MODEL_NAME = "smollm2:135m"
AVG_LATENCY = None


def is_server_running():
    try:
        response = requests.get(f"{BASE_URL}/tags")
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def start_ollama_server():
    system = platform.system().lower()
    try:
        if system == "windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            subprocess.Popen(["ollama", "serve"])
        time.sleep(5)
    except Exception as e:
        pytest.exit(f"Failed to start Ollama server: {e}")


def ensure_model_available():
    try:
        response = requests.get(f"{BASE_URL}/tags")
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]

        if MODEL_NAME not in model_names:
            print(f"Pulling model {MODEL_NAME}...")
            subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        # Wait until model is fully loaded
        for _ in range(10):
            res = requests.get(f"{BASE_URL}/tags")
            loaded_models = [m["name"] for m in res.json().get("models", [])]
            if MODEL_NAME in loaded_models:
                return
            print("Waiting for model to load...")
            time.sleep(10)
    except Exception as e:
        pytest.exit(f"Failed to ensure model availability: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    print("Preparing Ollama server and model...")
    if not is_server_running():
        start_ollama_server()
    assert is_server_running(), "Ollama server failed to start"
    ensure_model_available()


def test_model_availability():
    response = requests.get(f"{BASE_URL}/tags")
    if response.status_code != 200:
        print("🚨 Ollama Model Fetch Failed!")
        print(f"Status code: {response.status_code}")
        try:
            print("Response JSON:", response.json())
        except ValueError:
            print("Raw response text:", response.text)
        pytest.fail("Ollama generation returned 500")
    
    models = response.json()["models"]
    model_names = [model["name"] for model in models]
    assert MODEL_NAME in model_names


def test_basic_generation():
    payload = {
        "model": MODEL_NAME,
        "prompt": "What is machine learning?",
        "stream": False
    }
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    end_time = time.time()

    if response.status_code != 200:
        print("🚨 Ollama generation failed!")
        print(f"Status code: {response.status_code}")
        try:
            print("Response JSON:", response.json())
        except ValueError:
            print("Raw response text:", response.text)
        pytest.fail("Ollama generation returned 500")

    result = response.json()
    assert "response" in result
    assert len(result["response"]) > 20

    print(f"Generation time: {end_time - start_time:.2f}s")
    print(f"Response snippet: {result['response'][:100]}...")


def test_inference_latency():
    global AVG_LATENCY

    payload = {
        "model": MODEL_NAME,
        "prompt": "Explain quantum computing in 50 words",
        "stream": False
    }

    num_runs = 3
    latencies = []

    for i in range(num_runs):
        start = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        end = time.time()
        if response.status_code != 200:
            print("🚨 Ollama generation failed!")
            print(f"Status code: {response.status_code}")
            try:
                print("Response JSON:", response.json())
            except ValueError:
                print("Raw response text:", response.text)
            pytest.fail("Ollama generation returned 500")

        latency = end - start
        latencies.append(latency)
        print(f"Run {i+1}: {latency:.2f}s")
        time.sleep(1)

    AVG_LATENCY = sum(latencies) / len(latencies)
    print(f"Average latency: {AVG_LATENCY:.2f}s")
    assert AVG_LATENCY <= 5.0


def test_generate_report():
    global AVG_LATENCY
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "model": MODEL_NAME,
        "test_timestamp": time.time(),
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["ollama_version"] = result.stdout.strip()
    except Exception:
        info["ollama_version"] = "unknown"

    if AVG_LATENCY is not None:
        info["avg_inference_latency"] = round(AVG_LATENCY, 3)

    os.makedirs("test-results", exist_ok=True)
    report_path = os.path.join("test-results", "system_report.json")
    with open(report_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"System report saved to {report_path}")
    assert os.path.exists(report_path)
