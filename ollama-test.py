#!/usr/bin/env python3
"""
Ollama cross-platform integration test suite.
This file tests Ollama functionality across different operating systems.
"""

import os
import sys
import time
import json
import platform
import unittest
import subprocess
import requests


class TestOllamaIntegration(unittest.TestCase):
    """Test suite for validating Ollama functionality across platforms."""
    
    # Model to use for testing
    MODEL_NAME = "smollm2:135m"
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and ensure Ollama server is running."""
        cls.base_url = "http://localhost:11434/api"
        
        # Give Ollama server time to fully initialize if already running
        print("Waiting for Ollama server to be ready...")
        time.sleep(3)
        
        # Check if Ollama server is running
        try:
            cls._check_server_status()
        except requests.ConnectionError:
            print("Ollama server not running, attempting to start it...")
            cls._start_ollama_server()
            
        # Pull the test model if not already available
        cls._ensure_model_available()
        
    @classmethod
    def _check_server_status(cls):
        """Check if Ollama server is running."""
        response = requests.get(f"{cls.base_url}/tags")
        if response.status_code != 200:
            raise ConnectionError(f"Ollama server returned status code {response.status_code}")
        return True
        
    @classmethod
    def _start_ollama_server(cls):
        """Start Ollama server based on platform."""
        system = platform.system().lower()
        
        try:
            if system == "windows":
                # Windows startup
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                # Unix-like startup
                process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
            # Wait for server to start
            time.sleep(5)
            
            # Verify server is running
            cls._check_server_status()
            print("Ollama server started successfully")
            
        except Exception as e:
            print(f"Failed to start Ollama server: {e}")
            sys.exit(1)
    
    @classmethod
    def _ensure_model_available(cls):
        """Ensure test model is available, download if needed."""
        try:
            # Check if model exists
            response = requests.get(f"{cls.base_url}/tags")
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if cls.MODEL_NAME not in model_names:
                print(f"Pulling model {cls.MODEL_NAME}...")
                subprocess.run(["ollama", "pull", cls.MODEL_NAME], check=True)
                print(f"Model {cls.MODEL_NAME} pulled successfully")
            else:
                print(f"Model {cls.MODEL_NAME} already available")
                
        except Exception as e:
            print(f"Failed to ensure model availability: {e}")
            sys.exit(1)

    def test_model_availability(self):
        """Test that our required model is available."""
        response = requests.get(f"{self.base_url}/tags")
        self.assertEqual(response.status_code, 200, "Failed to get model tags")
        
        models = response.json()["models"]
        model_names = [model["name"] for model in models]
        self.assertIn(self.MODEL_NAME, model_names, 
                      f"Required model {self.MODEL_NAME} not found")
        
    def test_basic_generation(self):
        """Test that the model can generate text."""
        payload = {
            "model": self.MODEL_NAME,
            "prompt": "What is machine learning?",
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/generate", json=payload)
        end_time = time.time()
        
        self.assertEqual(response.status_code, 200, 
                         f"Generation failed with status {response.status_code}")
        
        result = response.json()
        self.assertIn("response", result, "No response field in result")
        
        # Verify response has meaningful content
        self.assertTrue(len(result["response"]) > 20, 
                        "Response too short, possibly invalid")
        
        # Check response time (adjust threshold based on model and hardware)
        elapsed_time = end_time - start_time
        print(f"Generation took {elapsed_time:.2f} seconds")
        
        # Output some stats for the CI logs
        print(f"Response length: {len(result['response'])} characters")
        print(f"Response snippet: {result['response'][:100]}...")
        
        
    def test_inference_latency(self):
        """Test that inference completes within acceptable time. (5 seconds for testing)"""
        payload = {
            "model": self.MODEL_NAME,
            "prompt": "Explain quantum computing in 50 words",
            "stream": False
        }
        
        # Run multiple times to get average latency
        num_runs = 3
        latencies = []
        
        for i in range(num_runs):
            start_time = time.time()
            response = requests.post(f"{self.base_url}/generate", json=payload)
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200, 
                            f"Generation failed with status {response.status_code}")
            
            elapsed_time = end_time - start_time
            latencies.append(elapsed_time)
            print(f"Run {i+1}: Inference took {elapsed_time:.2f} seconds")
            
            # Short pause between runs
            time.sleep(1)
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average inference latency: {avg_latency:.2f} seconds")
        assert avg_latency <= 5.0
        
        
    def test_generate_report(self):
        """Generate a JSON report with platform and model info for CI artifacts."""
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "model": self.MODEL_NAME,
            "test_timestamp": time.time(),
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # Get Ollama version
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                system_info["ollama_version"] = result.stdout.strip()
        except Exception:
            system_info["ollama_version"] = "unknown"
            
        # Create reports directory if it doesn't exist
        os.makedirs("test-results", exist_ok=True)
        
        # Save report to file
        report_path = os.path.join("test-results", "system_report.json")
        with open(report_path, "w") as f:
            json.dump(system_info, f, indent=2)
            
        print(f"System report saved to {report_path}")
        self.assertTrue(os.path.exists(report_path), 
                       "Failed to create system report file")


if __name__ == "__main__":
    unittest.main()