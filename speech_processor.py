import numpy as np
import whisper
import threading
import queue
import time
import re
import sounddevice as sd
import json
import logging
import sys
import torch
import paho.mqtt.client as mqtt
import os
from datetime import datetime
from collections import deque

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', "u6edac82.ala.us-east-1.emqxsl.com")
MQTT_PORT = int(os.getenv('MQTT_PORT', "8883"))
MQTT_USERNAME = os.getenv('MQTT_USERNAME', "test_user")
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', "test_pass")
MQTT_CERT_PATH = os.getenv('MQTT_CERT_PATH', "emqxsl-ca.crt")
MQTT_TOPIC = "iot/display/message"

# Audio configuration
CHANNELS = 1
RATE = 16000
CHUNK = 1024 * 2  # Increase chunk size
BUFFER_SECONDS = 5  # Increase buffer duration


class AudioTranscriptionProcessor:
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

        self.initialize_mqtt()

        logging.info("Loading Whisper model...")
        self.model = whisper.load_model("base")  # Use base Whisper model
        logging.info("Model loaded.")

        self.audio_queue = queue.Queue()
        self.running = True
        self.last_process_time = time.time()

        # Set up device for computation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

    def initialize_mqtt(self):
        """Initialize MQTT client"""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.mqtt_client.tls_set(ca_certs=MQTT_CERT_PATH)

        # Attach MQTT event handlers
        self.mqtt_client.on_connect = self._handle_mqtt_connect
        self.mqtt_client.on_disconnect = self._handle_mqtt_disconnect

        # Attempt to connect to MQTT broker
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
            self.mqtt_client.loop_start()
            logging.info("MQTT connection established")
        except Exception as e:
            logging.error(f"Failed to connect to MQTT: {e}")

    def _handle_mqtt_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection"""
        if rc == 0:
            logging.info("Connected to MQTT broker")
        else:
            logging.error(f"MQTT connection error: {rc}")

    def _handle_mqtt_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        logging.warning(f"Disconnected from MQTT broker: {rc}")
        if rc != 0:
            logging.info("Attempting to reconnect...")

    def clean_transcription(self, text):
        """Clean up transcribed text by removing unwanted characters and repetitions"""
        text = re.sub(r'[^a-zA-Zа-яА-ЯąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9.,!? ]', '', text)
        words = text.split()
        return ' '.join(word for i, word in enumerate(words)
                        if i == 0 or word != words[i - 1])

    def check_silence(self, audio_data, threshold=0.01):
        """Check if the audio data contains silence"""
        return np.abs(audio_data).max() < threshold

    def audio_input_callback(self, indata, frames, time, status):
        """Callback to handle audio input"""
        if status:
            logging.warning(f"Audio stream status: {status}")
        try:
            self.audio_queue.put(indata.copy())
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    def process_transcription(self, audio_data):
        """Transcribe audio and publish the result to MQTT"""
        try:
            result = self.model.transcribe(
                audio_data,
                language="pl",
                fp16=(self.device == "cuda"),
                temperature=0.2
            )

            # Clean the transcribed text
            text = self.clean_transcription(result["text"]).strip()
            if text:
                # Prepare the MQTT message
                message = {"text": text}

                # Log the message for debugging
                logging.info(f"Sending message: {json.dumps(message)}")

                # Publish the message to MQTT
                mqtt_result = self.mqtt_client.publish(
                    MQTT_TOPIC,
                    json.dumps(message),
                    qos=1,
                    retain=True
                )

                if mqtt_result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logging.info(f"Message successfully sent: {text}")
                else:
                    logging.error(f"MQTT send error: {mqtt_result.rc}")

        except Exception as e:
            logging.error(f"Transcription error: {e}")

    def audio_processing_loop(self):
        """Process audio data in chunks and handle transcription"""
        buffer = []
        samples_per_buffer = int(RATE * BUFFER_SECONDS)

        while self.running:
            try:
                # Collect audio chunks
                while len(buffer) < samples_per_buffer:
                    if not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        buffer.extend(chunk.flatten())
                    else:
                        time.sleep(0.1)
                        continue

                # Convert to numpy array
                audio_data = np.array(buffer[:samples_per_buffer], dtype=np.float32)
                buffer = buffer[samples_per_buffer:]  # Remove processed data

                # Skip silent audio
                if not self.check_silence(audio_data):
                    self.process_transcription(audio_data)

            except Exception as e:
                logging.error(f"Audio processing error: {e}")
                time.sleep(0.1)

    def list_audio_devices(self):
        """List all available audio input devices"""
        devices = sd.query_devices()
        logging.info("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            logging.info(f"{i}: {device['name']} (max inputs: {device['max_input_channels']})")
        return devices

    def start(self):
        """Start audio processing and transcription"""
        try:
            devices = self.list_audio_devices()

            # Select the device with the most input channels
            device_index = max(range(len(devices)),
                               key=lambda i: devices[i]['max_input_channels'])

            logging.info(f"\nUsing device {device_index}: {devices[device_index]['name']}")

            # Start the audio processing thread
            process_thread = threading.Thread(target=self.audio_processing_loop)
            process_thread.start()

            # Start recording audio
            with sd.InputStream(device=device_index,
                                channels=CHANNELS,
                                samplerate=RATE,
                                blocksize=CHUNK,
                                callback=self.audio_input_callback):
                logging.info("Recording started. Press Ctrl+C to stop.")
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logging.info("Stopping...")
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            self.running = False
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logging.info("Resources released")


if __name__ == "__main__":
    processor = AudioTranscriptionProcessor()
    processor.start()
