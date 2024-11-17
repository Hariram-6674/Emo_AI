from flask import Flask, render_template, request
from twilio.rest import Client
import random
from flask import jsonify
import pyttsx3

def speak_text(text):
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()
    
    # Get and set speaking rate
    rate = engine.getProperty('rate')
    print(f"Current rate: {rate}")  # optional print to see the current rate
    engine.setProperty('rate', 125)  # set desired speaking rate
    
    # Get and set volume
    volume = engine.getProperty('volume')
    print(f"Current volume: {volume}")  # optional print to see the current volume level
    engine.setProperty('volume', 1.0)  # set maximum volume
    
    # Set the voice (can be customized for male/female voices, etc.)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Set voice to the first available one
    
    # Speak the given text
    engine.say(text)
    engine.runAndWait()  # Run the speech engine
    
    # Stop the engine after speaking
    engine.stop()

# Example of usage
# speak_text("Hello, welcome to the world of text to speech conversion!")

app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = 'AC423bd1d1ddf7afb96ccc3d2a222051ec'
TWILIO_AUTH_TOKEN = '3a87bd54fa4d1e2160e823bae5447c99'
TWILIO_PHONE_NUMBER = '(385)215-8366'
USER_PHONE_NUMBER = '+91 78240 43672'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

STRESS_THRESHOLD = 90
HEARTBEAT_THRESHOLD = 100
COLLISION_THRESHOLD = 30  # Distance in cm

def generate_stress_data():
    """Generates random stress levels between 0.10 and 100.19."""
    return round(random.uniform(90.0, 100.0), 2)

def generate_heartbeat_data():
    """Generates random heartbeat levels between 60 and 150 BPM."""
    return random.randint(60, 150)

def generate_distance_data():
    """Generates random distance values between 10 and 500 cm."""
    return random.randint(10, 500)

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/api/health')
def health_data():
    """Provide stress, heartbeat, and distance data via API."""
    stress = generate_stress_data()
    heartbeat = generate_heartbeat_data()
    distance = generate_distance_data()

    # Trigger alerts
    if stress > STRESS_THRESHOLD:
        send_sms_alert("stress", stress)
    if heartbeat > HEARTBEAT_THRESHOLD:
        send_sms_alert("heartbeat", heartbeat)
    if distance < COLLISION_THRESHOLD:
        send_sms_alert("collision", distance)

    return jsonify({'stress': stress, 'heartbeat': heartbeat, 'distance': distance})

def send_sms_alert(metric, value):
    """Send SMS alert for stress, heartbeat, or collision detection."""
    try:
        message_body = f"Alert! High {metric} detected! {metric.capitalize()} level: {value}"
        speak_text(message_body)
        if metric == "collision":
            message_body = f"Collision warning! Object detected {value} cm away."
        
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=USER_PHONE_NUMBER
        )
        print(f"Alert sent: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)