import multiprocessing
import subprocess

# Function to run a Python script with its full path
def run_script(script_path):
    subprocess.run(["python", script_path])

# Function to run Flask server
def run_flask():
    subprocess.run(["flask", "run"])

if __name__ == "__main__":
    # List of full paths to Python scripts
    scripts = [ 
        "aditya comps/dabba_code.py",
        "drowsy_detection/trial.py",
    ]

    # Create processes for each script
    processes = []

    # Add script processes
    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(script,))
        processes.append(p)
        p.start()  # Start each script concurrently

    # Add Flask process
    flask_process = multiprocessing.Process(target=run_flask)
    processes.append(flask_process)
    flask_process.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All scripts have finished execution.")
