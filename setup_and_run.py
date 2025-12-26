import os
import sys
import subprocess
import time

def print_colored(text, color="green"):
    """Print colored text for better visibility"""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, colors['green'])}{text}{colors['end']}")

def run_command(command):
    """Run a command and return output and status"""
    try:
        print_colored(f"Running: {command}", "blue")
        process = subprocess.run(command, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Command failed with error code {e.returncode}", "red")
        print_colored(f"Error output: {e.stderr}", "red")
        return False, e.stderr

def main():
    """Main function to set up and run the income prediction app"""
    print_colored("Income Prediction App Setup", "green")
    print_colored("=========================", "green")
    print()

    # Install required packages
    print_colored("Installing required packages...", "yellow")
    success, output = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if not success:
        print_colored("Failed to install packages. Continuing anyway...", "yellow")
    
    # Train the model
    print_colored("Training the income prediction model...", "yellow")
    success, output = run_command(f"{sys.executable} train_model.py")
    if not success:
        print_colored("Model training failed, but we'll continue with simulated predictions.", "yellow")
    else:
        print_colored("Model training complete!", "green")
    
    # Start the Flask app
    print_colored("Starting the Flask application...", "yellow")
    print_colored("Once started, open your browser and go to: http://127.0.0.1:5000", "blue")
    print_colored("Press Ctrl+C to stop the server when done.", "blue")
    print()
    
    # Use subprocess.Popen to run Flask app without waiting for completion
    try:
        flask_process = subprocess.Popen([sys.executable, "app.py"], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
        
        # Wait a bit for Flask to start
        time.sleep(2)
        
        # Check if process is still running
        if flask_process.poll() is None:
            print_colored("Flask app started successfully!", "green")
            print_colored("Server running at http://127.0.0.1:5000", "blue")
            
            # Keep running until user interrupts
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print_colored("\nShutting down the server...", "yellow")
                flask_process.terminate()
        else:
            # Process exited quickly, indicating an error
            stdout, stderr = flask_process.communicate()
            print_colored("Flask app failed to start properly.", "red")
            print_colored(f"Output: {stdout}", "red")
            print_colored(f"Error: {stderr}", "red")
    
    except Exception as e:
        print_colored(f"Error starting Flask app: {str(e)}", "red")
    
if __name__ == "__main__":
    main() 