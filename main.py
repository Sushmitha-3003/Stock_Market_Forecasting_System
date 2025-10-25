import subprocess
import sys
import os

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")

def run_script(script_name, subdirectory):
    """
    Runs a single Python script from a specified subdirectory within the src folder.
    Args:
        script_name (str): The name of the script to run (e.g., 'stock_collector.py').
        subdirectory (str): The name of the subdirectory where the script is located.
    """
    print(f"--- Running {script_name} from {subdirectory} ---")
    try:
        script_path = os.path.join(SRC_DIR, subdirectory, script_name)
        
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"--- {script_name} completed successfully. ---\n")
    except FileNotFoundError:
        print(f"Error: The script '{script_name}' was not found at '{script_path}'. Please check your file paths.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: {script_name} failed with exit code {e.returncode}.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running {script_name}: {e}")
        sys.exit(1)

def main():
    """
    Main function to run all the project's scripts in a predefined order.
    """
    # Step 1: Collect stock and news data from 'data_collection'
    run_script("stock_collector.py", "data_collection")
    run_script("news_collector.py", "data_collection")
    
    # Step 2: Enhance the collected data with technical indicators from 'data_processing'
    run_script("feature_engineer.py", "data_processing")
    
    # Step 3: Preprocess the collected data from 'data_processing'
    run_script("preprocessor.py", "data_processing")
    
    # Step 4: Train the LSTM models from 'training'
    run_script("model_trainer.py", "training")
    
    # Step 5: Launch the Streamlit application from the 'src' directory
    print("--- Launching Streamlit App ---")
    try:
        app_path = os.path.join(SRC_DIR, "app.py")
        # Ensure the current working directory is the parent of src
        os.chdir(os.path.dirname(SRC_DIR))
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except FileNotFoundError:
        print("Error: Streamlit not found. Please ensure it is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while launching the Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
