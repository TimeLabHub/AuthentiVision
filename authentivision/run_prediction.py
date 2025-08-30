# Copyright 2024 TimeLabHub. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# run_prediction.py

import subprocess
import os

def run_authentivision_on(path_to_image_or_folder, model_path="best_model.pth"):
    """
    A simple Python wrapper to call the predict.py script.

    Args:
        path_to_image_or_folder (str): The path to the image or folder to analyze.
        model_path (str): Path to the trained model weights.
    """
    # Ensure the main prediction script exists
    prediction_script = "predict.py"
    if not os.path.exists(prediction_script):
        print(f"Error: The main prediction script '{prediction_script}' was not found.")
        return

    # Construct the command
    command = [
        "python",
        prediction_script,
        "--input-path",
        path_to_image_or_folder,
        "--model-path",
        model_path
    ]

    print(f"Running command: {' '.join(command)}")
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the prediction script: {e}")
    except FileNotFoundError:
        print("Error: 'python' command not found. Is Python installed and in your PATH?")


if __name__ == '__main__':
    # --- USAGE EXAMPLE ---
    # Replace this with the path to your image or folder
    target_path = "/path/to/your/image.jpg" # <--- CHANGE THIS
    # target_path = "/path/to/your/folder_of_images" # <--- OR THIS
    
    if target_path.startswith("/path/to/your"):
        print("Please update the 'target_path' variable in 'run_prediction.py' to point to your image or folder.")
    else:
        run_authentivision_on(target_path)
