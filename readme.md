# Project README

To properly set up and run this project, follow the instructions below:

## Installation

1. Install python 3.9

2. Clone the project repository to your local machine.

   ```bash
   git clone <repository-url>
   ```

3. Navigate to the cloned project directory.

   ```bash
   cd <project-directory>
   ```

4. Install `virtualenv` using `pip`.

   ```bash
   pip install virtualenv
   ```

5. Create a virtual environment named `venv`.

   ```bash
   python -m virtualenv venv
   ```

6. Activate the virtual environment.

   ```bash
   # For Windows
   venv\Scripts\Activate.ps1

   # For Linux/Mac
   source venv/bin/activate
   ```

7. Install the project dependencies from `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Set the `FLASK_APP` environment variable to point to the `webapp.py` file.

   ```bash
   # For Windows PowerShell
   $env:FLASK_APP = "webapp.py"

   # For Linux/Mac Terminal
   export FLASK_APP=webapp.py
   ```

2. Run the Flask application.

   ```bash
   flask run
   ```

The application should now be up and running. You can access it by visiting the provided URL in your web browser.

## Compatibility issues 

1. In case opencv-python is already installed and encountered a compatibility issue then please run the following commands:

   ```bash
   pip uninstall opencv-python
   pip uninstall opencv-contrib-python
   pip uninstall opencv-contrib-python-headless
   pip3 install opencv-contrib-python==4.5.5.62
   ```

## Import notes 

1. The initial uploaded images and videos get stored in the `uploads` folder. After they have been preprocessed and/or just handled, they will be deleted from there as they only serve as temporary store.

2. The processed images get stored in the `output` folder under `image.jpg`. These get properly decoded and displayed in the front-end without any issues. Furthermore, the file gets overwritten when a new image has been processed or the same one using different models.

3. The processed videos get saved in the back-end under the `output` folder as `preprocess_video` as we were not able to decode the video in the front-end. Thus, if you process the video in the front-end please check the output folder to see the results of the preprocessed video. Keep in mind, these always get overwriten when you use a new video or preprocessed the same one using different models.
