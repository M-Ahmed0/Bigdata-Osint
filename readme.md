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