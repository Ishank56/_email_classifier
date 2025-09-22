#Email Reply Classifier: ML Pipeline & FastAPI
This project contains a complete machine learning pipeline to train an email reply classifier and a FastAPI application to serve the model through a simple web interface.

The project is divided into two main parts:

Part A (reply_classfier.py): Trains a baseline Logistic Regression model and fine-tunes a DistilBERT transformer model on the provided dataset. It then evaluates both and saves the best-performing model.

Part B (app.py): Loads the saved model and deploys it as a web service with a user-friendly interface to classify new email replies in real-time.

Getting Started: A Step-by-Step Guide
Follow these instructions to set up and run the entire project on your local machine.

Step 1: Clone the Repository
First, get the project files on your computer. If you haven't already, clone the repository:

git clone [https://github.com/Ishank56/_email_classifier.git](https://github.com/Ishank56/_email_classifier.git)
cd _email_classifier

Step 2: Set Up a Virtual Environment (Recommended)
This creates an isolated environment for the project's dependencies.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

You will see (venv) appear at the beginning of your terminal prompt.

Step 3: Install All Dependencies
The requirements.txt file contains all the necessary Python libraries. Install them with this single command:

pip install -r requirements.txt

Step 4: Train the Model (Part A)
Now, run the machine learning pipeline script. This will train both the baseline and the transformer models, compare their performance, and automatically save the best model to a new folder named reply_classifier_model/.

python reply_classfier.py

Let this script run to completion. You will see the evaluation scores for both models printed in the terminal.
<img width="1919" height="1012" alt="Screenshot 2025-09-22 223347" src="https://github.com/user-attachments/assets/6e11f583-f421-442c-97a9-11c62e6ae6d1" />


Step 5: Run the FastAPI Application (Part B)
Once the model is trained and saved, you can start the web server. This command uses uvicorn to run the application defined in app.py.

uvicorn app:app --reload
<img width="1919" height="1079" alt="Screenshot 2025-09-22 223404" src="https://github.com/user-attachments/assets/347b89fd-4686-48a2-beb9-0f09388210c8" />



The terminal will show a message indicating that the server is running, usually on http://127.0.0.1:8000.

INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)

Step 6: Use the Web Interface
This is the final step! Your model is now live on your local machine.

Open your favorite web browser (like Chrome, Firefox, or Edge).

In the address bar, navigate to:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

You will see the Email Reply Classifier interface. Type any sentence into the text box and click "Classify" to get a live prediction from your model!

You have now successfully set up, trained, and deployed the entire project.
