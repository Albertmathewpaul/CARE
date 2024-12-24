# CARE
Setup Instructions
Prerequisites
Python: Install Python (version 3.9 recommended) from python.org.
Git: Install Git from git-scm.com.
Basic Knowledge: Familiarity with Python virtual environments is helpful but not required.
Step-by-Step Setup
1. Clone the Repository
bash
Copy code
git clone https://github.com/Albertmathewpaul/CARE.git
cd CARE
2. Set Up a Virtual Environment
Create a virtual environment:
bash
Copy code
python -m venv myenv
Activate the environment:
Windows:
bash
Copy code
myenv\Scripts\activate
Mac/Linux:
bash
Copy code
source myenv/bin/activate
3. Install Required Libraries
Run the following command to install all necessary libraries:
bash
Copy code
pip install -r requirements.txt


4. Run the Streamlit Application
Make sure your virtual environment is activated.

Start the app by running:
bash
Copy code
streamlit run app.py

Open the link provided in the terminal (usually http://localhost:8501) in your web browser.

Notes for Colab Users (If Training is Needed)

Model training was done in Google Colab. You can replicate the training by uploading the model architecture and training script to Colab. 

Save the trained weights as classification_model.pth and place it in the project folder.

File Structure
bash
Copy code
CARE/
│
├── app.py                  # Main Streamlit app
├── classification_model.pth # Pretrained model file
├── requirements.txt        # Required libraries
└── README.md               # Project description
Troubleshooting
Access Denied: Run the command prompt as Administrator when starting the Streamlit app.
Virtual Environment Issues: Ensure the correct Python version is used, and your environment is activated.
Incorrect Results: Retrain the model in Colab if the classifications are not accurate.
