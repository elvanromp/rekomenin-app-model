# Rekomenin
The Rekomenin Model is the result of a model developed by our team on the Rekomenin web application

## Features
- Course Recommendations - The recommendation system uses a Collaborative Filtering. This system operates by predicting course ratings for users who haven't yet provided ratings, allowing system to recommend the most relevant courses to each user.
- Job Opening Recommendations - The recommendation system uses a Content-Based Filtering approach. This system operates by calculating similarities using Cosine Similarity to recommend the most relevant jobs based on course descriptions, or vice versa.
  
## Installation
To get started with this Next.js project, follow these steps:

1. **Clone the Repository**

   Clone this repository to your local machine using git:

   ```bash
   git clone https://github.com/elvanromp/rekomenin-app-model.git
   
2. **Navigate to the Project Directory**

   Move into the directory of the cloned project:
   ```bash
   cd rekomenin-app-model
   ```
   
3. **Create a Virtual Environment**

    Create a virtual environment to isolate the project's dependencies. You can use venv for this:
    
    ```bash
    python -m venv venv
    ```
    
4. **Activate the Virtual Environment**

   Activate the virtual environment. The command depends on your operating system:

    For Windows:
    
    ```bash
    venv\Scripts\activate
    ```
    For macOS and Linux:
    
    ```bash
    source venv/bin/activate
    ```
5. **Install Dependencies**

    Install all required dependencies using pip and the requirements.txt file:
    
    ```bash
    pip install -r requirements.txt
    ```

6. **Run the Application**

    Once the dependencies are installed, you can run the application (modify this step as necessary for your specific project):
    
    ```bash
    python main.py
    ```

## Contact
If you have any questions or feedback, please contact us at elvanro2015@gmail.com
