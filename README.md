# Document Chatbot

## System Requirements
- **Python Version**: Python 3.10.11

## How to Run the Python Application

### For Windows
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   .\venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python run.py
   ```

### For Linux
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python3 run.py
   ```

## How the Application Works
- Upload documents (.txt, .pdf, .docs, .csv, .xlsx) at the root URL (`/`).
- These documents are converted into a vectorDB where similarity searches are performed.
- Similar results are sent to OpenAI.
- You receive a streaming response from OpenAI, which can be viewed on the chatbot page (`/chatbot`).
- The application also handles embeddings, search rankings, and other advanced features.

## File Structure and Function Explanations

### Backend Directory
- `backend\app\api`: Contains all routes defined for APIs and Jinja templates.
- `backend\app\utils\logger.py`: Logger function for special debugging in OpenAI functions.
- `backend\app\utils\openai_call_functions.py`: Creates chains to handle retrieval and final streaming.
- `backend\app\utils\streaming_handler.py`: Helper functions for streaming responses from OpenAI.
- `backend\app\utils\vectorisation.py`: Handles document conversion to vectorDB (view, add, delete).