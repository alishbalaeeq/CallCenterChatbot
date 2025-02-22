# 📞 Call Center Chatbot

An AI-powered **Call Center Chatbot** using **completely open-source models** that can run on **CPU and GPU**. This chatbot handles customer inquiries, retrieves company policies, and accesses customer data efficiently.

---

## **✨ Features**
✅ Uses **fully open-source models** like **Mistral-7B-Instruct (GGUF)**  
✅ Runs **locally** on **CPU or GPU**—no external APIs  
✅ Supports **customer data lookup** via SQL database  
✅ Fetches **company policies dynamically**  
✅ Built with **Flask** for easy web deployment  

---

## **🚀 Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/alishbalaeeq/CallCenterChatbot.git
cd CallCenterChatbot
```

### **2️⃣ Create a Virtual Environment**
```bash
conda create --name chatbot python=3.10 -y  # For Conda users
conda activate chatbot
# OR
python -m venv chatbot  # For venv users
source chatbot/bin/activate  # macOS/Linux
chatbot\Scripts\activate  # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set the Model Path as an Environment Variable**
Before running, define the **model path** (GGUF format):

#### **🔹 Linux/macOS**
```bash
export MODEL_PATH="/absolute/path/to/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
```

#### **🔹 Windows (PowerShell)**
```powershell
$env:MODEL_PATH="C:\absolute\path\to\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
```

---

## **🛠️ Running the Chatbot**
```bash
python app.py
```
The chatbot will be available at **http://127.0.0.1:5000/** in your browser.

---

## **🧠 Model & LLM Setup**
This project exclusively uses **open-source AI models** to ensure **privacy, transparency, and full local control**.  
It leverages:
- **Mistral-7B-Instruct (GGUF format) from TheBloke**
- Runs via **LlamaCpp**, enabling execution on **CPU or GPU**  
- No proprietary APIs—**everything runs locally**  

```python
from langchain_community.llms import LlamaCpp
import os

# Load model from environment variable
model_path = os.getenv("MODEL_PATH")

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    n_ctx=2048,
    n_gpu_layers=50,  # Set to 0 for CPU-only
    verbose=True
)
```

### **⚡ Want to Run on CPU?**
If you **don't have a GPU**, simply set:
```python
n_gpu_layers = 0  # Forces LlamaCpp to run entirely on CPU
```
This ensures that it runs smoothly even on **CPU-only** machines.

---

## **📂 Project Structure**
```
CallCenterChatbot/
│── files/                     # Stores database & company policies
│── static/                    # CSS and frontend assets
│── templates/                 # HTML templates
│── utils/                     # Modular utility scripts
│   ├── config.py              # Logging & global settings
│   ├── llm_setup.py           # LLM initialization (LlamaCpp)
│   ├── tools.py               # Tools for database & policy queries
│── app.py                     # Main Flask application
│── requirements.txt           # Dependencies list
│── README.md                  # You’re reading it now!
```

---

## **🛠️ Tools & Technologies**
- **Flask** → Web Framework  
- **LangChain** → AI Agent Orchestration  
- **LlamaCpp** → Efficient local model inference  
- **SQLite** → Customer data storage  
- **Hugging Face Transformers** → Model loading  

---

## **💡 Customization**
Want to use a different **open-source model**? Modify the **model path**:
```bash
export MODEL_PATH="/path/to/another-model.gguf"
```
Then restart the app!

---

## **🔒 Privacy & Security**
✅ **No external API calls**—All processing happens **locally**  
✅ **No data sharing**—Ensures **customer privacy**  
✅ **Fully Open-Source**—No vendor lock-in  

---

## **📬 Contact & Contributions**
We welcome **contributions**! Feel free to **fork**, submit **issues**, or suggest improvements. 🚀  

---
🔥 **Call Center Chatbot** → AI-driven customer support using **100% open-source models**! 🔥