from flask import Flask, render_template, request
from utils.config import logger
from utils.tools import agent
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler

# Logging Handlers
logfile = "output.log"
handler_1 = FileCallbackHandler(logfile)
handler_2 = StdOutCallbackHandler()

# Initialize Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('prompt') 
        if not user_input:
            return render_template('index.html', error="Please enter a query.")

        logger.info(f"User Input: {user_input}")

        try:
            response = agent.invoke(user_input, {"callbacks": [FileCallbackHandler("output.log"), StdOutCallbackHandler()]})
        except Exception as e:
            response = f'An unexpected error {e} occurred. Please try later.'

        logger.info(f"Response: {response}")
        return render_template('index.html', user_prompt=user_input, generated_code=response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)