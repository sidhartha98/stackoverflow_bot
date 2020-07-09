#!/usr/bin/env python3

from flask import Flask, request, render_template
import numpy as np
import pickle as p
import json
from utils import *
from dialogue_manager import DialogueManager

app = Flask(__name__)


@app.route('/')
def home():
	return render_template("web.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    dialogue_manager = DialogueManager(RESOURCE_PATH)
    return dialogue_manager.generate_answer(userText)


if __name__ == '__main__':
    app.run(debug=True)