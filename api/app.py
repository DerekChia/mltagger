import os
import sys
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from serve import get_model_api

app = Flask(__name__)
CORS(app)

# Initialise model object
model_api = get_model_api(sys.argv[1])

@app.route('/api', methods=['POST'])
def api():
	# Data from the API POST request will be request.json
	input_data = request.json
	app.logger.info('api_input: ' + str(input_data))
	# Feeding input data to output
	output_data = model_api(input_data)
	app.logger.info('api_output: ' + str(output_data))
	response = jsonify(output_data)
	return response

@app.route('/')
def index():
	return 'It Works!'

@app.errorhandler(404)
def url_error(e):
	return """
	Wrong URL!
	<pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_erorr(e):
	return"""
	Internal error occured: <pre>{}</pre>
	See logs for full stacktrace.
	""".format(e), 500

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)