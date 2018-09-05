import collections
import re
import sys
sys.path.append('../')

from mltagger.src.model import MLTModel
from mltagger.src.evaluator import MLTEvaluator

try:
    import ConfigParser as configparser
except:
    import configparser

def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary..
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config

def is_float(value):
    """
    Check in value is of type float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_input_data(input_data):
	"""
	Read in input data and return sentences that are tokenised.
	Example:
		sentences = [
		[['Not'], ['only'], ['as'], ['a'], ['hobby'], ['.']], # First sentence
		[['They'], ['use'], ['computers'], ['for'], ['their'], ['works'], ['.']] # Second sentence
		]
	"""
	sentences = []
	sentences_tokenised = []
	if len(input_data) > 0:
		input_data.strip()
		if list(input_data)[-1] != ".":
			input_data += '.'
		sentences = input_data.split(".")[:-1]
		for sentence in sentences:
			sentence = re.findall(r"[\w']+|[.,!?;]", sentence)

			sentence_new = []
			for token in sentence:
				token = [token]
				sentence_new.append(token)
			sentence_new.append(['.'])
			sentences_tokenised.append(sentence_new)
	print(sentences_tokenised)
	return sentences_tokenised

def get_model_api(model_path):
	# Initialise model and its configuration
	model = MLTModel.load(model_path)
	evaluator = MLTEvaluator(model.config)

	def model_api(input_data):
		"""
		Args:
			input_data: submitted to the API, raw string (JSON/application/json)
		Returns:
			output_data: return output data after running through inference
		"""
		batch_size = 32
		output_data_all = []

		data = read_input_data(input_data)

		for i in range(0, len(data), batch_size):
			batch = data[i:i+batch_size]
			cost, sentence_scores, token_scores_list = model.process_batch(batch, False, 0.0)

			_id = 0
			for j in range(len(batch)):
				for k in range(len(batch[j])):
					_id += 1
					output_data = {'id': str( _id ), 'result' : {'token' : str(batch[j][k][0]), 'score_token': str(token_scores_list[0][j][k]), 'score_sentence': str(sentence_scores[j])} }
					# print('token', batch[j][k][0])
					# print('token score', str(token_scores_list[0][j][k]))
					# print('sentence score', str(sentence_scores[j]))
					output_data_all.append(output_data)
			evaluator.append_data(cost, batch, sentence_scores, token_scores_list)

		data = {"input": input_data, 'output': output_data_all}

		return data
	return model_api