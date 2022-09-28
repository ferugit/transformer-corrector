import os
import json

class Reporter:
    def __init__(self, filename):
        self.output_file = filename
        self.data = {}

    def __del__(self): 
        with open(self.output_file , 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def report(self, key, data):
        self.data[key] = data

    def load(self, filepath):
        self.data = json.load(open(filepath, 'r'))