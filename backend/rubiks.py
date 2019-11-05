from flask import Flask, jsonify, request
from rubiks_cube_project_code import run_solver

app = Flask(__name__)

# @app.route('/')
# def welcome():
#     return "My name is jeff"

@app.route('/rubiks-pictures/', methods=['GET', 'POST'])
def save_pictures():
    __import__('pdb').set_trace()
    if request.method == 'POST':
        # read the data from request
        # dump it into a file .jpg
        print('POSTAGE')
        return rubiks()

    return 'Show login form'

def rubiks():
    solution = run_solver()
    solution = [str(move) for move in solution]
    return jsonify({ 'results': solution })
