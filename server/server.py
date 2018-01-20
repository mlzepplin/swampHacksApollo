from flask import Flask, session, redirect, url_for, escape, request

app = Flask(__name__)

@app.route('/')
def index():
    print("req arg:", );
    return 'hello ' + request.args.get('username')

@app.route('/page')
def page():
    return '''
        <form action="/filesubmit">
             <input type="file">
             <input type="submit">
        </form>
    '''

@app.route('/filesubmit')
def filesubmit():
    print("--- hello world ---")
    files = request.files['file']
    print(files)
    return "cool thanks"