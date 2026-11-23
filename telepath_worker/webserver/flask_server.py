from flask import Flask
from flask_bootstrap import Bootstrap
from flask import Blueprint, render_template, flash, redirect, url_for

import telepath

def create_app():
  app = Flask(__name__)
  Bootstrap(app)
  return app

app = create_app()


@app.route("/hello")
def home():
    return "Hello, World!"

@app.route("/tb")
def tb():
    return render_template('template1.html')


@app.route("/ps")
def ps():
    return render_template('cmd.html')

@app.route("/bs4base.css")
def bs4base():
    return render_template('bs4base.css')


@app.route("/")
def t4():
    return render_template('rawbs4_template.html',mapper={'1':'3'})

if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask


