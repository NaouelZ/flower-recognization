import neural_network
from flask import Flask,render_template,request


app = Flask(__name__)
app.config["DEBUG"] = False


@app.route('/home',methods=['GET'])
def random():
    return render_template('index.html')


@app.route('/profile-upload-single',methods=['GET','POST'])
def profile():
    if request.method=='GET' : 
        return render_template('index.html')
    else : 
        phrase=neural_network.random_image()
        return render_template('index.html',result= phrase)