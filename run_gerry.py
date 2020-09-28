import intentClassifier as ic
import flask

app = flask.Flask("Gerry")

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    incoming_data = flask.request.get_json(force=True)
    print(incoming_data)
    if flask.request.method == "POST":
        text = incoming_data['text']
        prediciton = ic.classify_intent(text)
        data["intent"] = prediciton

        data["success"] = True

    return flask.jsonify(data)

if __name__=="__main__":
    print("Starting Gerry Server")
    app.run()