install:
    pip install -r requirements.txt

run:
    FLASK_APP=app.py FLASK_ENV=development ./env/bin/flask run --port 3000
