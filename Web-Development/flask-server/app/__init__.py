from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate

app = Flask(__name__)

# Set the configuration for the SQLAlchemy URI
app.config['SQLALCHEMY_DATABASE_URI'] = '<database-connection>'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True

db = SQLAlchemy(app)
migrate = Migrate(app, db)

CORS(app, resources={r"/*": {"origins": "*"}})
