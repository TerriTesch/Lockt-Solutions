from pymongo.mongo_client import MongoClient

# Your actual connection string with the password
uri = "mongodb+srv://LocktAdmin:L0cktForever@locktcluster.dfui7.mongodb.net/?retryWrites=true&w=majority&appName=LocktC>
# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Error:", e)
