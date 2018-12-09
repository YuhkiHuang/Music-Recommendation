import numpy as np
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import Row, SparkSession, SQLContext
from pyspark import SparkContext
import datetime

starttime = datetime.datetime.now()

sc = SparkContext()
#sqlContext = SQLContext(sc)
rawData = sc.textFile('/Users/apple/Downloads/milliondata.txt')
dataset = rawData.map(lambda line: line.split('\t')).map(lambda element: Rating(int(element[0]), int(element[1]), float(element[2])))
print(type(dataset))

rawSongData = sc.textFile('/Users/apple/Downloads/song_data.csv')  # '/Users/apple/Downloads/fc_2.csv'
songData = rawSongData.map(lambda song: song.split(',')).map(lambda element: (int(element[0]), element[1]))

trainingData, testData = dataset.randomSplit([0.8, 0.2])
#trainingData, validationData, testData = dataset.randomSplit([0.6, 0.2, 0.2])

#tModel = ALS.trainImplicit(trainingData, rank=10, seed=5)
tModel = ALS.train(trainingData,30,20,0.04)

testdata = testData.map(lambda t: (t[0], t[1]))
predictions = tModel.predictAll(testdata).map(lambda t: ((t[0], t[1]), t[2]))
ratesAndPreds = testData.map(lambda t: ((t[0], t[1]), t[2])).join(predictions)
MSE = ratesAndPreds.map(lambda s: (s[1][0] - s[1][1])**2).mean()  
print("Root Mean Squared Error = " + str(np.sqrt(MSE)/10))
print("ratesAndPreds",ratesAndPreds.take(50))

# Use all data train a new model to recommend
model = ALS.train(dataset,30,20,0.04) #(dataset,50,10,0.01)
#model = ALS.trainImplicit(dataset, rank=10, seed=5)
print('model',model)
userFeatures = model.userFeatures()
print('userFeatures',userFeatures.take(2))
productFeatures = model.productFeatures()
print('productFeatures',productFeatures.take(2))
#print(model.userFeatures().count())     #
#print(model.productFeatures().count())  #

# The predict function takes a user id and a product id and produces a single floating point value
# Return a single floating point value
predictOne = model.predict(123,456)
print('predictOne',predictOne)

# Given a product id and a number of users (N), recommendUsers will find the top N users for the product id
# Returns a List of Ratings in Descending Order by Rating
product_id = 242
id242RecoUsers20 = model.recommendUsers(product_id, 20)
print(type(id242RecoUsers20))
#print('id242RecoUsers20',id242RecoUsers20[0:5])
print("The user interested in song %i : " % product_id)
j = 0
for user in id242RecoUsers20:
    print("User ID "+":"+str(id242RecoUsers20[j].user)+","+str(id242RecoUsers20[j].rating))
    j += 1

# Save model
#path = '/Users/apple/PycharmProjects/main/venv'
#model.save(sc, path)
#sameModel = LDAModel.load(sc, path)

# Generate the top recommendations for one user
# Returns a List of Ratings
user_id = 234
user234RecoItems20 = model.recommendProducts(user_id, 20)

songMap = songData.collectAsMap()
i = 0
print("The songs recommended for user %i : " % user_id)
for song in user234RecoItems20:
    print("Song "+str(i)+": "+songMap.get(song[1])+','+str(user234RecoItems20[i].rating))
    i += 1

# Generating recommendations in batch
TopProductsAllUsers = model.recommendProductsForUsers(10)
print(type(TopProductsAllUsers))
print('TopProductsAllUsers')
TopProductsAllUsers.first()

# Give a required number of recommendations (N) and the function will produce N users for every product in the training data
TopUsersAllProducts = model.recommendUsersForProducts(10)
print(type(TopUsersAllProducts))
print('TopUsersAllProducts')
TopUsersAllProducts.first()

endtime = datetime.datetime.now()

print ("Run time : ",endtime - starttime)#.seconds
