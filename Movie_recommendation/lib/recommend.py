import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS,MatrixFactorizationModel, Rating
from pyspark.ml.evaluation import RegressionEvaluator
import math
from lib.extractor import svd_extractor
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML, Image

class SP_ALS:
    def __init__(self):
        pass
    def train(self,df,rank=10,numIterations=20):
        assert len(df.columns) == 3
        """Create SparkContext"""
        ### You can only have one sparkContext at the same time
        sc = pyspark.SparkContext.getOrCreate()
        sqlContext = SQLContext(sc)
        """Create Spark dataframe"""
        df_spark = sqlContext.createDataFrame(df)
        
        """Train ALS model """
        self.model = ALS.train(df_spark, rank, numIterations)
        self.rmse = self.computeRmse(df_spark)
        
    def computeRmse(self,df):
        """
        Compute RMSE (Root mean Squared Error).
        """
        true_rating = df[['rating']].rdd.map(lambda x:x[0]).collect()    

        rdd_x = df[['UserId',"tmdbId"]].rdd
        pred_rating = model.predictAll(rdd_x).map(lambda x:x[2]).collect()

        MSE = np.mean([(x-y)**2 for x,y, in zip(true_rating,pred_rating)])
        RMSE = math.sqrt(MSE)
        return RMSE    
    
    def recommend_to_user(self,userId =10,num_products=10):
        return self.model.recommendProducts(userId,num_products)
    def recommend_to_products(self,productId =10,num_users=10):
        return self.model.recommendUsers(productId,num_users)
    def predict_score(self,userId,productId):
        return self.model.predict(userId,productId)
    def get_product_features(self):
        productFeatures = self.model.productFeatures()
        return productFeatures.collect()


class cosine_recommender:
    def __init__(self):
        pass
    def construct_cosine_matrix(self,features):
        similarity_matrix = cosine_similarity(features)
        return similarity_matrix
    
    def recommend(self,similarity_matrix,ID,top_n=5):
        n_most_similar = similarity_matrix[ID,:].argsort()[::-1][:top_n+1]
        return n_most_similar
    
    def cosine_matrix_numpy(self,features):
        features = np.array(features)
        m = features.transpose()
        d = m.T @ m
        norm = (m * m).sum(0, keepdims=True) ** .5
        similarity_matrix_numpy = d / norm / norm.T
        return similarity_matrix_numpy
    
    def cosine_matrix_tf(self,features):
        '''
        Row based similirty

        default: user is in row, item in column
        ''' 
        features = np.array(features)

        dim = features.shape

        constant = tf.constant(1e-9,dtype=tf.float32)

        df = tf.placeholder(shape=[dim[0],dim[1]],dtype=tf.float32)

        similar_user = tf.matmul(df,tf.transpose(df)) + constant

        norm_user = tf.reshape(tf.sqrt(tf.diag_part(similar_user)),[-1,1])

        norm_user_matrix = tf.matmul(norm_user,tf.transpose(norm_user))

        similar_user = similar_user/norm_user_matrix


        with tf.Session() as sess:
            similarity_matrix_tf = sess.run(similar_user,feed_dict={df:features})

        return similarity_matrix_tf
    
class cnn_recommender(cosine_recommender):
    def __init__(self):
        super().__init__()
        
    def display_images(self,indices,poster_links):
        links_list = [poster_links[index] for index in indices]
        
        images = ''
        for i in links_list:
            images += f"<img style='width: 100px; margin: 0px; float: left; border: 1px solid black;' src='{i}' />" 
        display(HTML(images)) 
    