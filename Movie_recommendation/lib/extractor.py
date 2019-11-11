import tensorflow as tf
import urllib
from PIL import Image
import numpy as np
import IPython
import matplotlib.pyplot as plt


from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np
import tensorflow as tf

class svd_extractor:
    def build_rating_matrix(self,df,userCol="userId",productCol="tmdbId",ratingCol="rating"):
        self.id_2_product = {i:j for i,j in enumerate(df[productCol].unique())}
        self.product_2_id = {j:i for i,j in self.id_2_product.items()}
        
        self.id_2_user = {i:j for i,j in enumerate(df[userCol].unique())}
        self.user_2_id = {j:i for i,j in self.id_2_user.items()}

        self.rating_matrix = np.zeros([len(self.id_2_user),len(self.id_2_product)])
        
        for row in df.itertuples():
            rawUser, rawProduct, rating = row[1], row[2], row[3]
            user, product = self.user_2_id[rawUser], self.product_2_id[rawProduct]

            self.rating_matrix[user][product] = rating
        return self.rating_matrix
    

    def svd_scipy(self,rating_matrix,num_features=10):
        u, s, vt = svds(rating_matrix, k=num_features) # k is the number of factors
        return u,s,vt.transpose()
    
        
    def svd_tf(self,rating_matrix):
        tf.reset_default_graph()
        nb_users, nb_products = rating_matrix.shape
        with tf.Session() as sess:
            rating_matrix_tf = tf.placeholder(tf.float32, shape=(nb_users, nb_products))
            S, U, V = tf.svd(rating_matrix_tf)

            s,u,v = sess.run([S,U,V],feed_dict={rating_matrix_tf:rating_matrix})
        return s,u,v     


class Img_extractor(object):
    def __init__(self, model='VGG16', *args, **kwargs):
        self.models = [
            'VGG16', 'VGG19', 'ResNet50',
            'DenseNet121', 'DeseNet169', 'inception_v3',
            'inceptionResNetV2'
        ]
        if model == "VGG16":
            self.model = tf.keras.applications.VGG16(*args, **kwargs)
        elif model == "VGG19":
            self.model = tf.keras.applications.VGG19(*args, **kwargs)
        elif model == "Resnet50":
            self.model = tf.keras.applications.ResNet50(*args, **kwargs)
        elif model == "DenseNet121":
            self.model = tf.keras.applications.DenseNet121(*args, **kwargs)
        elif model == "DeseNet169":
            self.model = tf.keras.applications.DeseNet169(*args, **kwargs)
        elif model == "inception_v3":
            self.model = tf.keras.applications.inception_v3(*args, **kwargs)
        elif model == "inceptionResNetV2":
            self.model = tf.keras.applications.inceptionResNetV2(*args, **kwargs)
        else:
            print("you must select one model from {}".format(self.models))

    def get_features(self, img_path, **kwargs):
        
        raw_arr = self.read_img(img_path) ### read image 

        img_arr = np.expand_dims(raw_arr, axis=0) ### generate x, dim of x should be (None,224,224,3)
        
        features = self.model.predict(img_arr)
        return features

    def read_img(self, img_path, *args, **kwargs):
        # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), *args, **kwargs)
        # raw_arr = tf.keras.preprocessing.image.img_to_array(img)        
        path = urllib.request.urlopen(img_path) if img_path.startswith("http") else img_path

        raw_img = Image.open(img_path)
        resized_img = raw_img.resize((224, 224))
        raw_arr = np.array(resized_img)

        return raw_arr

    def show_img(self, img_path, **kwargss):
        img_arr = self.read_img(img_path)
        plt.imshow(img_arr, aspect="auto")
        plt.show()