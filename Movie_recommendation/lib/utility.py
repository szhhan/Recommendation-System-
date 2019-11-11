import tmdbsimple as tmdb

import urllib
import os
from IPython.display import display, HTML, Image



key_v4 = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxMGY0MGYwZDVkNzk0ZTRiYWNiMjY2MTg4MTI4YTg5NiIsInN1YiI6IjViZGE1NjNlMGUwYTI2MDNjYTAwM2Q1MCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.6yPX2IdoGMMDQ_yjXkj9CyIFG0c6c6qcOaxYn7hC_RQ'
key_v3 = '10f40f0d5d794e4bacb266188128a896'

tmdb_connector = tmdb
tmdb_connector.API_KEY = key_v3

# https://api.themoviedb.org/3/movie/2/images/7VqciAWfFkYrFK7XlQXVjej1Fup.jpg?api_key=10f40f0d5d794e4bacb266188128a896
# http://image.tmdb.org/t/p/w185//
"""
w92", "w154", "w185", "w342", "w500", "w780" is the size of image in the url
"""

class scraper:
    def __init__(self,ratingLink=None):
        tmdbBase = 'http://image.tmdb.org/t/p/w185/' 
        key_v3 = '10f40f0d5d794e4bacb266188128a896'
        
        tmdb_connector = tmdb
        tmdb_connector.API_KEY = key_v3
        self.tmdb = tmdb
        
        ratingLink = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip" if not ratingLink else ratingLink
        if "data" not in os.listdir():
            os.mkdir("data")
            os.mkdir("data/posters")
            print("Create new folder to save data")
        
    def download_rating(self):
        if "ml-latest-small.zip" not in os.listdir("data"):
            urllib.request.urlretrieve(self.ratingLink,"data/ml-latest-small.zip")
            os.system("unzip -a -n data/ml-latest-small.zip -d data/")
    
    def proces_rating(self):
        rating_df = pd.read_csv('data/ml-latest-small/ratings.csv')
        linkes_df = pd.read_csv('data/ml-latest-small/links.csv')
        df_merged = pd.merge(rating_df,linkes_df,on=['movieId'])
        df_merged.dropna(how="any",inplace=True)
        df_merged[['tmdbId']] = df_merged[['tmdbId']].astype(int)
        self.rating = df_merged
        return df_merged
    
    def download_posters(self,method="map",target_folder="data/poster/"):
        TMDBIds = self.rating.tmdbId.unique()
        
        if method == "map":
            list(map(lambda x:scrape_poster(id=x,target_folder=target_folder),TMDBIds))
        elif method == "multiprocess":
            workers = 6
            with ProcessPoolExecutor(max_workers=workers) as executor:
                links_multiprocess = executor.map(scrape_poster, TMDBIds)     
        elif method == 'loop':   
            for Id in TMDBIds:
                scrape_poster(id=Id)
        else:
            print("Must select a method")
            
    def get_poster_link(self,id):
        count = 1
        while count < 3:
            try:
                tmdbBase = 'http://image.tmdb.org/t/p/w185/' 
                movieInfo = self.tmdb.Movies(id).info()
                posterLink = movieInfo['poster_path']
                fullLink = tmdbBase + posterLink
                return fullLink
            except:
                count += 1
        return None
    def scrape_poster(self,id,target_folder="data/poster/"):
        posterLink = self.get_poster_link(id)
        if posterLink:
            urllib.request.urlretrieve(fullLink, f"{target_folder}{id}.jpg")
            print(f'Poster {id} successfully downloaded')
        else:
            print(f"Unable to scrape data for poster :{id}")


def cosine_matrix(data):
    '''
    Row based similirty
    
    default: user is in row, item in column
    ''' 
    dim = data.shape
    
    constant = tf.constant(1e-9,dtype=tf.float32)

    df = tf.placeholder(shape=[dim[0],dim[1]],dtype=tf.float32)

    similar_user = tf.matmul(df,tf.transpose(df)) + constant

    norm_user = tf.reshape(tf.sqrt(tf.diag_part(similar_user)),[-1,1])

    norm_user_matrix = tf.matmul(norm_user,tf.transpose(norm_user))

    similar_user = similar_user/norm_user_matrix


    with tf.Session() as sess:
        ans = sess.run(similar_user,feed_dict={df:data})

    return ans