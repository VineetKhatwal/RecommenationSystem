import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Lambda, Activation, Reshape
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from reco import vis
import multiprocessing as mp
from multiprocessing import Pool, Process
import os
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"


class AmazonBooks():

    def __init__(self):
        # self.books_df = pd.read_csv('FileAmazon.tsv', sep='\t')
        self.books_df = pd.read_csv(
            'FileAmazon.tsv', sep='\t', error_bad_lines=False)
        print(self.books_df.head())
        # df[df.id.apply(lambda x: x.isnumeric())]

        self.books_df = self.books_df[['customer_id',
                                       'product_id', 'product_title', 'star_rating']]
        '''
        self.books = self.books.astype(
            {"customer_id": int, "product_id": int, "product_title": str, "star_rating": float})
        '''
        self.books_df[self.books_df.product_id.apply(lambda x: x.isnumeric())]

        # Splitting Explicit and Implicit user ratings
        self.explicitRatingsDF = self.books_df[self.books_df.star_rating != 0]
        self.implicitRatingsDF = self.books_df[self.books_df.star_rating == 0]

        # Each Books Mean ratings and Total Rating Count
        self.average_rating = pd.DataFrame(
            self.explicitRatingsDF.groupby('product_id')['star_rating'].mean())

        self.average_rating['ratingCount'] = pd.DataFrame(
            self.explicitRatingsDF.groupby('product_id')['star_rating'].count())
        self.average_rating = self.average_rating.rename(
            columns={'star_rating': 'MeanRating'})

        '''
        self.average_rating = pd.DataFrame(
            self.explicitRatingsDF.groupby('product_id')['star_rating'].mean())
        '''
        self.average_rating = self.average_rating[self.average_rating['ratingCount'] > 5]

        print(self.average_rating.ratingCount.mean())
        # df[df['temperature']>30]

        self.average_rating = self.average_rating.sort_values(by=[
                                                              'ratingCount'], ascending=False)

        print("---------  AVERAGE RATING ----------")
        print(self.average_rating)

        # To get a stronger similarities
        counts1 = self.explicitRatingsDF['customer_id'].value_counts()
        self.explicitRatingsDF = self.explicitRatingsDF[self.explicitRatingsDF['customer_id'].isin(
            counts1[counts1 >= 50].index)]

        # Explicit Books and ISBN
        self.explicit_product_id = self.explicitRatingsDF.product_id.unique()
        self.explicit_books = self.books_df.loc[self.books_df['product_id'].isin(
            self.explicit_product_id)]

        print("---------  EXPLICIT RATING ----------")
        print(self.explicit_product_id)

        # Look up dict for Book and BookID
        self.Book_lookup = dict(
            zip(self.explicit_books["product_id"], self.explicit_books["product_title"]))
        self.ID_lookup = dict(
            zip(self.explicit_books["product_title"], self.explicit_books["product_id"]))

    def Top_Books(self, n=10, RatingCount=100, MeanRating=2):

        BOOKS = self.books_df.merge(
            self.average_rating, how='right', on='product_id')
        BOOKS = BOOKS.drop_duplicates(['product_id'])

        # print("==========================        BOOKS         =======================")
        # print(BOOKS)

        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values(
            'MeanRating', ascending=False).head(n)

        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values(
            'ratingCount', ascending=False).head(n)

        '''
        print("===================================================================")
        print(M_Rating)
        print("===================================================================")
        print(H_Rating)
        '''
        return M_Rating, H_Rating

    def Explicit_MF_Bias(n_users, n_items, n_factors):

        # Item Layer
        item_input = Input(shape=[1], name='Item')
        item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(
            1e-5), name='ItemEmbedding')(item_input)
        item_vec = Flatten(name='FlattenItemE')(item_embedding)

    # Item Bias
        item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(
            1e-6), name='ItemBias')(item_input)
        item_bias_vec = Flatten(name='FlattenItemBiasE')(item_bias)

    # User Layer
        user_input = Input(shape=[1], name='User')
        user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(
            1e-6), name='UserEmbedding')(user_input)
        user_vec = Flatten(name='FlattenUserE')(user_embedding)

    # User Bias
        user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(
            1e-6), name='UserBias')(user_input)
        user_bias_vec = Flatten(name='FlattenUserBiasE')(user_bias)

    # Dot Product of Item and User & then Add Bias
        DotProduct = Dot(axes=1, name='DotProduct')([item_vec, user_vec])
        AddBias = Add(name="AddBias")(
            [DotProduct, item_bias_vec, user_bias_vec])

    # Scaling for each user
        y = Activation('sigmoid')(AddBias)
        rating_output = Lambda(
            lambda x: x * (max_rating - min_rating) + min_rating)(y)

    # Model Creation
        model = Model([user_input, item_input], rating_output)

    # Compile Model
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

        return model


class SVD(AmazonBooks):

    def __init__(self, n_latent_factor=20):
        super().__init__()

        self.n_latent_factor = n_latent_factor

        # nba["College"].fillna("No College", inplace = True)

        self.explicitRatingsDF.customer_id.fillna(0, inplace=True)
        self.explicitRatingsDF.product_id.fillna("NOT PRESENT", inplace=True)
        self.explicitRatingsDF.star_rating.fillna(0.0, inplace=True)

        print(
            "*******************        EXPLICIT RATINGS        ***************************")
        print(self.explicitRatingsDF)

        # ar = ar.drop_duplicates(['Received', 'Merch Ref'])

        self.explicitRatingsDF = self.explicitRatingsDF.drop_duplicates(
            ['customer_id', 'product_id'])

        self.ratings_mat = self.explicitRatingsDF.pivot(
            index="customer_id", columns="product_id", values="star_rating").fillna(0)

        print("********************      RATINGS  MAT  *****************************")
        print(self.ratings_mat)

        # self.ratings_mat = self.explicitRatingsDF.pivot(
        #   index="customer_id", columns="product_id", values="star_rating").fillna(0)
        self.uti_mat = self.ratings_mat.values

        # print("*************************************************************")
        # print(self.uti_mat)

        # normalize by each users mean
        self.user_ratings_mean = np.mean(self.uti_mat, axis=1)
        self.mat = self.uti_mat - self.user_ratings_mean.reshape(-1, 1)

        # print("*************************************************************")
        # print(self.mat)

        self.explicit_users = np.sort(
            self.explicitRatingsDF.customer_id.unique())
        self.User_lookup = dict(
            zip(range(1, len(self.explicit_users)), self.explicit_users))

        self.prediction_user = None

    def returnData(self):
        temp = self.explicitRatingsDF[['customer_id',
                                       'product_id', 'star_rating']]

        print(temp)

        return temp

    def parallelize(self):

        # df_split = np.array_split(df, n_cores)
        print("Parallelization Started")

        # singular value decomposition
        U, S, Vt = svds(self.mat, k=self.n_latent_factor)

        S_diag_matrix = np.diag(S)

        # Reconstructing Original Prediction Matrix
        XPredReconstructed = np.dot(np.dot(U, S_diag_matrix), Vt) + \
            self.user_ratings_mean.reshape(-1, 1)

        self.prediction_user = pd.DataFrame(
            XPredReconstructed, columns=self.ratings_mat.columns, index=self.ratings_mat.index)

        print("Parallelization Completed")

        print(self.prediction_user)

        return XPredReconstructed

    def SVDBreakDown(self):

        U, S, Vt = svds(self.mat, k=self.n_latent_factor)

        S_diag_matrix = np.diag(S)

        # Reconstructing Original Prediction Matrix
        XPredReconstructed = np.dot(np.dot(U, S_diag_matrix), Vt) + \
            self.user_ratings_mean.reshape(-1, 1)

        self.prediction_user = pd.DataFrame(
            XPredReconstructed, columns=self.ratings_mat.columns, index=self.ratings_mat.index)
        return

    def Recommend_Books(self, userID, num_RecommendationsForUser=10):

        # Get and sort the user's prediction_user
        user_row_number = self.User_lookup[userID]
        sortedUser_Prediction = self.prediction_user.loc[user_row_number].sort_values(
            ascending=False)

        # Get the user's data and merge in the books information.
        user_data = self.explicitRatingsDF[self.explicitRatingsDF.customer_id == (
            self.User_lookup[userID])]

        # print("************************************  User Data  *******************************************")
        # print(user_data)

        user_data.star_rating.fillna(0, inplace=True)

        user_full = (user_data.merge(self.books_df, how='left', left_on='product_id', right_on='product_id')  # .
                     # sort_values(['star_rating'])
                     )

        # print("************************************  User Full  *******************************************")
        # print(user_full)

        # Recommend the highest predicted rating books that the user hasn't seen yet.
        recommendation = (self.books_df[~self.books_df['product_id'].isin(user_full['product_id'])].
                          merge(pd.DataFrame(sortedUser_Prediction).reset_index(), how='left',
                                left_on='product_id',
                                right_on='product_id'))
        recommendation = recommendation.rename(
            columns={user_row_number: 'Predictions'})
        recommend = recommendation.sort_values(
            by=['Predictions'], ascending=False)
        RecommendationsForUser = recommend.iloc[:
                                                num_RecommendationsForUser, :-1]

        print("*************** Recommendations  ******************")
        print(RecommendationsForUser)
        return user_row_number, user_full, RecommendationsForUser


class KNN(AmazonBooks):

    def __init__(self, n_neighbors=10):
        super().__init__()
        self.n_neighbors = n_neighbors

        self.explicitRatingsDF = self.explicitRatingsDF.drop_duplicates(
            ['customer_id', 'product_id'])

        self.ratings_mat = self.explicitRatingsDF.pivot(
            index="product_id", columns="customer_id", values="star_rating").fillna(0)
        self.uti_mat = csr_matrix(self.ratings_mat.values)

        print(self.explicitRatingsDF)

        # KNN Model Fitting
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.uti_mat)

    def Recommend_Books(self, book, n_neighbors=10):

        # Book Title  to BookID
        # bID = list(self.Book_lookup.keys())[list(self.Book_lookup.values()).index(book)]
        bID = self.ID_lookup[book]

        query_index = self.ratings_mat.index.get_loc(bID)

        KN = self.ratings_mat.iloc[query_index, :].values.reshape(1, -1)

        dist, indexes = self.model_knn.kneighbors(
            KN, n_neighbors=n_neighbors + 1)

        Rec_books = list()
        Book_dis = list()

        for i in range(1, len(dist.flatten())):
            Rec_books.append(self.ratings_mat.index[indexes.flatten()[i]])
            Book_dis.append(dist.flatten()[i])

        Book = self.Book_lookup[bID]
        # print("BOOKS =", Book)
        # print("AMAZON BOOKS", self.books_df)
        Recommmended_Books = self.books_df[self.books_df['product_id'].isin(
            Rec_books)]
        Recommmended_Books = Recommmended_Books.drop_duplicates(
            ['product_title'])

        return Book, Recommmended_Books, Book_dis
