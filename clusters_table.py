from google.cloud import bigquery
import operator
import networkx as nx
from nltk.util import ngrams
from nltk import FreqDist
from gensim.models import Word2Vec
import os
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import KMeans
import csv 
import sys #used for passing in the argument
import pickle
from bs4 import BeautifulSoup
import pandas
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from google.cloud import bigquery

G = nx.Graph()
client = bigquery.Client.from_service_account_json('My_Project-c23185ac100b.json')

dataset_ref = client.dataset('Stackunderflow')
dataset = bigquery.Dataset(dataset_ref)
client.project


with open('clusters' + '.pkl', 'rb') as f:
    v1 = pickle.load(f)


def get_like_tags(tags):
    st = ["\'%"+ tag + "%\'" for tag in tags] 
    string = " (tags like "
    string += " or tags like ".join(st)
    string += ")"
    return string

def create_tag_query(string, ids):
    
    query = """
        SELECT
            id, tags, score, body
        FROM
            `bigquery-public-data.stackoverflow.posts_questions`
        where score > 0
        and 
        """
    query += string
    query += """
      and id in ( 
        (
        select post_id as id from `bigquery-public-data.stackoverflow.post_links`
        where related_post_id in 
        (
        SELECT
          id
        FROM
          `bigquery-public-data.stackoverflow.posts_questions`
        WHERE
        score > 0 and
          id IN (
          SELECT
            related_post_id
          FROM
            `bigquery-public-data.stackoverflow.post_links`
          GROUP BY
            related_post_id)
        )
    )
    UNION ALL

      (
        SELECT
          id
        FROM
          `bigquery-public-data.stackoverflow.posts_questions`
        WHERE
        score > 0 and
          id IN (
          SELECT
            related_post_id
          FROM
            `bigquery-public-data.stackoverflow.post_links`
          GROUP BY
            related_post_id)
        )
    )

        """
    query += "and id not in unnest(@a)"
    return query


def create_table(query, query_params, table):
    table_ref = dataset.table(table)

    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = query_params
    # Set the destination table to the table reference created above.
    job_config.destination = table_ref


    query_job = client.query(query, job_config=job_config)
    results = query_job.result()  # Waits for the query to finish
    return query_job, results


def clean(question):
    question = question.lower()
    question_text = BeautifulSoup(question, "lxml").get_text()
    tokens = tokenizer.tokenize(question_text)
    filtered_sentence = [w for w in tokens if not w in stop_words]
    
    filtered_stems = [stemmer.stem(plural) for plural in filtered_sentence]
    return filtered_stems

def create_cleaned_table(table):
    filename = str(table) + ".csv"
    table_ref = dataset_ref.table("cleaned_" + table)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True

    with open(filename, 'rb') as source_file:
        job = client.load_table_from_file(
            source_file,
            table_ref,
            location='US',  # Must match the destination dataset location.
            job_config=job_config)  # API request

    job.result()  # Waits for table load to complete.

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z][^\s]*\b')
collected_ids = []
for cluster in v1:
    try:
        print("started", cluster)
        query  = create_tag_query(get_like_tags(v1[cluster]), collected_ids)
        query_params = [
        bigquery.ArrayQueryParameter(
            'a', 'INT64', collected_ids)
        ]
        query_job, results = create_table(query, query_params, str(cluster))
        ids = [row.id for row in results]
        collected_ids.extend(ids)
        data_frame = query_job.to_dataframe()
        data_frame['stemmed_cleaned_words'] = data_frame.apply(lambda row: clean(row['body']), axis=1)
        filename = str(cluster) + ".csv"
        data_frame.to_csv(filename, header = False)
        create_cleaned_table(str(cluster))
        print("completed", cluster)
    except IOError as e:
        if e.errno == errno.EPIPE:
            print('pipe error')
        else:
            print(e.errno) 
            print(e.strerror)


