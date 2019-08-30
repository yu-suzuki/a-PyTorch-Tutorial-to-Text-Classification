import sys
from utils import create_input_files_fromdb, train_word2vec_model

if __name__ == '__main__':
    args = sys.argv
    create_input_files_fromdb(output_folder='./data'+args[1],
                              hostname=args[2],
                              database=args[3],
                       sentence_limit=15,
                       word_limit=20,
                       min_word_count=5)

    train_word2vec_model(data_folder='./data'+args[1],
                         algorithm='skipgram')
