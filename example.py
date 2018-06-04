import sys
sys.path.append(".")
import gensim
import numpy as np
import factor_rotation as fr


def load_word2vec_model():
    print('loading the model...')
    model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_FILE, binary=False)
    print('pre-trained word2vec model loaded...')
    return model


def export_numpy_array(filename, np_array):
    print('exporting the array (' + filename + ') ...')
    np.save(SAVE_PATH + filename, np_array)


def unrotated_reps(model):
    sampled_reps = []; sampled_words = []
    for word in list(model.vocab.keys()):
        if type(word) is not str:
            word = str(word, encoding='utf-8')
        sampled_reps.append(model[word])
        sampled_words.append(word)
    sampled_reps = np.array(sampled_reps)
    sampled_words = np.array(sampled_words)
    print('converting word2vec embeddings to np_array')
    return sampled_reps, sampled_words


def varimax(unrotated):
    print('starting varimax rotation...')
    return fr.rotate_factors(unrotated, 'varimax_CF')


def parsimony(unrotated):
    print('starting factor parsimony rotation...')
    return fr.rotate_factors(unrotated, 'parsimony')


def parsimax(unrotated):
    print('starting factor parsimax rotation...')
    return fr.rotate_factors(unrotated, 'parsimax')


def quartimax(unrotated):
    print('starting factor quartimax rotation...')
    return fr.rotate_factors(unrotated, 'quartimax_CF')


def main():
    model = load_word2vec_model()
    unrotated, word_list = unrotated_reps(model)

    del model
    del word_list

    print('rescaling...')
    scale = .01
    unrotated *= scale
    print(np.max(unrotated))
    print(np.min(unrotated))

    print('start rotating...')
    mat_L, mat_T, *_ = method_dic[method_name](unrotated)
    export_numpy_array('{}_axis.npy'.format(method_name), mat_T)
    export_numpy_array('{}_rotated.npy'.format(method_name), mat_L)


if __name__ == '__main__':
    MODEL_FILE = './text8_word2vec_50_5_100.csv'
    SAVE_PATH = './'
    method_dic = {"varimax": varimax, "quartimax": quartimax, "parsimony": parsimony, "parsimax":parsimax}
    method_name = "parsimax"

    main()
