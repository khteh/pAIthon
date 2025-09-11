import numpy, spacy
from tqdm import tqdm
class Embeddings():
    _nlp = None
    _path:str = None
    _words = None
    _word_to_vec_map = None
    _word_to_vec_map_unit_vectors: None
    def __init__(self, path: str = None, word_to_vec_map = None, build_unit_vectors:bool = False):
        # $ pp spacy download en_core_web_md
        self._nlp = spacy.load("en_core_web_md")
        if word_to_vec_map:
            self._word_to_vec_map = word_to_vec_map
        elif path:
            self._path = path
            self._words = set()
            self._word_to_vec_map = {}
            self._read_glove_vecs()
        else:
            raise RuntimeError("Please provide a word_to vec map or path to load from!")
        if build_unit_vectors:
            # The paper assumes all word vectors to have L2 norm as 1 and hence the need for this calculation
            self._word_to_vec_map_unit_vectors = {
                word: embedding / numpy.linalg.norm(embedding)
                for word, embedding in tqdm(self._word_to_vec_map.items())
            }

    def _cosine_similarity(self, u: numpy.ndarray, v: numpy.ndarray) -> float:
        """Compute the cosine similarity between two vectors"""
        norm_u = numpy.linalg.norm(u)
        norm_v = numpy.linalg.norm(v)
        # Avoid division by 0
        return 0 if numpy.isclose(norm_u * norm_v, 0, atol=1e-32) else (u @ v) / (norm_u * norm_v)

    def Word2VecMap(self):
        return self._word_to_vec_map
    
    def LookupEmbeddings(self, nlp):
        print(f"=== {self.LookupEmbeddings.__name__} ===")
        dog_embedding = nlp.vocab["dog"].vector
        cat_embedding = nlp.vocab["cat"].vector
        apple_embedding = nlp.vocab["apple"].vector
        tasty_embedding = nlp.vocab["tasty"].vector
        delicious_embedding = nlp.vocab["delicious"].vector
        truck_embedding = nlp.vocab["truck"].vector    
        print(f"dog embedding: shape: {dog_embedding.shape} {dog_embedding[:10]}")

        dog_cat_similarity = self._cosine_similarity(dog_embedding, cat_embedding)
        print(f"dog - cat similarity: {dog_cat_similarity}")

        delicious_tasty_similarity = self._cosine_similarity(delicious_embedding, tasty_embedding)
        print(f"delicious - tasty similarity: {delicious_tasty_similarity}")
        #assert delicious_tasty_similarity > dog_cat_similarity # unlike the dog and cat embeddings, delicious and tasty have similar word embeddings because you can use them interchangeably.
        
        apple_delicious_similarity = self._cosine_similarity(apple_embedding, delicious_embedding)
        print(f"apple - delicious similarity: {apple_delicious_similarity}")

        apple_tasty_similarity = self._cosine_similarity(apple_embedding, tasty_embedding)
        print(f"apple - tasty similarity: {apple_tasty_similarity}")

        apple_dog_similarity = self._cosine_similarity(apple_embedding, dog_embedding)
        print(f"apple - dog similarity: {apple_dog_similarity}")
        assert apple_dog_similarity < dog_cat_similarity

        truck_delicious_similarity = self._cosine_similarity(truck_embedding, delicious_embedding)
        print(f"truck - delicious similarity: {truck_delicious_similarity}")
        assert truck_delicious_similarity < apple_delicious_similarity
        assert truck_delicious_similarity < apple_tasty_similarity

        truck_tasty_similarity = self._cosine_similarity(truck_embedding, tasty_embedding)
        print(f"truck - tasty similarity: {truck_tasty_similarity}")
        assert truck_tasty_similarity < apple_delicious_similarity
        assert truck_tasty_similarity < apple_tasty_similarity

    def complete_analogy(self, word_a, word_b, word_c, word_to_vec_map):
        """
        In the word analogy task, complete this sentence:
        "a is to b as c is to ____".

        An example is:
        'man is to woman as king is to queen' .

        You're trying to find a word d, such that the associated word vectors  𝑒𝑎,𝑒𝑏,𝑒𝑐,𝑒𝑑
        are related in the following manner:
        𝑒𝑏−𝑒𝑎≈𝑒𝑑−𝑒𝑐

        Measure the similarity between  𝑒𝑏−𝑒𝑎
        and  𝑒𝑑−𝑒𝑐
        using cosine similarity.

        Performs the word analogy task as explained above: a is to b as c is to ____. 
        
        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
        
        Returns:
        best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
        """
        print(f"=== {self.complete_analogy.__name__} ===")
        # convert words to lowercase
        word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
        
        ### START CODE HERE ###
        # Get the word embeddings e_a, e_b and e_c (≈1-3 lines)
        e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
        ### END CODE HERE ###
        
        words = word_to_vec_map.keys()
        max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
        best_word = None                   # Initialize best_word with None, it will help keep track of the word to output
        
        # loop over the whole word vector set
        for w in words:   
            # to avoid best_word being one the input words, skip the input word_c
            # skip word_c from query
            if w == word_c:
                continue
            
            # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
            cosine_sim = self._cosine_similarity((e_b - e_a), (word_to_vec_map[w] - e_c))
            
            # If the cosine_sim is more than the max_cosine_sim seen so far,
                # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w
        return best_word
    
    def Bias(self):
        """
        See how the GloVe word embeddings relate to gender. You'll begin by computing a vector 𝑔=𝑒𝑤𝑜𝑚𝑎𝑛−𝑒𝑚𝑎𝑛, where 𝑒𝑤𝑜𝑚𝑎𝑛 represents the word vector corresponding to the word woman, and 𝑒𝑚𝑎𝑛 corresponds to the word vector corresponding to the word man. 
        The resulting vector 𝑔 roughly encodes the concept of "gender".
        You might get a more accurate representation if you compute  𝑔1=𝑒𝑚𝑜𝑡ℎ𝑒𝑟−𝑒𝑓𝑎𝑡ℎ𝑒𝑟,  𝑔2=𝑒𝑔𝑖𝑟𝑙−𝑒𝑏𝑜𝑦, etc. and average over them, but just using  𝑒𝑤𝑜𝑚𝑎𝑛−𝑒𝑚𝑎𝑛 will give good enough results for now.        
        """
        print(f"=== {self.Bias.__name__} ===")
        g = self._word_to_vec_map['woman'] - self._word_to_vec_map['man']
        print(f"e(woman) - e(man) = {g}")
        print ('\nList of names and their similarities with constructed vector g:')
        # girls and boys name
        name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
        for w in name_list:
            # Positive values mean closer to woman; negative values mean closer to man
            print (w, self._cosine_similarity(self._word_to_vec_map[w], g))
        print('\nOther words and their similarities with constructed vector g:')
        word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
                    'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
        for w in word_list:
            # Positive values mean closer to woman; negative values mean closer to man
            print (w, self._cosine_similarity(self._word_to_vec_map[w], g))

    def NeutralizeGenderBias(self, word):
        """
        Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
        This function ensures that gender neutral words are zero in the gender subspace.
        𝑒𝑏𝑖𝑎𝑠_𝑐𝑜𝑚𝑝𝑜𝑛𝑒𝑛𝑡=𝑒⋅𝑔||𝑔||22∗𝑔(2)
        𝑒𝑑𝑒𝑏𝑖𝑎𝑠𝑒𝑑=𝑒−𝑒𝑏𝑖𝑎𝑠_𝑐𝑜𝑚𝑝𝑜𝑛𝑒𝑛𝑡(3)

        Arguments:
            word -- string indicating the word to debias
            g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
            word_to_vec_map -- dictionary mapping words to their corresponding vectors.
        
        Returns:
            e_debiased -- neutralized word vector representation of the input "word"        
        """
        print(f"=== {self.NeutralizeGenderBias.__name__} ===")
        g = self._word_to_vec_map['woman'] - self._word_to_vec_map['man']
        g_unit = self._word_to_vec_map_unit_vectors['woman'] - self._word_to_vec_map_unit_vectors['man']
        # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
        e = self._word_to_vec_map[word]
        
        # Compute e_biascomponent using the formula given above. (≈ 1 line)
        e_biascomponent = ((e @ g_unit) / numpy.linalg.norm(g_unit, ord=2)) * g_unit
    
        # Neutralize e by subtracting e_biascomponent from it 
        # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
        e_debiased = e - e_biascomponent
        print("cosine similarity between " + word + " and g, before neutralizing: ", self._cosine_similarity(self._word_to_vec_map[word], g))
        print("cosine similarity between " + word + " and g_unit, after neutralizing: ", self._cosine_similarity(e_debiased, g_unit))
    
    def _read_glove_vecs(self):
        with open(self._path, 'r') as f:
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                self._words.add(curr_word)
                self._word_to_vec_map[curr_word] = numpy.array(line[1:], dtype=numpy.float64)

def complete_analogy_tests():
    a = [3, 3] # Center at a
    a_nw = [2, 4] # North-West oriented vector from a
    a_s = [3, 2] # South oriented vector from a
    
    c = [-2, 1] # Center at c
    # Create a controlled word to vec map
    word_to_vec_map = {'a': a,
                       'synonym_of_a': a,
                       'a_nw': a_nw, 
                       'a_s': a_s, 
                       'c': c, 
                       'c_n': [-2, 2], # N
                       'c_ne': [-1, 2], # NE
                       'c_e': [-1, 1], # E
                       'c_se': [-1, 0], # SE
                       'c_s': [-2, 0], # S
                       'c_sw': [-3, 0], # SW
                       'c_w': [-3, 1], # W
                       'c_nw': [-3, 2] # NW
                      }
    nlp = Embeddings(None, word_to_vec_map)
    
    # Convert lists to numpy.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = numpy.array(word_to_vec_map[key])
            
    assert(nlp.complete_analogy('a', 'a_nw', 'c', word_to_vec_map) == 'c_nw')
    assert(nlp.complete_analogy('a', 'a_s', 'c', word_to_vec_map) == 'c_s')
    assert(nlp.complete_analogy('a', 'synonym_of_a', 'c', word_to_vec_map) != 'c'), "Best word cannot be input query"
    assert(nlp.complete_analogy('a', 'c', 'a', word_to_vec_map) == 'c')
    print("\033[92mAll tests passed")

def triads_analogy_tests():
    nlp = Embeddings('data/glove.6B.50d.txt')
    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print ('{} -> {} :: {} -> {}'.format( *triad, nlp.complete_analogy(*triad, nlp.Word2VecMap())))

def bias():
    nlp = Embeddings('data/glove.6B.50d.txt')
    nlp.Bias()

def neutralize_gender_bias_tests():
    nlp = Embeddings('data/glove.6B.50d.txt', None, True)
    nlp.NeutralizeGenderBias("receptionist")

if __name__ == "__main__":
    complete_analogy_tests()
    triads_analogy_tests()
    bias()
    neutralize_gender_bias_tests()