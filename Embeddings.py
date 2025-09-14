import numpy, spacy
from tqdm import tqdm
class Embeddings():
    _nlp = None
    _path:str = None
    _words = None
    _word_to_vec_map = None
    _word_to_vec_map_unit_vectors: None
    _gender = None
    _bias_axis = None
    def __init__(self, path: str = None, word_to_vec_map = None):
        # $ pp spacy download en_core_web_md
        self._nlp = spacy.load("en_core_web_md")
        if word_to_vec_map:
            self._word_to_vec_map = word_to_vec_map
        elif path:
            self._path = path
            self._words = set()
            self._word_to_vec_map = {}
            self._read_glove_vecs()
            # The [paper](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf), which the debiasing algorithm is from, assumes all word vectors to have L2 norm as 1 and hence the need for the calculations below:
            self._word_to_vec_map_unit_vectors = {
                word: embedding / numpy.linalg.norm(embedding, ord=2)
                for word, embedding in tqdm(self._word_to_vec_map.items())
            }
            self._gender = (self._word_to_vec_map['female'] - self._word_to_vec_map['male'] + self._word_to_vec_map['woman'] - self._word_to_vec_map['man'] + self._word_to_vec_map['mother'] - self._word_to_vec_map['father'] + self._word_to_vec_map['girl'] - self._word_to_vec_map['boy'] + self._word_to_vec_map['gal'] - self._word_to_vec_map['guy']) / 5
            self._bias_axis = (self._word_to_vec_map_unit_vectors['female'] - self._word_to_vec_map_unit_vectors['male'] + self._word_to_vec_map_unit_vectors['woman'] - self._word_to_vec_map_unit_vectors['man'] + self._word_to_vec_map_unit_vectors['mother'] - self._word_to_vec_map_unit_vectors['father'] + self._word_to_vec_map_unit_vectors['girl'] - self._word_to_vec_map_unit_vectors['boy'] + self._word_to_vec_map_unit_vectors['gal'] - self._word_to_vec_map_unit_vectors['guy']) / 5
            print(f"_gender: {self._gender}, _bias_axis: {self._bias_axis} {numpy.sum(self._bias_axis)}")
        else:
            raise RuntimeError("Please provide a word_to vec map or path to load from!")

    def _cosine_similarity(self, u: numpy.ndarray, v: numpy.ndarray) -> float:
        """Compute the cosine similarity between two vectors"""
        # Special case. Consider the case u = [0, 0], v=[0, 0]
        if numpy.all(u == v):
            return 1
        norm_u = numpy.linalg.norm(u, ord=2)
        norm_v = numpy.linalg.norm(v, ord=2)
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
    
    def Bias(self, words):
        """
        See how the GloVe word embeddings relate to gender. You'll begin by computing a vector gender = e(woman) - e(man), where e(woman) represents the word vector corresponding to the word woman, and e(man) corresponds to the word vector corresponding to the word man.
        The resulting vector gender roughly encodes the concept of "gender".
        You might get a more accurate representation if you compute  g1=e(mother) - e(father), g2= e(girl) - e(boy) etc. and average over them, but just using e(woman) - e(man) will give good enough results.
        words inclined to female will have +ve cosine similarity while those inclined to male will have -ve consine similarity.
        """
        print(f"\n=== {self.Bias.__name__} ===")
        for w in words:
            if w in self._word_to_vec_map:
                # Positive values mean closer to woman; negative values mean closer to man
                print (f"{w}: {self._cosine_similarity(self._word_to_vec_map[w], self._gender)}, {self._cosine_similarity(self.NeutralizeGenderBias(w), self._bias_axis)}")
            else:
                print(f"Skipping {w} not in vocabulary")

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
        #print(f"=== {self.NeutralizeGenderBias.__name__} ===")
        if word in self._word_to_vec_map:
            # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
            e = self._word_to_vec_map[word]
            
            # Compute e_biascomponent using the formula given above. (≈ 1 line)
            e_biascomponent = ((e @ self._bias_axis) / numpy.square(numpy.linalg.norm(self._bias_axis, ord=2))) * self._bias_axis
        
            # Neutralize e by subtracting e_biascomponent from it 
            # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
            e_debiased = e - e_biascomponent
            return e_debiased
        else:
            print(f"Skipping {word} not in vocabulary")
        #print(f"cosine similarity between {w} and gender, before neutralizing: {self._cosine_similarity(self._word_to_vec_map[w], self._gender)}")
        #print(f"cosine similarity between {w} and bias_axis, after neutralizing: {self._cosine_similarity(e_debiased, self._bias_axis)}")

    def Equalize(self, pair):
        """
        Next, let's see how debiasing can also be applied to word pairs such as "actress" and "actor." Equalization is applied to pairs of words that you might want to have differ only through the gender property. 
        As a concrete example, suppose that "actress" is closer to "babysit" than "actor." By applying neutralization to "babysit," you can reduce the gender stereotype associated with babysitting. But this still does not guarantee that "actor" and "actress" are equidistant from "babysit." 
        The equalization algorithm takes care of this.
        The key idea behind equalization is to make sure that a particular pair of words are equidistant from the 49-dimensional g_orthogonal. The equalization step also ensures that the two equalized steps are now the same distance from e(receptionist_debiased), or from any other work that has been neutralized.

        Arguments:
        pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
        bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
        word_to_vec_map -- dictionary mapping words to their corresponding vectors
        
        Returns
        e_1 -- word vector corresponding to the first word
        e_2 -- word vector corresponding to the second word
        """
        print(f"\n=== {self.Equalize.__name__} ===")
        # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
        w1, w2 = pair[0], pair[1]
        e_w1, e_w2 = self._word_to_vec_map[w1], self._word_to_vec_map[w2]
        #print(f"e_w1: {e_w1}, e_w2: {e_w2}")
        # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
        mu = numpy.mean([e_w1, e_w2])
        #print(f"mu: {mu}, bias_axis: {bias_axis.shape}, mu * bias_axis: {mu * bias_axis}")

        # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
        mu_B = ((mu * self._bias_axis) / numpy.square(numpy.linalg.norm(self._bias_axis, ord=2))) * self._bias_axis
        mu_orth = mu - mu_B
        mu_orth_norm = numpy.square(numpy.linalg.norm(mu_orth, ord=2))
        #print(f"mu_orth: {mu_orth}, mu_orth_norm: {mu_orth_norm}") > 1 in the case of ["waiter", "waitress"] which fails numpy.sqr

        # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
        e_w1B = ((e_w1 * self._bias_axis) / numpy.square(numpy.linalg.norm(self._bias_axis, ord=2))) * self._bias_axis
        e_w2B = ((e_w2 * self._bias_axis) / numpy.square(numpy.linalg.norm(self._bias_axis, ord=2))) * self._bias_axis
        #print(f"e_w1B: {e_w1B}, e_w2B: {e_w2B}")
        
        # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
        tmp = numpy.sqrt(1 - numpy.square(numpy.linalg.norm(mu_orth, ord=2)))
        #print(f"tmp: {tmp}")
        corrected_e_w1B = tmp * ((e_w1B - mu_B)/numpy.linalg.norm(e_w1B - mu_B, ord=2))
        corrected_e_w2B = tmp * ((e_w2B - mu_B)/numpy.linalg.norm(e_w2B - mu_B, ord=2))
        #print(f"corrected_e_w1B: {corrected_e_w1B}, corrected_e_w2B: {corrected_e_w2B}")
        # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
        e1 = corrected_e_w1B + mu_orth
        e2 = corrected_e_w2B + mu_orth
        print("cosine similarities after equalizing:")
        print(f"cosine_similarity({w1}, gender) = {self._cosine_similarity(e1, self._bias_axis)}")
        print(f"cosine_similarity({w2}, gender) = {self._cosine_similarity(e2, self._bias_axis)}")
        #return e1, e2

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
    nlp.Bias(['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin'])
    nlp.Bias(['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer', 'scientist'])

def equalize():
    nlp = Embeddings('data/glove.6B.50d.txt')
    nlp.Equalize(("male", "female"))
    nlp.Equalize(("man", "woman"))
    nlp.Equalize(("father", "mother"))
    nlp.Equalize(("boy", "girl"))
    nlp.Equalize(("guy", "gal"))
    nlp.Equalize(("waiter", "waitress"))
    nlp.Equalize(("actor", "actress"))

if __name__ == "__main__":
    complete_analogy_tests()
    triads_analogy_tests()
    bias()
    equalize()