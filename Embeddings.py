import numpy, spacy

def cosine_similarity(u: numpy.ndarray, v: numpy.ndarray) -> float:
    """Compute the cosine similarity between two vectors"""
    return (u @ v) / (numpy.linalg.norm(u) * numpy.linalg.norm(v))

def LookupEmbeddings(nlp):
    print(f"=== {LookupEmbeddings.__name__} ===")
    dog_embedding = nlp.vocab["dog"].vector
    cat_embedding = nlp.vocab["cat"].vector
    apple_embedding = nlp.vocab["apple"].vector
    tasty_embedding = nlp.vocab["tasty"].vector
    delicious_embedding = nlp.vocab["delicious"].vector
    truck_embedding = nlp.vocab["truck"].vector    
    print(f"dog embedding: shape: {dog_embedding.shape} {dog_embedding[:10]}")

    dog_cat_similarity = cosine_similarity(dog_embedding, cat_embedding)
    print(f"dog_cat_similarity: {dog_cat_similarity}")

    delicious_tasty_similarity = cosine_similarity(delicious_embedding, tasty_embedding)
    print(f"delicious_tasty_similarity: {delicious_tasty_similarity}")
    #assert delicious_tasty_similarity > dog_cat_similarity # unlike the dog and cat embeddings, delicious and tasty have similar word embeddings because you can use them interchangeably.
    
    apple_delicious_similarity = cosine_similarity(apple_embedding, delicious_embedding)
    print(f"apple_delicious_similarity: {apple_delicious_similarity}")

    apple_tasty_similarity = cosine_similarity(apple_embedding, tasty_embedding)
    print(f"apple_tasty_similarity: {apple_tasty_similarity}")

    apple_dog_similarity = cosine_similarity(apple_embedding, dog_embedding)
    print(f"apple_dog_similarity: {apple_dog_similarity}")
    assert apple_dog_similarity < dog_cat_similarity

    truck_delicious_similarity = cosine_similarity(truck_embedding, delicious_embedding)
    print(f"truck_delicious_similarity: {truck_delicious_similarity}")
    assert truck_delicious_similarity < apple_delicious_similarity
    assert truck_delicious_similarity < apple_tasty_similarity

    truck_tasty_similarity = cosine_similarity(truck_embedding, tasty_embedding)
    print(f"truck_tasty_similarity: {truck_tasty_similarity}")
    assert truck_tasty_similarity < apple_delicious_similarity
    assert truck_tasty_similarity < apple_tasty_similarity

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")
    LookupEmbeddings(nlp)