import faiss
import numpy as np

class VectorDB:
    def __init__(self):
        pass

    def create_database(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.name = [] # this will store the identification names for the indexes

    def add(self, name, vector):
        # check if name already exists
        if name in self.name:
            return "Name already exists."
        vector = np.array([vector], dtype=np.float32)
        # check if vector already exists
        D, I = self.index.search(vector, 1)
        if D[0][0] == 0:
            return "Vector already exists."
        
        # check if vector is of the correct dimension
        if vector.shape[1] != self.index.d:
            return "Vector dimension does not match database dimension."
        
        self.index.add(vector)
        self.name.append(name)
        return True
    
    def search(self, query_vector, k):
        query_vector = np.array([query_vector], dtype=np.float32)
        D, I = self.index.search(query_vector, k)
        results = []
        for i in range(k):
            results.append({"name": self.name[I[0][i]], "distance": D[0][i]})
        return results
    
    def remove(self, name=None, vector=None):
        if name is not None:
            index = self.name.index(name)
            if index == -1:
                return False
            self.index.remove(index)
            self.name.remove(name)
            return True
        elif vector is not None:
            index = self.index.search(vector, 1)[1][0][0]
            if index == -1:
                return False
            self.index.remove(index)
            self.name.pop(index)
            return True
        else:
            raise ValueError("Either name or vector should be provided.")
    

    def save(self, path):
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
    
    def __len__(self):
        return self.index.ntotal