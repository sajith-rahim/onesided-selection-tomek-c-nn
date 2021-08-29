from utils import cosine_distance


class KNN:

    def __init__(self, X_train, y, k=1):
        self.X_train = X_train
        self.k = k
        self.y = y

    def find_neighbours(self, query_point):
        neighbors = {}
        nearest_neighbors = []
        self.y_index = {}

        for idx, point in enumerate(self.X_train):
            neighbors[point.tostring()] = cosine_distance(point, query_point)
            self.y_index[point.tostring()] = idx
            # Note: use list and append tuples to avoid hashing issue - .tostring

        # find k nearest neighbours
        for i in range(self.k):
            nearest_neighbor = min(neighbors, key=neighbors.get)
            #nearest_neighbors.append(np.fromstring(nearest_neighbor, dtype=float))
            nearest_neighbors.append(nearest_neighbor)
            # remove to avoid repetition
            neighbors.pop(nearest_neighbor)

        return nearest_neighbors

    def classify(self, query_point):

        neighbouring_class = {}  # <CLASS:COUNT>
        for neighbor in self.find_neighbours(query_point):
            neighbouring_class[self.y[self.y_index[neighbor]]] = neighbouring_class.get(self.y[self.y_index[neighbor]], 0) + 1
        # vote
        winner = max(neighbouring_class, key=neighbouring_class.get)
        return winner

    def averege(self, query_point):
        total = 0
        nearest_neighbors = self.find_neighbours(query_point)

        for neighbor in nearest_neighbors:
            total += float(self.y_index[neighbor])

        mean = total / len(nearest_neighbors)
        return mean




