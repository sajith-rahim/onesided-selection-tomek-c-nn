import random

from utils import cosine_distance
import numpy as np


class IterativeCondensedNN:

    def __init__(self):
        self.y_index = {}

    def fit(self, X_train,y):

        for idx, observation in enumerate(X_train):
            self.y_index[observation.tostring()] = idx

        samples = []
        _random = random.randint(0, len(X_train) - 1)
        samples.append(X_train[_random])

        X_train = np.delete(X_train, _random, axis=0)

        n_samples = len(samples)



        while True:
            # set initial distance high to always improve at the beginning
            minSampleDistance = 99999
            closestSample = None
            closestClass = None
            for idx, observation in enumerate(X_train):
                # self.y_index[observation.tostring()] = idx
                for sample in samples:
                    print(observation.shape, sample.shape)
                    sampleDistance = cosine_distance(observation, sample)
                    if sampleDistance < minSampleDistance:
                        minSampleDistance = sampleDistance
                        closestClass = y[self.y_index[sample.tostring()]]
                if closestClass == y[self.y_index[observation.tostring()]]:
                    # both are the same class and keep the closest sample
                    continue
                else:  # different add to cleansed dataset
                    samples.append(X_train[idx])
                    X_train = np.delete(X_train, idx, axis=0)
            if len(samples) == n_samples:
                # no new samples on this pass, exit the while loop
                break
            # update the number of samples to check against the next run
            n_samples = len(samples)
        print("Number of samples selected: " + str(len(samples)))

        return samples
