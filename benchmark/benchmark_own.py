import unittest
import time
from apriori.apriori import apriori
import random
import string


class BenchmarkApriori(unittest.TestCase):
    """
    Tests the apriori call on a larger test data set to benchmark
    against other known apriori implementations
    TODO: Set up test data with set amount of density/sparsity
    TODO: Check memory usage (from app?)
    """

    def test_apriori_against_apyori(self):

        num_input_rows:int = 100_000
        num_unique_input_els = len(string.ascii_letters)
        input_rows = [random.sample(string.ascii_letters, random.randint(2, 6))
                      for _ in range(num_input_rows)]
        min_support_1 = 100

        tf_apriori_start = time.time()
        apriori(input_rows, min_support_1)
        tf_apriori_end = time.time()
        tf_apriori_duration = tf_apriori_end - tf_apriori_start
        print(f"Tensorflow apriori took: {tf_apriori_duration}")

        apyori_start = time.time()
        # TODO: call apyori
        apyori_end = time.time()
        apyori_duration = apyori_end - apyori_start
        print(f"Apyori apriori took: {apyori_duration}")

        self.assertGreater(apyori_duration, tf_apriori_duration)
