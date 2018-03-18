import unittest
from apriori.apriori import apriori, GroupCounts, AprioriFrequentSets


class TestApriori(unittest.TestCase):

    def test_apriori(self):
        input_dict_1 = {0: ['A', 'B', 'C'],
                        1: ['A', 'B', 'C'],
                        2: ['A', 'B', 'D'],
                        3: ['A', 'B', 'D'],
                        4: ['A', 'B', 'E']}
        min_support_1 = 2
        group_counts_list_1 = [GroupCounts({'A'}, 5), GroupCounts({'B'}, 5), GroupCounts({'C'}, 2),
                               GroupCounts({'D'}, 2),
                               GroupCounts({'E'}, 1),
                               GroupCounts({'A', 'B'}, 5), GroupCounts({'A', 'C'}, 2), GroupCounts({'A', 'D'}, 2),
                               GroupCounts({'B', 'C'}, 2), GroupCounts({'B', 'D'}, 2),
                               GroupCounts({'A', 'B', 'C'}, 2), GroupCounts({'A', 'B', 'D'}, 2)]
        expected_apriori_frequent_sets_1 = AprioriFrequentSets(group_counts_list_1, min_support_1)
        result = apriori(input_dict_1, min_support_1)

        self.assertEqual(expected_apriori_frequent_sets_1, result)
        return result


