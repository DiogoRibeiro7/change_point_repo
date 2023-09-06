import unittest
from cumsum import CUSUM

class TestCUSUM(unittest.TestCase):
    
    def setUp(self):
        self.cusum_instance = CUSUM()
        
    def test_initialization(self):
        self.assertEqual(self.cusum_instance.k, 0.25)
        self.assertEqual(self.cusum_instance.h, 8.0)
        self.assertEqual(self.cusum_instance.burnin, 50)
        self.assertEqual(self.cusum_instance.mu, 0.0)
        self.assertEqual(self.cusum_instance.sigma, 1.0)
    
    def test_update_mean_variance(self):
        self.cusum_instance.update_mean_variance(5.0)
        self.assertAlmostEqual(self.cusum_instance.mu, 2.5, places=2)
        self.assertAlmostEqual(self.cusum_instance.sigma, 2.598, places=2)
    
    def test_update_statistics(self):
        self.cusum_instance.update_mean_variance(5.0)
        self.cusum_instance.update_statistics(5.0)
        self.assertAlmostEqual(self.cusum_instance.S[-1], 0.712, places=3)
        self.assertEqual(self.cusum_instance.T[-1], 0.0)
    
    def test_decision_rule(self):
        self.cusum_instance.burnin = 1
        self.cusum_instance.h = 1.0
        self.cusum_instance.update_mean_variance(5.0)
        self.cusum_instance.update_statistics(5.0)
        self.cusum_instance.decision_rule(1)
        self.assertEqual(len(self.cusum_instance.changepoints), 0)
    
    def test_process(self):
        data_stream = [0.0, 5.0, 5.0]
        self.cusum_instance.process(data_stream)
        self.assertEqual(len(self.cusum_instance.changepoints), 0)
        
    def test_reset(self):
        self.cusum_instance.update_mean_variance(5.0)
        self.cusum_instance.reset()
        self.assertEqual(self.cusum_instance.mu, 0.0)
        self.assertEqual(self.cusum_instance.sigma, 1.0)

# Run the tests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestCUSUM))