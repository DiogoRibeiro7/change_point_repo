import unittest

from ewma import EWMA

# Define test cases for the EWMA class
class TestEWMA(unittest.TestCase):
    
    def setUp(self):
        self.ewma_instance = EWMA()
        
    def test_update_mean_variance(self):
        self.ewma_instance.update_mean_variance(5.0)
        self.assertAlmostEqual(self.ewma_instance.mu, 2.5, places=2)
        self.assertAlmostEqual(self.ewma_instance.sigma, 2.598, places=2)
        
    def test_update_statistics(self):
        self.ewma_instance.update_mean_variance(5.0)
        self.ewma_instance.update_statistics(1, 5.0)
        self.assertAlmostEqual(self.ewma_instance.Z[-1], 0.5, places=2)
        self.assertAlmostEqual(self.ewma_instance.sigma_Z[-1], 0.2598, places=2)
        
    def test_decision_rule(self):
        self.ewma_instance.update_mean_variance(5.0)
        self.ewma_instance.update_statistics(1, 5.0)
        self.ewma_instance.decision_rule(51)
        self.assertEqual(len(self.ewma_instance.changepoints), 1)
        
        self.ewma_instance.update_mean_variance(10.0)
        self.ewma_instance.update_statistics(52, 10.0)
        self.ewma_instance.decision_rule(52)
        self.assertEqual(len(self.ewma_instance.changepoints), 2)
        
# Run the tests to validate the EWMA class
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestEWMA))
