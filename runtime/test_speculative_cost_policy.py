"""CPU decisions for measured MTP costs; no model/GPU is loaded."""
import unittest

from engine.speculative_policy import AdaptiveDraftDepth, CostAwareDraftDepth


class CostPolicyTests(unittest.TestCase):
    def observe(self, policy, depth, tokens, seconds):
        policy.observe_round(True, depth=depth, accepted_tokens=tokens, elapsed_seconds=seconds)

    def warm(self, policy, cost1, cost2):
        for _ in range(2): self.observe(policy, 2, 3, cost2)
        for _ in range(2): self.observe(policy, 1, 2, cost1)

    def test_high_acceptance_does_not_override_slower_rounds(self):
        policy=CostAwareDraftDepth()
        self.warm(policy, .01, .025)
        self.assertEqual(policy.depth(), 1)  # 200 vs 120 tokens/s

    def test_lower_acceptance_can_still_be_faster(self):
        policy=CostAwareDraftDepth()
        for _ in range(2):
            policy.observe_round(False, depth=2, accepted_tokens=1, elapsed_seconds=.005)
            self.observe(policy, 1, 2, .02)
        self.assertEqual(policy.depth(), 2)

    def test_measured_total_cost_not_mean_of_instantaneous_rates(self):
        policy=CostAwareDraftDepth()
        self.observe(policy, 1, 1, .001)
        self.observe(policy, 1, 2, .099)
        self.assertAlmostEqual(policy.rate(1), 30.)

    def test_exploration_is_bounded_and_revisits_old_shape(self):
        policy=CostAwareDraftDepth(revisit=3)
        self.assertEqual([policy.depth() for _ in range(10)], [2]*10)
        self.warm(policy, .01, .03)
        self.assertEqual(policy.depth(), 1)
        for _ in range(3): self.observe(policy, 1, 2, .01)
        self.assertEqual(policy.depth(), 2)
        self.observe(policy, 2, 3, .03)
        self.assertEqual(policy.depth(), 1)

    def test_noise_margin_and_request_isolation(self):
        a,b=CostAwareDraftDepth(),CostAwareDraftDepth()
        self.warm(a, .02, .0303)  # 1% is below the 3% switch margin.
        self.assertEqual(a.depth(), 2)
        self.assertEqual(b.snapshot()['rounds'], 0)
        self.assertIsNone(b.rate(1))

    def test_bad_observations_do_not_poison_state(self):
        policy=CostAwareDraftDepth()
        for seconds in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): self.observe(policy, 1, 1, seconds)
        for depth,tokens in ((0,1),(3,1),(1,0),(1,3),(2,4)):
            with self.assertRaises(ValueError): self.observe(policy, depth, tokens, .01)
        self.assertEqual(policy.rounds, 0)

    def test_old_policy_remains_unchanged(self):
        policy=AdaptiveDraftDepth()
        self.assertEqual(policy.depth(), 2)
        for _ in range(4): policy.observe(False)
        self.assertEqual(policy.depth(), 1)


if __name__ == '__main__': unittest.main()
