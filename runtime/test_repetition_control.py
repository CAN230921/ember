"""CPU exact-token controls: periodic loops, template negatives and bounds."""
import unittest

from repetition_control import ReasoningRepetitionDetector


class RepetitionControlTests(unittest.TestCase):
    def observe(self,detector,tokens):
        return [detector.observe(token) for token in tokens]

    def test_four_nonoverlapping_cycles_after_minimum_reasoning_trigger(self):
        detector=ReasoningRepetitionDetector()
        warm=list(range(1000,1384));cycle=list(range(32))
        result=self.observe(detector,warm+cycle*4)
        self.assertFalse(any(result[:-1]));self.assertTrue(result[-1])
        self.assertEqual(detector.seen,512)
        self.assertEqual(detector.period_tokens,32)

    def test_full_periodicity_covers_intervening_tokens_not_only_four_headers(self):
        detector=ReasoningRepetitionDetector()
        header=list(range(32));tokens=list(range(1000,1400))
        for branch in range(4):
            tokens+=header+list(range(2000+branch*80,2048+branch*80))
        self.assertFalse(any(self.observe(detector,tokens)))
        self.assertFalse(detector.triggered)

    def test_longer_paragraph_cycles_are_detected(self):
        detector=ReasoningRepetitionDetector()
        warm=list(range(1000,1500));cycle=list(range(96))
        result=self.observe(detector,warm+cycle*4)
        self.assertTrue(any(result));self.assertEqual(detector.period_tokens,96)
        # Four equal 32-gram starts plus their complete intervening cycles are
        # enough; the fourth paragraph need not finish before closing planning.
        self.assertEqual(detector.seen,500+96*3+32)

    def test_three_cycles_and_overlapping_four_grams_do_not_trigger(self):
        detector=ReasoningRepetitionDetector()
        self.assertFalse(any(self.observe(detector,list(range(1000,1500))+list(range(32))*3)))
        detector=ReasoningRepetitionDetector()
        self.assertFalse(any(self.observe(detector,list(range(1000,1500))+[9]*35)))

    def test_episode_minimum_is_not_a_requirement_for_512_repeated_tokens(self):
        detector=ReasoningRepetitionDetector()
        self.assertFalse(any(self.observe(detector,list(range(32))*4)))
        self.assertEqual(detector.seen,128)
        self.assertFalse(detector.triggered)

    def test_long_nonrepeating_reasoning_is_bounded(self):
        for window in (2048,4096):
            detector=ReasoningRepetitionDetector(window_tokens=window)
            self.assertFalse(any(self.observe(detector,range(25000))))
            state=detector.snapshot()
            self.assertEqual(state['buffered_tokens'],window)
            self.assertEqual(state['indexed_grams'],window-31)
            self.assertEqual(state['distinct_grams'],window-31)
            self.assertEqual(len(detector._tail),32)
            self.assertLessEqual(sum(len(p) for p in detector._positions.values()),window-31)
            self.assertEqual(state['indexed_long_blocks'],window-95)
            self.assertLessEqual(sum(len(p) for p in detector._long_positions.values()),window-95)

    def test_identical_long_blocks_with_variable_surrounding_prose_trigger(self):
        detector=ReasoningRepetitionDetector()
        tokens=list(range(1000,1400));block=list(range(96))
        for number in range(4):
            tokens+=list(range(2000+number*100,2000+number*100+number*11+3))+block
        self.assertTrue(any(self.observe(detector,tokens)))
        self.assertEqual(detector.reason,'repeated_reasoning_block')
        self.assertIsNone(detector.period_tokens)

    def test_three_long_blocks_or_expired_fourth_do_not_trigger(self):
        block=list(range(96))
        for tokens in (list(range(1000,1400))+block+list(range(1500,1511))+block+
                       list(range(1600,1623))+block,
                       list(range(1000,1400))+block+list(range(1500,1511))+block+
                       list(range(1600,1623))+block+list(range(3000,5200))+block):
            self.assertFalse(any(self.observe(ReasoningRepetitionDetector(),tokens)))

    def test_expired_occurrences_cannot_combine_with_recent_grams(self):
        detector=ReasoningRepetitionDetector()
        cycle=list(range(32))
        tokens=list(range(1000,1500))+cycle*3+list(range(3000,5100))+cycle
        self.assertFalse(any(self.observe(detector,tokens)))
        self.assertFalse(detector.triggered)

    def test_reset_detaches_the_previous_reasoning_episode(self):
        detector=ReasoningRepetitionDetector()
        self.observe(detector,list(range(1000,1384))+list(range(32))*4)
        self.assertTrue(detector.triggered);detector.reset()
        self.assertEqual(detector.seen,0);self.assertFalse(detector.triggered)
        self.assertEqual(detector.snapshot()['buffered_tokens'],0)
        self.assertFalse(any(self.observe(detector,list(range(32))*3)))

    def test_settings_and_token_ids_are_validated(self):
        for settings in ({'gram_tokens':31},{'repeats':3},{'min_tokens':511},
                         {'window_tokens':8192},{'gram_tokens':True},
                         {'gram_tokens':1024,'repeats':5}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                ReasoningRepetitionDetector(**settings)
        detector=ReasoningRepetitionDetector()
        for token in (True,-1,1.5):
            with self.assertRaises(ValueError):detector.observe(token)


if __name__=='__main__':unittest.main()
