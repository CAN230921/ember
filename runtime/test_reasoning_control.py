"""CPU token-history oracles for natural and bounded reasoning termination."""
import unittest

from reasoning_control import ReasoningControl


class ReasoningControlTests(unittest.TestCase):
    def test_disabled_and_unbounded_are_distinct_and_do_not_force(self):
        disabled=ReasoningControl(False,None,None,[])
        disabled.observe([1,2,3]);self.assertEqual(disabled.plan(),(None,None))
        self.assertEqual(disabled.snapshot()['used_tokens'],0)
        unbounded=ReasoningControl(True,None,None,[30,31])
        unbounded.observe([1,2,3]);self.assertEqual(unbounded.plan(),(None,None))
        self.assertEqual(unbounded.snapshot()['used_tokens'],3)
        self.assertFalse(unbounded.snapshot()['budget_triggered'])

    def test_natural_multitoken_close_stops_counting_before_answer(self):
        control=ReasoningControl(True,'medium',8,[30,31])
        control.observe([1,2,30]);self.assertTrue(control.active)
        control.observe([1,2,30,31,40,41]);self.assertFalse(control.active)
        self.assertEqual(control.plan(),(None,None))
        self.assertEqual(control.snapshot(),dict(enabled=True,effort='medium',budget_tokens=8,
            used_tokens=4,budget_triggered=False,forced_control_tokens=0,loop_triggered=False,
            reason=None,trigger_at_tokens=None,loop_period_tokens=None,reasoning_reopened=False,
            termination_reason=None,intercepted_reasoning_tokens=0))

    def test_hard_cap_forces_only_control_and_excludes_forced_from_reasoning_count(self):
        control=ReasoningControl(True,'low',2,[30,31])
        self.assertEqual(control.plan(),(None,2))
        generated=[1];control.observe(generated);self.assertEqual(control.plan(),(None,1))
        generated.append(2);control.observe(generated)
        for token in (30,31):
            self.assertEqual(control.plan(),(token,None))
            generated.append(token);control.observe(generated)
        generated.extend([40,41]);control.observe(generated)
        self.assertFalse(control.active);self.assertEqual(control.plan(),(None,None))
        self.assertEqual(control.snapshot(),dict(enabled=True,effort='low',budget_tokens=2,
            used_tokens=2,budget_triggered=True,forced_control_tokens=2,loop_triggered=False,
            reason=None,trigger_at_tokens=None,loop_period_tokens=None,reasoning_reopened=False,
            termination_reason=None,intercepted_reasoning_tokens=0))

    def test_natural_close_at_exact_cap_does_not_insert_another_close(self):
        control=ReasoningControl(True,'low',3,[30,31])
        control.observe([1,30,31])
        self.assertEqual(control.plan(),(None,None));self.assertFalse(control.active)
        self.assertEqual(control.used,3);self.assertEqual(control.forced,0)
        self.assertFalse(control.triggered)

    def test_cap_during_natural_close_completes_only_missing_suffix(self):
        control=ReasoningControl(True,'low',2,[30,31,32])
        generated=[1,30];control.observe(generated)
        for token in (31,32):
            self.assertEqual(control.plan(),(token,None))
            generated.append(token);control.observe(generated)
        self.assertEqual(generated,[1,30,31,32])
        self.assertEqual(control.used,2);self.assertEqual(control.forced,2)
        self.assertFalse(control.active)

    def test_cap_after_two_natural_marker_tokens_preserves_existing_prefix(self):
        control=ReasoningControl(True,'low',3,[30,31,32])
        control.observe([1,30,31]);self.assertEqual(control.plan(),(32,None))
        control.observe([1,30,31,32]);self.assertEqual(control.forced,1)
        self.assertFalse(control.active)

    def test_output_length_can_end_mid_forced_marker_without_exceeding_reasoning_cap(self):
        control=ReasoningControl(True,'low',1,[30,31])
        control.observe([1]);self.assertEqual(control.plan(),(30,None))
        # The independent overall output budget is two tokens. The caller
        # terminates with length here, rather than adding the pending 31.
        control.observe([1,30]);self.assertEqual(control.plan(),(31,None))
        self.assertEqual(control.used,1);self.assertEqual(control.forced,1)
        self.assertTrue(control.active)

    def test_observation_never_recounts_seen_tokens_and_rejects_shorter_history(self):
        control=ReasoningControl(True,None,None,[30,31]);control.observe([1,2])
        with self.assertRaises(ValueError):control.observe([1])
        control=ReasoningControl(True,None,None,[30,31]);control.observe([1,2]);control.observe([1,2])
        self.assertEqual(control.used,2)
        control.observe([1,2,3]);self.assertEqual(control.used,3)

    def test_wrong_control_token_and_over_cap_model_output_are_errors(self):
        control=ReasoningControl(True,'low',1,[30,31]);control.observe([1]);control.plan()
        with self.assertRaises(RuntimeError):control.observe([1,99])
        control=ReasoningControl(True,'low',1,[30,31])
        with self.assertRaises(RuntimeError):control.observe([1,2])

    def test_invalid_settings_are_rejected(self):
        settings=[(1,None,None,[]),(True,None,True,[30]),(True,None,0,[30]),
            (True,None,1,[]),(True,None,1,[True]),(True,None,1,[-1])]
        for args in settings:
            with self.subTest(args=args),self.assertRaises(ValueError):ReasoningControl(*args)

    def test_periodic_reasoning_forces_canonical_close_without_capping_answer(self):
        control=ReasoningControl(True,'high',16384,[9000,9001],opening_ids=[8000])
        generated=list(range(1000,1384))+list(range(32))*4
        control.observe(generated)
        self.assertEqual(control.used,512);self.assertTrue(control.loop_triggered)
        self.assertEqual(control.trigger_at_tokens,512)
        for token in (9000,9001):
            self.assertEqual(control.plan(),(token,None))
            generated.append(token);control.observe(generated)
        self.assertFalse(control.active);self.assertFalse(control.triggered)
        self.assertEqual(control.forced,2);self.assertEqual(control.used,512)
        # Natural answers may legitimately repeat. No answer cap or guard is
        # introduced, and the already-used reasoning allowance is not reset.
        generated+=list(range(32))*40;control.observe(generated)
        self.assertEqual(control.plan(),(None,None));self.assertEqual(control.used,512)
        state=control.snapshot()
        self.assertEqual(state['reason'],'repeated_reasoning_cycle')
        self.assertEqual(state['loop_period_tokens'],32)
        self.assertEqual(control.repetition.snapshot()['buffered_tokens'],0)
        self.assertFalse(state['reasoning_reopened'])

    def test_long_nonrepeating_reasoning_is_allowed_without_lowering_high_budget(self):
        control=ReasoningControl(True,'high',16384,[9000,9001])
        generated=list(range(10000,16000));control.observe(generated)
        self.assertEqual(control.plan(),(None,10384))
        self.assertEqual(control.used,6000);self.assertFalse(control.loop_triggered)
        self.assertLessEqual(control.repetition.snapshot()['buffered_tokens'],2048)

    def test_natural_answer_repetition_is_never_sent_to_reasoning_detector(self):
        control=ReasoningControl(True,'medium',4096,[9000,9001])
        generated=[1,9000,9001]+list(range(1000,1384))+list(range(32))*4
        control.observe(generated)
        self.assertFalse(control.active);self.assertEqual(control.used,3)
        self.assertFalse(control.loop_triggered);self.assertEqual(control.plan(),(None,None))

    def test_three_token_observations_complete_only_partial_loop_closing_marker(self):
        control=ReasoningControl(True,'medium',4096,[9000,9001,9002])
        tokens=list(range(1000,1382))+list(range(32))*4+[0,1,9000]
        generated=[]
        for start in range(0,len(tokens),3):
            generated+=tokens[start:start+3];control.observe(generated)
            control.observe(generated)  # Re-observation never recounts.
        self.assertEqual(control.used,513);self.assertEqual(control.trigger_at_tokens,512)
        self.assertEqual(control.plan(),(9001,None))
        for token in (9001,9002):
            self.assertEqual(control.plan(),(token,None))
            generated.append(token);control.observe(generated)
        self.assertEqual(control.used,513);self.assertEqual(control.forced,2)
        self.assertFalse(control.active)

    def test_reopened_reasoning_is_intercepted_without_reset_or_false_budget_accounting(self):
        control=ReasoningControl(True,'low',5,[9000,9001],opening_ids=[8000])
        generated=[1,2,9000,9001,41]
        control.observe(generated);self.assertEqual(control.used,4)
        generated+=[8000,71,72];control.observe(generated);control.observe(generated)
        self.assertFalse(control.active);self.assertTrue(control.reasoning_reopened)
        self.assertEqual(control.termination_reason,'reasoning_reopened')
        self.assertEqual(control.used,4);self.assertEqual(control.intercepted_reasoning_tokens,2)
        self.assertEqual(control.forced,0);self.assertFalse(control.triggered)
        self.assertEqual(control.plan(),(None,None))

    def test_reopened_after_exhausted_budget_does_not_force_m1_for_normal_answer(self):
        control=ReasoningControl(True,'low',2,[9000,9001],opening_ids=[8000])
        generated=[1,2];control.observe(generated)
        for token in (9000,9001):
            self.assertEqual(control.plan(),(token,None))
            generated.append(token);control.observe(generated)
        self.assertEqual(control.plan(),(None,None))
        generated+=[41,42,43];control.observe(generated)
        self.assertEqual(control.plan(),(None,None))
        generated+=[8000,71,72];control.observe(generated)
        self.assertTrue(control.reasoning_reopened);self.assertEqual(control.used,2)
        self.assertEqual(control.forced,2);self.assertEqual(control.intercepted_reasoning_tokens,2)

    def test_multitoken_opening_is_intercepted_only_after_complete_marker(self):
        control=ReasoningControl(True,'low',5,[9000,9001],opening_ids=[8000,8001,8002])
        generated=[1,9000,9001,41,8000];control.observe(generated)
        self.assertFalse(control.reasoning_reopened)
        generated+=[8001];control.observe(generated);self.assertFalse(control.reasoning_reopened)
        generated+=[8002,71,72];control.observe(generated)
        self.assertTrue(control.reasoning_reopened);self.assertEqual(control.intercepted_reasoning_tokens,2)
        self.assertEqual(control.used,3)

    def test_disabled_reasoning_does_not_intercept_opening_marker(self):
        control=ReasoningControl(False,None,None,[],opening_ids=[8000])
        control.observe([8000,1,2]);self.assertFalse(control.reasoning_reopened)
        self.assertEqual(control.used,0);self.assertEqual(control.intercepted_reasoning_tokens,0)

    def test_opening_marker_settings_are_validated(self):
        for opening in ([True],[-1],[9000,9001]):
            with self.subTest(opening=opening),self.assertRaises(ValueError):
                ReasoningControl(True,None,None,[9000,9001],opening_ids=opening)


if __name__=='__main__':unittest.main()
