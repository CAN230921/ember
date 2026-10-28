"""Actual prompt boundaries and exact template-option forwarding; CPU only."""
import unittest

from engine.tokenization import chat_token_ids,prompt_starts_in_reasoning


class Tokenizer:
    def __init__(self,tail='',encoded=None):
        self.tail=tail;self.encoded=[10,11,12] if encoded is None else encoded
        self.templates=[];self.decodes=[]

    def apply_chat_template(self,messages,**options):
        self.templates.append((messages,options))
        return self.encoded

    def decode(self,ids,**options):
        self.decodes.append((list(ids),options))
        return self.tail


class TokenizationTests(unittest.TestCase):
    def test_legacy_ids_and_template_options_are_unchanged_when_preserve_omitted(self):
        tokenizer=Tokenizer();messages=[dict(role='user',content='hi')]
        self.assertEqual(chat_token_ids(tokenizer,messages),[10,11,12])
        self.assertEqual(tokenizer.templates[0],(messages,dict(tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)))

    def test_explicit_preserve_thinking_is_forwarded_to_real_template_with_tools(self):
        tools=[dict(type='function',function=dict(name='add'))]
        for preserve in (False,True):
            tokenizer=Tokenizer()
            ids=chat_token_ids(tokenizer,[dict(role='user',content='hi')],True,tools,
                preserve_thinking=preserve)
            self.assertEqual(ids,[10,11,12])
            options=tokenizer.templates[0][1]
            self.assertIs(options['enable_thinking'],True)
            self.assertIs(options['preserve_thinking'],preserve)
            self.assertEqual(options['tools'],tools)
        with self.assertRaises(ValueError):chat_token_ids(Tokenizer(),[],preserve_thinking='true')

    def test_template_mapping_and_listlike_return_conventions_still_work(self):
        class ListLike:
            def tolist(self):return [10,11,12]
        for encoded in (dict(input_ids=[10,11,12]),ListLike(),(10,11,12)):
            self.assertEqual(chat_token_ids(Tokenizer(encoded=encoded),[]),[10,11,12])

    def test_actual_assistant_prefix_open_reasoning_is_detected(self):
        for tail in ('<|im_start|>assistant\n<think>\n',
                '<|im_start|>assistant\n<think>already prefetched reasoning',
                '<|im_start|>assistant\n<think>x</think><think>\n'):
            self.assertTrue(prompt_starts_in_reasoning(Tokenizer(tail),[1,2,3]))

    def test_closed_disabled_or_tool_prefix_does_not_start_reasoning(self):
        for tail in ('<|im_start|>assistant\n',
                '<|im_start|>assistant\n<think>\n\n</think>\n',
                '<|im_start|>assistant\n<tool_call>\n',
                '<|im_start|>assistant\n<think>history<|im_end|>',
                '<|im_start|>assistant\n<think>history<|endoftext|>'):
            self.assertFalse(prompt_starts_in_reasoning(Tokenizer(tail),[1,2,3]))

    def test_user_literals_and_previous_assistant_thoughts_do_not_determine_state(self):
        tails=['<|im_start|>user\nThe literal <think> belongs in text.',
            '<|im_start|>user\n<think>literal<|im_end|><|im_start|>assistant\n',
            '<|im_start|>assistant\n<think>older thought<|im_end|><|im_start|>user\nnext',
            '<|im_start|>assistant\n<think>older thought<|im_end|><|im_start|>assistant\n',
            'literal text <think> without an assistant generation prefix']
        for tail in tails:
            self.assertFalse(prompt_starts_in_reasoning(Tokenizer(tail),[1,2,3]))

    def test_only_actual_token_tail_is_decoded_and_ids_are_not_changed(self):
        ids=list(range(300));before=list(ids);tokenizer=Tokenizer('<|im_start|>assistant\n<think>\n')
        self.assertTrue(prompt_starts_in_reasoning(tokenizer,ids))
        self.assertEqual(ids,before)
        self.assertEqual(tokenizer.decodes,[(ids[-128:],dict(skip_special_tokens=False,
            clean_up_tokenization_spaces=False))])
        self.assertEqual(tokenizer.templates,[])
        self.assertFalse(prompt_starts_in_reasoning(tokenizer,[]))
        for invalid in (0,-1,True,1.5):
            with self.assertRaises(ValueError):prompt_starts_in_reasoning(tokenizer,ids,tail_tokens=invalid)


if __name__=='__main__':unittest.main()
