"""Protocol boundaries: XML fragmentation, UTF-8-safe framing and tool history."""
import json
import unittest
from api_protocol import normalize_request,OutputFramer,RequestError

MODEL='qwen3.8-flash-next-gb10'
TOOLS=[dict(type='function',function=dict(name='add_numbers',parameters=dict(type='object',
    properties=dict(a=dict(type='integer'),b=dict(type='integer')),required=['a','b'],additionalProperties=False)))]
CALL='<tool_call>\n<function=add_numbers>\n<parameter=a>\n173\n</parameter>\n<parameter=b>\n269\n</parameter>\n</function>\n</tool_call>'


class ProtocolTests(unittest.TestCase):
    def body(self,**options):
        return dict(model=MODEL,messages=[dict(role='user',content='计算')],**options)

    def test_default_and_explicit_thinking_switch_keep_unbounded_semantics(self):
        options=normalize_request(self.body(),MODEL)
        self.assertFalse(options['thinking'])
        self.assertIsNone(options['reasoning_effort'])
        self.assertIsNone(options['thinking_budget'])
        self.assertIsNone(options['preserve_thinking'])
        for field in ('enable_thinking','chat_template_kwargs'):
            body=self.body(**{field:True if field=='enable_thinking' else dict(enable_thinking=True)})
            options=normalize_request(body,MODEL)
            self.assertTrue(options['thinking']);self.assertIsNone(options['thinking_budget'])

    def test_effort_maps_to_real_budget_and_explicit_budget_overrides(self):
        for effort,budget in (('low',1024),('medium',4096),('high',16384)):
            for nested in (False,True):
                flags=dict(reasoning_effort=effort)
                options=normalize_request(self.body(**(dict(chat_template_kwargs=flags) if nested else flags)),MODEL)
                self.assertTrue(options['thinking']);self.assertEqual(options['thinking_budget'],budget)
                self.assertEqual(options['reasoning_effort'],effort)
                explicit=normalize_request(self.body(reasoning_effort=effort,thinking_budget_tokens=37),MODEL)
                self.assertEqual(explicit['thinking_budget'],37)
        for effort in ('none','off'):
            options=normalize_request(self.body(reasoning_effort=effort),MODEL)
            self.assertFalse(options['thinking']);self.assertIsNone(options['thinking_budget'])

    def test_budget_aliases_can_enable_thinking_and_agree_across_locations(self):
        for field in ('thinking_budget','thinking_budget_tokens'):
            for nested in (False,True):
                flags={field:23}
                options=normalize_request(self.body(**(dict(chat_template_kwargs=flags) if nested else flags)),MODEL)
                self.assertTrue(options['thinking']);self.assertEqual(options['thinking_budget'],23)
        options=normalize_request(self.body(thinking_budget=23,thinking_budget_tokens=23,
            chat_template_kwargs=dict(thinking_budget=23,thinking_budget_tokens=23)),MODEL)
        self.assertEqual(options['thinking_budget'],23)

    def test_thinking_conflicts_are_explicit_request_errors(self):
        conflicts=[dict(enable_thinking=False,reasoning_effort='low'),
            dict(enable_thinking=True,reasoning_effort='none'),
            dict(enable_thinking=True,reasoning_effort='off'),
            dict(enable_thinking=False,thinking_budget=1),
            dict(reasoning_effort='off',thinking_budget_tokens=1),
            dict(enable_thinking=False,chat_template_kwargs=dict(enable_thinking=True)),
            dict(reasoning_effort='high',chat_template_kwargs=dict(reasoning_effort='low')),
            dict(thinking_budget=1,thinking_budget_tokens=2),
            dict(thinking_budget=1,chat_template_kwargs=dict(thinking_budget_tokens=2)),
            dict(preserve_thinking=True,chat_template_kwargs=dict(preserve_thinking=False))]
        for flags in conflicts:
            with self.subTest(flags=flags),self.assertRaises(RequestError):
                normalize_request(self.body(**flags),MODEL)

    def test_invalid_thinking_values_and_unknown_template_options_are_rejected(self):
        invalid=dict(reasoning_effort=(None,True,1,'xhigh','HIGH',''),
            thinking_budget=(None,True,False,0,-1,1.5,'1024'),
            thinking_budget_tokens=(None,True,0,-1,'1024'),
            enable_thinking=(None,1,'true'),preserve_thinking=(None,1,'true'))
        for field,values in invalid.items():
            for value in values:
                for nested in (False,True):
                    flags={field:value}
                    with self.subTest(field=field,value=value,nested=nested),self.assertRaises(RequestError):
                        normalize_request(self.body(**(dict(chat_template_kwargs=flags) if nested else flags)),MODEL)
        for template in ([],False,'',dict(thinking_depth=4)):
            with self.subTest(template=template),self.assertRaises(RequestError):
                normalize_request(self.body(chat_template_kwargs=template),MODEL)

    def test_preserve_thinking_is_explicit_and_does_not_block_tools(self):
        for preserve in (False,True):
            options=normalize_request(self.body(tools=TOOLS,chat_template_kwargs=dict(
                enable_thinking=True,preserve_thinking=preserve)),MODEL)
            self.assertTrue(options['thinking']);self.assertEqual(options['tools'],TOOLS)
            self.assertIs(options['preserve_thinking'],preserve)
        messages=[dict(role='user',content='先前问题'),
            dict(role='assistant',content='42',reasoning_content='先前思考'),
            dict(role='user',content='继续')]
        options=normalize_request(dict(model=MODEL,messages=messages,preserve_thinking=True),MODEL)
        self.assertEqual(options['messages'][1]['reasoning_content'],'先前思考')

    def test_prompt_opened_reasoning_closing_fragments_never_enter_content(self):
        text='检查173+269。</think>{"answer":442}<|im_end|>'
        for width in range(1,24):
            with self.subTest(width=width):
                events=[];parser=OutputFramer([],events.append,starts_in_reasoning=True)
                for end in range(width,len(text)+width,width):parser.feed(text[:end])
                parser.feed(text,final=True)
                self.assertEqual(parser.message(),dict(role='assistant',content='{"answer":442}',reasoning_content='检查173+269。'))
                self.assertEqual(''.join(event.get('reasoning_content','') for event in events),'检查173+269。')
                self.assertEqual(''.join(event.get('content','') for event in events),'{"answer":442}')

    def test_prompt_opened_reasoning_then_fragmented_tool_call(self):
        text='先调用工具。</think>'+CALL+'<|im_end|>'
        for width in (1,2,7,19):
            events=[];parser=OutputFramer(TOOLS,events.append,starts_in_reasoning=True)
            for end in range(width,len(text)+width,width):parser.feed(text[:end])
            parser.feed(text,final=True)
            message=parser.message();self.assertEqual(message['reasoning_content'],'先调用工具。')
            self.assertIsNone(message['content']);self.assertEqual(len(message['tool_calls']),1)
            self.assertEqual(''.join(event.get('content','') for event in events),'')

    def test_truncated_open_reasoning_preserves_field_without_partial_control_tags(self):
        for text,expected in (('思考仍在继续','思考仍在继续'),('思考仍在继续</thi','思考仍在继续'),
                ('思考仍在继续<tool_','思考仍在继续')):
            events=[];parser=OutputFramer([],events.append,starts_in_reasoning=True)
            for end in range(1,len(text)+1):parser.feed(text[:end])
            parser.feed(text,final=True)
            self.assertEqual(parser.message(),dict(role='assistant',content='',reasoning_content=expected))
            self.assertEqual(''.join(event.get('content','') for event in events),'')
        with self.assertRaises(ValueError):OutputFramer([],starts_in_reasoning=1)

    def test_partial_control_tags_never_leak_in_content(self):
        text='<think>检查173+269。</think>'+CALL+'<|im_end|>'
        for width in range(1,27):
            with self.subTest(width=width):
                events=[]; parser=OutputFramer(TOOLS,events.append)
                for end in range(width,len(text)+width,width):parser.feed(text[:end])
                parser.feed(text,final=True)
                message=parser.message()
                self.assertEqual(message['reasoning_content'],'检查173+269。')
                self.assertIsNone(message['content'])
                self.assertEqual(len(message['tool_calls']),1)
                self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments']),dict(a=173,b=269))
                self.assertEqual(''.join(x.get('content','') for x in events),'')
                self.assertEqual(len([x for x in events if 'tool_calls' in x]),1)

    def test_normal_answer_stream_matches_final_content(self):
        text='北京今天晴，气温21℃。普通文本含 <x>。<|im_end|>'
        for width in (1,2,7,16):
            events=[]; parser=OutputFramer([],events.append)
            for end in range(width,len(text)+width,width):parser.feed(text[:end])
            parser.feed(text,final=True)
            self.assertEqual(parser.content,text.removesuffix('<|im_end|>'))
            self.assertEqual(''.join(x.get('content','') for x in events),parser.content)

    def test_tool_roundtrip_arguments_decode_to_template_object(self):
        body=dict(model=MODEL,messages=[dict(role='user',content='计算'),
            dict(role='assistant',content=None,tool_calls=[dict(id='call_x',type='function',
                function=dict(name='add_numbers',arguments='{"a":173,"b":269}'))]),
            dict(role='tool',tool_call_id='call_x',content='{"result":442}')],tools=TOOLS)
        options=normalize_request(body,MODEL)
        self.assertEqual(options['messages'][1]['tool_calls'][0]['function']['arguments'],dict(a=173,b=269))
        self.assertEqual(options['messages'][2]['tool_call_id'],'call_x')

    def test_explicitly_reject_unsupported_sampling(self):
        with self.assertRaises(RequestError):
            normalize_request(dict(model=MODEL,messages=[dict(role='user',content='hi')],temperature=.7),MODEL)

    def test_incomplete_or_unavailable_tool_is_an_error(self):
        for text,tools in ((CALL[:-3],TOOLS),(CALL,[])):
            with self.subTest(text=text):
                parser=OutputFramer(tools)
                with self.assertRaises(ValueError):parser.feed(text,final=True)


if __name__=='__main__':unittest.main()
