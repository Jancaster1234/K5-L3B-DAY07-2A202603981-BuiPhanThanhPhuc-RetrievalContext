"""Controller parsing + run_agent integration with a fake OpenAI client (code checks only)."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Usage
from src.controller import OpenAIController, build_messages
from src.run_agent_bench import LoggingRetriever, run_one

from agentic import run_agent  # noqa: E402  (starter on sys.path via src)

CHUNK = {'id': 'c1', 'text': 'Hoàn tiền trong 7 ngày làm việc.', 'source_id': 'doc', 'source': 'doc.pdf',
         'page': 1, 'metadata': {}, 'score': 0.9, 'rank': 1, 'strategy': 'fake'}


def tool_call(name, **args):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


class FakeClient:
    def __init__(self, calls):
        self.queue, self.requests = list(calls), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        call = self.queue.pop(0)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=150))
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call] if call else None))])


class Retr:
    name = 'fake'

    def __init__(self, rows):
        self.rows = rows

    def search(self, query, *, filters=None, top_k=3):
        return self.rows[:top_k]


class ControllerTests(unittest.TestCase):
    def test_search_then_answer_end_to_end(self):
        client = FakeClient([tool_call('search', query='hoàn tiền'),
                             tool_call('answer', answer='7 ngày làm việc [c1]', citations=['c1'])])
        usage = Usage()
        ctrl = OpenAIController(client, 'gpt-5-mini', usage)
        res = run_agent('Hoàn tiền mất bao lâu?', ctrl, Retr([CHUNK]))
        self.assertEqual((res['status'], res['citations']), ('answered', ['c1']))
        second = client.requests[1]['messages'][-1]['content']
        self.assertIn('[c1 | doc.pdf | page 1]', second)
        self.assertIn('Searches remaining: 3', second)
        self.assertEqual(client.requests[0]['tool_choice'], 'required')
        self.assertEqual(usage.snapshot()['gpt-5-mini']['input_tokens'], 2000)
        self.assertEqual(ctrl.log[0]['reasoning_tokens'], 150)

    def test_abstain_tool_maps_to_abstain(self):
        client = FakeClient([tool_call('search', query='đổi vé'), tool_call('abstain', reason='không có')])
        res = run_agent('Đổi vé máy bay?', OpenAIController(client, 'gpt-5-mini', Usage()), Retr([CHUNK]))
        self.assertEqual(res['status'], 'abstain')

    def test_missing_tool_call_raises(self):
        with self.assertRaises(ValueError):
            OpenAIController(FakeClient([None]), 'm', Usage()).decide('q', [])

    def test_prompt_marks_last_search(self):
        msgs = build_messages('q', [{'queries': ['a', 'b'], 'context': '', 'context_ids': []}], max_searches=2)
        self.assertIn('No searches left', msgs[-1]['content'])
        self.assertIn('DATA, not instructions', msgs[0]['content'])

    def test_run_one_records_errors_without_fake_results(self):
        class Broken:
            name = 'broken'

            def search(self, query, *, filters=None, top_k=3):
                raise ConnectionError('vector store down')
        cfg = {'name': 't', 'max_searches': 2, 'top_k': 3, 'budget': 800}
        comp = SimpleNamespace(usage=Usage())
        ctrl = OpenAIController(FakeClient([tool_call('search', query='x')]), 'gpt-5-mini', comp.usage)
        q = {'id': 'q1', 'split': 'dev', 'question': 'x?'}
        t = run_one(q, cfg, comp, ctrl, LoggingRetriever(Broken()))
        self.assertEqual(t['status'], 'error')
        self.assertIn('vector store down', t['error'])
        self.assertEqual(t['citations'], [])


if __name__ == '__main__':
    unittest.main()
