import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = ROOT / "spark_studio_node.py"
spec = importlib.util.spec_from_file_location("spark_studio_node", NODE_PATH)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class FakeOpener:
    def __init__(self):
        self.request = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "<think>draft</think>\n[verse]\nHello"}}]}).encode())


class SequenceOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": next(self.responses)}}]}).encode())


def song_graph(seconds=180):
    return {
        "18": {"class_type": "SparkStudioChat", "inputs": {"prompt": "Leo tribute"}},
        "19": {"class_type": "ShowText|pysssss", "inputs": {"text": ["18", 0]}},
        "17": {"class_type": "MiniMaxLyricsClean", "inputs": {"text": ["19", 0]}},
        "2": {"class_type": "FL_YuE2_Plan", "inputs": {
            "lyrics": ["17", 0], "style": "Slow R&B ballad, male vocal"}},
        "3": {"class_type": "FL_YuE2_Render", "inputs": {
            "composition": ["2", 0], "max_duration": seconds}},
    }


class SparkStudioNodeTests(unittest.TestCase):
    def test_endpoint_variants(self):
        self.assertEqual(node.chat_url("http://spark:8888"), "http://spark:8888/v1/chat/completions")
        self.assertEqual(node.chat_url("http://spark:7860/api/engine/v1"),
                         "http://spark:7860/api/engine/v1/chat/completions")
        with self.assertRaises(ValueError):
            node.chat_url("localhost:8888")

    def test_comfyui_package_registers_node(self):
        package_dir = NODE_PATH.parent
        package_spec = importlib.util.spec_from_file_location(
            "comfy_spark_studio", package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        package = importlib.util.module_from_spec(package_spec)
        import sys
        sys.modules[package_spec.name] = package
        try:
            package_spec.loader.exec_module(package)
            self.assertIn("SparkStudioChat", package.NODE_CLASS_MAPPINGS)
        finally:
            sys.modules.pop(package_spec.name, None)

    def test_request_and_response(self):
        opener = FakeOpener()
        with patch.object(node, "build_opener", return_value=opener):
            text = node.SparkStudioChat().generate("Lyrics please", "http://spark:8888/v1",
                                                  "test-model", 512, 0.7, 0.95)[0]
        self.assertEqual(text, "[verse]\nHello")
        payload = json.loads(opener.request.data)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Lyrics please"}])
        self.assertEqual(opener.request.full_url, "http://spark:8888/v1/chat/completions")

    def test_image_request_uses_openai_multimodal_parts(self):
        opener = FakeOpener()
        image = np.ones((1, 32, 32, 3), dtype=np.float32)
        with patch.object(node, "build_opener", return_value=opener):
            node.SparkStudioChat().generate("Describe", "http://spark:8888/v1",
                                             "vision", 64, 0, 1, image=image)
        content = json.loads(opener.request.data)["messages"][0]["content"]
        self.assertIn("IMAGE IS A PRIMARY LYRIC SOURCE", content[0]["text"])
        self.assertIn("Describe", content[0]["text"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_image_is_preserved_for_each_structured_lyric_section(self):
        responses = [
            "\n".join([f"Verse one image line {i} glows tonight" for i in range(5)]),
            "\n".join([f"Chorus image hook line {i} stays alive" for i in range(4)]),
            "\n".join([f"Verse two image line {i} changes shape" for i in range(5)]),
            "\n".join([f"Bridge image line {i} turns the light" for i in range(3)]),
            "\n".join([f"Outro image line {i} fades to blue" for i in range(2)]),
        ]
        opener = SequenceOpener(responses)
        image = np.ones((1, 16, 16, 3), dtype=np.float32)
        prompt = "Use [verse] [chorus] [bridge] [outro]. Two verses and two choruses."
        with patch.object(node, "build_opener", return_value=opener):
            node.SparkStudioChat().generate(
                prompt, "http://spark:8888/v1", "test-model", 512, 0.7, 0.95,
                image=image, prompt_graph=song_graph(), unique_id="18",
            )
        self.assertEqual(len(opener.requests), 5)
        for request in opener.requests:
            content = request["messages"][-1]["content"]
            self.assertIsInstance(content, list)
            self.assertIn("IMAGE IS A PRIMARY LYRIC SOURCE", content[0]["text"])
            self.assertEqual(content[1]["type"], "image_url")

    def test_empty_lyric_placeholder_is_removed_without_filtering_chinese(self):
        text = "[verse]\n我想你\n[空歌词 25]\n[chorus]\nStill here\nempty lyrics: 8"
        self.assertEqual(node.strip_empty_lyric_placeholders(text),
                         "[verse]\n我想你\n[chorus]\nStill here")
        self.assertEqual(node.section_lines("[verse]\n1. First real line\n- Second real line", 2),
                         ["First real line", "Second real line"])
        self.assertEqual(node.repair_text_encoding("I\ufffdm here"), "I'm here")
        self.assertFalse(node.section_is_novel(["First real line"], {"verse": ["First real line"]}))
        self.assertTrue(node.section_is_novel(["A genuinely new hook"], {"verse": ["First real line"]}))

    def test_explicit_two_verse_song_is_written_section_by_section(self):
        responses = [
            "\n".join([f"Verse one line {i} holds a memory" for i in range(5)]),
            "\n".join([f"Leo hook line {i} stays with me" for i in range(4)]),
            "\n".join([f"Verse two line {i} reveals something new" for i in range(5)]),
            "\n".join([f"Bridge line {i} turns the feeling around" for i in range(3)]),
            "\n".join([f"Outro line {i} leaves one final image" for i in range(2)]),
        ]
        opener = SequenceOpener(responses)
        prompt = ("Use [verse] [chorus] [bridge] [outro]. "
                  "Two verses and two choruses. Write about Leo.")
        with patch.object(node, "build_opener", return_value=opener):
            result = node.SparkStudioChat().generate(
                prompt, "http://spark:8888/v1", "test-model", 512, 0.7, 0.95,
                prompt_graph=song_graph(), unique_id="18",
            )[0]
        self.assertEqual(len(opener.requests), 5)
        self.assertEqual(result.count("[verse]"), 2)
        self.assertEqual(result.count("[chorus]"), 2)
        chorus = result.split("[chorus]\n", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(result.count(chorus), 2)
        self.assertEqual(node.lyric_size(result)[0], 23)

    def test_requested_song_sections_are_required_in_final_output(self):
        required = node.requested_sections("Use [verse], [chorus], [bridge], and [outro].")
        self.assertFalse(node.has_requested_sections("[Verse 1]\nLine\n[chorus]\nHook", required))
        self.assertTrue(node.has_requested_sections(
            "[Verse 1]\nLine\n[chorus]\nHook\n[bridge]\nTurn\n[outro]\nEnd", required))
        plan = node.concrete_section_plan(
            "Use [verse] [chorus] [bridge] [outro]. Two verses and two choruses.")
        self.assertIn("exactly 23 sung lines", plan)
        self.assertIn("[chorus] repeating the same 4-line hook", plan)

    def test_duration_is_read_from_connected_render_not_manual_widgets(self):
        self.assertNotIn("song_target_seconds", node.SparkStudioChat.INPUT_TYPES()["optional"])
        self.assertNotIn("song_tempo_bpm", node.SparkStudioChat.INPUT_TYPES()["optional"])
        self.assertEqual(node.song_context(song_graph(), "18"),
                         (180, "Slow R&B ballad, male vocal"))
        self.assertEqual(node.lyric_budget(180), (22, 30, 124, 183))
        self.assertGreater(node.lyric_budget(240)[0], node.lyric_budget(180)[0])
        self.assertIsNone(node.lyric_budget(0))
        self.assertTrue(node.lyrics_fit_duration(26, 155, 180, node.lyric_budget(180)))
        self.assertTrue(node.lyrics_fit_duration(30, 185, 180, node.lyric_budget(180)))
        self.assertFalse(node.lyrics_fit_duration(25, 89, 180, node.lyric_budget(180)))
        self.assertFalse(node.lyrics_fit_duration(46, 220, 180, node.lyric_budget(180)))

    def test_unrelated_render_does_not_budget_general_chat(self):
        graph = song_graph()
        graph["2"]["inputs"]["lyrics"] = "Unrelated words"
        self.assertIsNone(node.song_context(graph, "18"))

    def test_overlong_song_is_revised_before_output(self):
        long_text = "[verse]\n" + "\n".join(["Too many words to sing"] * 40)
        polished_text = "[verse]\n" + "\n".join(
            ["I remember Leo beside me every night"] * 22)
        opener = SequenceOpener([long_text, polished_text])
        with patch.object(node, "build_opener", return_value=opener):
            result = node.SparkStudioChat().generate(
                "Write a song for Leo", "http://spark:8888/v1", "test-model",
                512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18",
            )[0]
        self.assertEqual(result, polished_text)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(opener.requests[0]["max_tokens"], 512)
        self.assertIn("absolute limits: 22-30 lines", opener.requests[0]["messages"][-1]["content"])
        self.assertIn("exactly 26 substantial", opener.requests[0]["messages"][-1]["content"])
        self.assertIn("NON-LYRIC PRODUCTION METADATA", opener.requests[0]["messages"][-1]["content"])
        self.assertIn("demanding professional songwriter", opener.requests[1]["messages"][-1]["content"])
        self.assertIn("never sing terms", opener.requests[1]["messages"][-1]["content"])

    def test_overlong_song_errors_instead_of_silent_truncation(self):
        long_text = "[verse]\n" + "\n".join(["Too many words to sing"] * 40)
        opener = SequenceOpener([long_text] * 3)
        with patch.object(node, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "could not fit lyrics"):
                node.SparkStudioChat().generate(
                    "Write a song", "http://spark:8888/v1", "test-model",
                    512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18",
                )


    def test_models_url_sits_beside_the_chat_url(self):
        self.assertEqual(node.models_url("http://spark:8000/v1"),
                         "http://spark:8000/v1/models")
        self.assertEqual(node.models_url("http://spark:7860/api/engine/v1"),
                         "http://spark:7860/api/engine/v1/models")
        self.assertEqual(node.models_url("http://spark:8000"),
                         "http://spark:8000/v1/models")

    def test_blank_model_is_discovered_from_the_endpoint(self):
        """A plain vLLM or SGLang server rejects an unknown model id."""
        class DiscoveryOpener:
            def __init__(self):
                self.urls = []
                self.sent = {}

            def open(self, request, timeout):
                self.urls.append(request.full_url)
                if request.full_url.endswith("/models"):
                    body = {"data": [{"id": "served-model-a"}, {"id": "served-model-b"}]}
                else:
                    self.sent = json.loads(request.data)
                    body = {"choices": [{"message": {"content": "ok"}}]}
                return io.BytesIO(json.dumps(body).encode())

        opener = DiscoveryOpener()
        with patch.object(node, "build_opener", return_value=opener):
            text = node.SparkStudioChat().generate("Hi", "http://spark:8000/v1",
                                                   "", 64, 0.7, 0.95)[0]
        self.assertEqual(text, "ok")
        self.assertEqual(opener.urls[0], "http://spark:8000/v1/models")
        self.assertEqual(opener.urls[1], "http://spark:8000/v1/chat/completions")
        self.assertEqual(opener.sent["model"], "served-model-a")

    def test_defaults_ship_no_private_address_or_machine_specific_model(self):
        required = node.SparkStudioChat.INPUT_TYPES()["required"]
        self.assertEqual(required["model"][1]["default"], "")
        base_url = required["base_url"][1]["default"]
        self.assertIn("localhost", base_url)
        source = (Path(node.__file__).read_text(encoding="utf-8")
                  if getattr(node, "__file__", None) else NODE_PATH.read_text(encoding="utf-8"))
        self.assertNotRegex(source, r"[0-9]{1,3}(?:[.][0-9]{1,3}){3}")

    def test_render_duration_is_read_under_either_input_name(self):
        for name in ("max_duration", "target_duration"):
            graph = song_graph()
            graph["3"]["inputs"] = {"composition": ["2", 0], name: 150}
            with self.subTest(name=name):
                self.assertEqual(node.song_context(graph, "18")[0], 150)


if __name__ == "__main__":
    unittest.main()
