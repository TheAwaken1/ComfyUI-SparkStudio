import importlib.util
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

NEWLINE = chr(10)


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
                512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18", fit_to_duration=True,
            )[0]
        self.assertEqual(result, polished_text)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(opener.requests[0]["max_tokens"], 512)
        self.assertIn("absolute limits: 22-30 lines", opener.requests[0]["messages"][-1]["content"])
        self.assertIn("exactly 26 substantial", opener.requests[0]["messages"][-1]["content"])
        self.assertIn("NON-LYRIC PRODUCTION METADATA", opener.requests[0]["messages"][-1]["content"])
        self.assertIn("demanding professional songwriter", opener.requests[1]["messages"][-1]["content"])
        self.assertIn("never sing terms", opener.requests[1]["messages"][-1]["content"])

    def test_overlong_song_returns_the_closest_draft_instead_of_failing(self):
        """A length target is a preference; failing would discard the render."""
        long_text = "[verse]" + NEWLINE + NEWLINE.join(["Too many words to sing"] * 40)
        opener = SequenceOpener([long_text] * 3)
        with patch.object(node, "build_opener", return_value=opener):
            text = node.SparkStudioChat().generate(
                "Write a song", "http://spark:8888/v1", "test-model",
                512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18", fit_to_duration=True,
            )[0]
        self.assertIn("Too many words to sing", text)
        self.assertEqual(len(opener.requests), 3, "it still tries three times")

    def test_closest_draft_is_the_one_returned(self):
        drafts = [
            "[verse]" + NEWLINE + NEWLINE.join(["Way too many words here"] * 40),
            "[verse]" + NEWLINE + NEWLINE.join(["Closer to the target now"] * 30),
            "[verse]" + NEWLINE + NEWLINE.join(["Far too many words again"] * 60),
        ]
        opener = SequenceOpener(drafts)
        with patch.object(node, "build_opener", return_value=opener):
            text = node.SparkStudioChat().generate(
                "Write a song", "http://spark:8888/v1", "test-model",
                512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18", fit_to_duration=True,
            )[0]
        self.assertIn("Closer to the target now", text)

    def test_budget_miss_prefers_complete_section_sets(self):
        budget = (20, 28, 117, 172)
        self.assertLess(node.budget_miss(24, 145, budget, True),
                        node.budget_miss(24, 145, budget, False))
        self.assertLess(node.budget_miss(28, 180, budget, True),
                        node.budget_miss(40, 300, budget, True))


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
        # Loopback and the documentation range are fine; a real host is not.
        for allowed in ("127.0.0.1", "0.0.0.0", "192.0.2."):
            source = source.replace(allowed, "")
        self.assertNotRegex(source, r"[0-9]{1,3}(?:[.][0-9]{1,3}){3}")

    def test_render_duration_is_read_under_either_input_name(self):
        for name in ("max_duration", "target_duration"):
            graph = song_graph()
            graph["3"]["inputs"] = {"composition": ["2", 0], name: 150}
            with self.subTest(name=name):
                self.assertEqual(node.song_context(graph, "18")[0], 150)


    def test_unresolvable_host_says_to_replace_the_example_address(self):
        """Pasting the README's placeholder host is the most common mistake."""
        from urllib.error import URLError
        import socket

        error = URLError(socket.gaierror(11001, "getaddrinfo failed"))
        message = str(node._unreachable("http://YOUR-SPARK-ADDRESS:7860/api/engine/v1/models", error))
        self.assertIn("YOUR-SPARK-ADDRESS", message)
        self.assertIn("does not exist on this network", message)
        self.assertIn("Tailscale", message)

    def test_refused_and_timeout_get_their_own_guidance(self):
        from urllib.error import URLError

        refused = str(node._unreachable("http://192.0.2.5:7860/v1/models",
                                        URLError(ConnectionRefusedError("Connection refused"))))
        self.assertIn("Nothing is listening", refused)
        timed = str(node._unreachable("http://192.0.2.5:7860/v1/models",
                                      URLError(TimeoutError("timed out"))))
        self.assertIn("firewall", timed)


    def _chat_graph(self, song_node):
        graph = {"18": {"class_type": "SparkStudioChat", "inputs": {"prompt": "Write a song"}}}
        graph.update(song_node)
        return graph

    def test_sampler_pack_reads_length_from_semantic_token_budget(self):
        """The ScryptHunter YuE2 pack has no duration input, only tokens."""
        graph = self._chat_graph({"5": {"class_type": "YuE2Sampler", "inputs": {
            "lyrics": ["18", 0], "style": "Slow ballad", "semantic_max_tokens": 9000}}})
        seconds, style = node.song_context(graph, "18")
        self.assertEqual(seconds, 360)
        self.assertEqual(style, "Slow ballad")

    def test_length_is_found_on_a_settings_node_feeding_the_sampler(self):
        graph = self._chat_graph({
            "5": {"class_type": "YuE2Sampler", "inputs": {
                "lyrics": ["18", 0], "style": "Trap", "semantic_max_tokens": ["6", 0]}},
            "6": {"class_type": "YuE2SamplingSettings", "inputs": {"semantic_max_tokens": 4500}},
        })
        self.assertEqual(node.song_context(graph, "18")[0], 180)

    def test_song_mode_without_any_length_control(self):
        """Some packs expose no length at all; song guidance still applies."""
        graph = self._chat_graph({"5": {"class_type": "SomeOtherYuE2", "inputs": {
            "lyrics": ["18", 0], "style": "Dream pop"}}})
        seconds, style = node.song_context(graph, "18")
        self.assertIsNone(seconds)
        self.assertEqual(style, "Dream pop")

    def test_song_mode_without_length_still_sends_style_guidance(self):
        graph = self._chat_graph({"5": {"class_type": "YuE2Plan", "inputs": {
            "lyrics": ["18", 0], "style": "Kalimba and rain"}}})
        opener = FakeOpener()
        with patch.object(node, "build_opener", return_value=opener):
            node.SparkStudioChat().generate("Write a song", "http://spark:8888/v1",
                                            "test-model", 512, 0.7, 0.95,
                                            prompt_graph=graph, unique_id="18")
        sent = json.loads(opener.request.data)["messages"][-1]["content"]
        self.assertIn("release-quality song lyric", sent)
        self.assertIn("Kalimba and rain", sent)
        self.assertIn("NEVER", sent, "production terms must not be sung")

    def test_ace_step_style_node_is_recognised(self):
        graph = self._chat_graph({"5": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "lyrics": ["18", 0], "tags": "indie rock", "duration": 120}}})
        self.assertEqual(node.song_context(graph, "18"), (120, "indie rock"))

    def test_lrc_and_utility_nodes_are_not_mistaken_for_song_nodes(self):
        """These take lyrics and a number, but they render nothing."""
        for class_type, inputs in (
            ("SongLyricsToLRC", {"lyrics": ["18", 0], "seconds": 180, "offset_seconds": 0}),
            ("MiniMaxLyricsToLRC", {"lyrics": ["18", 0], "seconds": 180}),
            ("SaveLyricsFileNode", {"lyrics": ["18", 0]}),
        ):
            with self.subTest(class_type=class_type):
                graph = self._chat_graph({"5": {"class_type": class_type, "inputs": inputs}})
                self.assertIsNone(node.song_context(graph, "18"))

    def test_lyrics_from_a_different_source_do_not_trigger_song_mode(self):
        graph = {
            "18": {"class_type": "SparkStudioChat", "inputs": {"prompt": "Describe the art"}},
            "9": {"class_type": "OtherText", "inputs": {"text": "hand written"}},
            "5": {"class_type": "YuE2Sampler", "inputs": {
                "lyrics": ["9", 0], "style": "Folk", "semantic_max_tokens": 4500}},
        }
        self.assertIsNone(node.song_context(graph, "18"))

    def test_shortest_length_wins_when_several_are_present(self):
        graph = self._chat_graph({
            "2": {"class_type": "FL_YuE2_Plan", "inputs": {"lyrics": ["18", 0], "style": "Rock"}},
            "3": {"class_type": "FL_YuE2_Render", "inputs": {
                "composition": ["2", 0], "max_duration": 240}},
            "4": {"class_type": "FL_YuE2_Render", "inputs": {
                "composition": ["2", 0], "max_duration": 150}},
        })
        self.assertEqual(node.song_context(graph, "18")[0], 150)


    def test_address_can_come_from_the_environment(self):
        """Typing a private address saves it into the workflow file."""
        with patch.dict(os.environ, {"SPARK_STUDIO_BASE_URL": "http://10.1.2.3:7860/v1"}):
            self.assertEqual(node.resolve_base_url(""), "http://10.1.2.3:7860/v1")
            self.assertEqual(node.resolve_base_url("env"), "http://10.1.2.3:7860/v1")
            self.assertEqual(node.resolve_base_url("http://typed:8000/v1"), "http://typed:8000/v1")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                node.resolve_base_url("")
            self.assertIn("SPARK_STUDIO_BASE_URL", str(caught.exception))

    def test_environment_address_is_masked_in_messages(self):
        """Errors must not reveal the address during a screen recording."""
        from urllib.error import URLError
        import socket

        with patch.dict(os.environ, {"SPARK_STUDIO_BASE_URL": "http://10.1.2.3:7860/v1"}):
            message = str(node._unreachable("http://10.1.2.3:7860/v1/models",
                                            URLError(socket.gaierror(11001, "getaddrinfo failed"))))
        self.assertNotIn("10.1.2.3", message)
        self.assertIn("<hidden>", message)

    def test_served_models_lists_every_id(self):
        class Listing:
            def open(self, request, timeout):
                return io.BytesIO(json.dumps(
                    {"data": [{"id": "one"}, {"id": "two"}, {"nope": 1}]}).encode())

        with patch.object(node, "build_opener", return_value=Listing()):
            self.assertEqual(node.served_models("http://spark:8000/v1"), ["one", "two"])


    def test_hide_address_is_a_ui_toggle_that_does_not_change_the_request(self):
        """Hiding is for screen recordings; it must not alter behaviour."""
        self.assertIn("hide_address", node.SparkStudioChat.INPUT_TYPES()["optional"])
        sent = []
        for hidden in (False, True):
            opener = FakeOpener()
            with patch.object(node, "build_opener", return_value=opener):
                node.SparkStudioChat().generate("Hello", "http://spark:8888/v1",
                                                "test-model", 64, 0.7, 0.95,
                                                hide_address=hidden)
            sent.append(json.loads(opener.request.data))
        self.assertEqual(sent[0], sent[1])


    def test_hidden_address_round_trips_through_the_private_store(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "sparkstudio_address.json")
            with patch.object(node, "private_address_file", lambda: target):
                self.assertEqual(node.load_private_address(), "")
                node.save_private_address("http://10.9.8.7:7860/api/engine/v1")
                self.assertEqual(node.load_private_address(),
                                 "http://10.9.8.7:7860/api/engine/v1")
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(node.resolve_base_url(""),
                                     "http://10.9.8.7:7860/api/engine/v1")
                    # A masked message must not leak the stored host either.
                    self.assertNotIn("10.9.8.7",
                                     node.hide_private_host("http://10.9.8.7:7860/v1/models"))
                node.save_private_address("")
                self.assertEqual(node.load_private_address(), "")

    def test_private_store_rejects_a_nonsense_address(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "sparkstudio_address.json")
            with patch.object(node, "private_address_file", lambda: target):
                with self.assertRaises(ValueError):
                    node.save_private_address("not-a-url")

    def test_typed_address_beats_the_store(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "sparkstudio_address.json")
            with patch.object(node, "private_address_file", lambda: target):
                node.save_private_address("http://10.9.8.7:7860/v1")
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(node.resolve_base_url("http://typed:8000/v1"),
                                     "http://typed:8000/v1")


    def test_length_cap_is_off_unless_asked_for(self):
        """The cap trims hard, and a render length is an upper bound."""
        self.assertFalse(
            node.SparkStudioChat.INPUT_TYPES()["optional"]["fit_to_duration"][1]["default"])
        long_text = "[verse]" + NEWLINE + NEWLINE.join(["Plenty of words to sing here"] * 40)
        opener = SequenceOpener([long_text])
        with patch.object(node, "build_opener", return_value=opener):
            out = node.SparkStudioChat().generate(
                "Write a song", "http://spark:8888/v1", "test-model",
                512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18",
            )[0]
        self.assertEqual(len(opener.requests), 1, "no revision pass without the cap")
        self.assertEqual(out.count("Plenty of words to sing here"), 40, "nothing trimmed")

    def test_style_guidance_still_applies_without_the_cap(self):
        opener = SequenceOpener(["[verse]" + NEWLINE + "A line"])
        with patch.object(node, "build_opener", return_value=opener):
            node.SparkStudioChat().generate(
                "Write a song", "http://spark:8888/v1", "test-model",
                512, 0.7, 0.95, prompt_graph=song_graph(), unique_id="18",
            )
        sent = opener.requests[0]["messages"][-1]["content"]
        self.assertIn("release-quality song", sent)
        self.assertIn("Slow R&B ballad", sent)
        self.assertNotIn("absolute limits", sent, "no numeric target without the cap")


if __name__ == "__main__":
    unittest.main()
