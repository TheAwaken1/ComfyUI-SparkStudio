# ComfyUI-SparkStudio

A single ComfyUI node that sends a prompt to a model server and returns a `STRING`.
It was built to offload writing to an NVIDIA DGX Spark while the local GPU renders
audio, images or video, but it talks plain OpenAI-compatible HTTP, so it works with
any server that speaks that protocol.

The node also has an optional `IMAGE` input, so a vision model can look at a picture
and write from it.

## Why

Running a large model on a DGX Spark and a diffusion or audio model on a desktop GPU
at the same time is awkward, because the text step and the render step want the same
machine. This node splits them. The Spark writes, the desktop renders, and the text
moves between them inside one graph with no copy and paste.

## Install

Through ComfyUI-Manager, search for `ComfyUI-SparkStudio`.

Or clone it:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TheAwaken1/ComfyUI-SparkStudio
```

Restart ComfyUI. No extra Python packages are needed.

## Quick start

1. Find your server's address. On the Spark itself that is `localhost`. From
   another machine it is the Spark's LAN IP or Tailscale address, such as
   `100.x.y.z`. `YOUR-SPARK-ADDRESS` below is a placeholder, not a real host.
2. Confirm it answers before wiring anything up:
   `curl http://YOUR-SPARK-ADDRESS:7860/api/engine/v1/models`
   You should get a JSON list of models. If that fails, the node will fail too.
3. Paste that same base URL into `base_url`. The node fills `model` in for you
   from whatever the server is running. Click the model button on the node to
   re-check or to pick a different one.

### Finding your address

There is no discovery, so you do tell the node where the server is. On the Spark
itself that is `localhost`. From another machine, run `hostname -I` on the Spark for
its LAN address, or `tailscale ip -4` for its tailnet address. Either works as long
as the desktop can reach it.

## Connecting

Set `base_url` to your server. Three forms are accepted:

| Form | Example | Use when |
|------|---------|----------|
| Spark Studio gateway | `http://YOUR-SPARK-ADDRESS:7860/api/engine/v1` | You run [Spark Studio](https://github.com/TheAwaken1/Spark-Studio) |
| Engine base | `http://YOUR-SPARK-ADDRESS:8000/v1` | vLLM, SGLang, llama.cpp, LM Studio, Ollama |
| Full chat URL | `http://YOUR-SPARK-ADDRESS:8000/v1/chat/completions` | Anything unusual |

The Spark Studio gateway is worth preferring if you have it, because that one URL
follows whichever engine and model is currently live. Swapping models does not break
your saved graph.

Leave `model` blank and the node asks the endpoint which models it serves and uses
the first one. Fill it in when a server hosts several and you want a specific one.

### Reaching a Spark from another machine

A model bound to `127.0.0.1` on the Spark is not reachable at `127.0.0.1` from your
desktop. Use one of:

- Tailscale, then point `base_url` at the Spark's tailnet address
- An SSH tunnel: `ssh -L 8000:127.0.0.1:8000 user@YOUR-SPARK-ADDRESS`
- Bind the server to the LAN, only on a network you trust

### Hiding your address

Two separate problems, two answers.

**To keep it off a screen recording**, switch on `hide_address` on the node. The
address field disappears from the node immediately. Switch it back off to edit.
This is the easy one and needs no setup.

**To keep it out of a workflow file you share**, the address has to live outside the
graph, because anything typed into a widget is saved into the JSON. Put it in an
environment variable named `SPARK_STUDIO_BASE_URL` and leave `base_url` empty.

The node then reads it at run time. It never enters the workflow, and it is masked
as `<hidden>` in error messages, so a failed connection mid-recording cannot leak it.

#### Setting the variable

In Pinokio, open the launcher's `start.js` and add it to the `env` block that is
already there:

```js
env: {
  TOKENIZERS_PARALLELISM: "false",
  SPARK_STUDIO_BASE_URL: "http://100.x.y.z:7860/api/engine/v1"
},
```

Save, stop the app, and start it again.

Launching ComfyUI yourself instead, set it in the same terminal first:

```bash
set SPARK_STUDIO_BASE_URL=http://100.x.y.z:7860/api/engine/v1
python main.py
```

On macOS or Linux use `export` in place of `set`. To make it permanent on Windows,
search for "Edit environment variables for your account" and add it there, then
restart ComfyUI.

Either way, confirm it worked: leave `base_url` empty and run the node. If the
variable is missing the node says so instead of failing silently.

### Authentication

If your endpoint requires a bearer token, set `SPARK_STUDIO_API_KEY` in ComfyUI's
environment. The node sends it as an `Authorization` header. Do not paste a token
into a widget, because widget values are saved inside workflow files.

Spark Studio itself has no authentication. Do not expose port 7860 to the internet.

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| prompt | STRING | The request |
| base_url | STRING | See the table above |
| model | STRING | Filled in from the server; blank auto-detects at run time |
| max_tokens | INT | Output limit, not song length |
| temperature | FLOAT | |
| top_p | FLOAT | |
| system_prompt | STRING | Optional |
| image | IMAGE | Optional, up to four frames, sent as JPEG |
| timeout_seconds | INT | Network timeout only |
| strip_thinking | BOOLEAN | Removes think tags from reasoning models |

Output is `generated_text`, a `STRING` you can wire anywhere.

## Songwriting mode

This is optional and turns itself on. If the node's text feeds a node that takes
lyrics plus a style or a length, the node treats the job as songwriting.

It is matched by shape, not by pack name, so it works across the YuE2 forks and
other song models rather than one specific pack. Verified against FL-YuE2, the
YuE2Sampler and YuE2Plan packs, ACE-Step, and MiniMax Music 3.

In song mode the node reads the style text from the song node and uses it to steer
mood and cadence, while forbidding production words such as instrument names, tempo
and artist references from being sung as lyrics. When the request asks for two
verses and two choruses, it writes each section in a separate call, refuses lines
copied between sections, and repeats the finished chorus exactly.

### Length

Length is optional, because packs express it three different ways.

| The song node exposes | What the node does |
|---|---|
| Seconds, such as `max_duration`, `target_duration` or `duration` | Sizes the lyric to that many seconds |
| A token budget, such as `semantic_max_tokens` | Converts it at 25 semantic tokens per second |
| Nothing at all | Applies style and structure guidance with no length target |

The length is also found when it sits on a renderer further down the chain, as in
FL-YuE2, or on a settings node feeding the sampler.

Note that these values are usually an upper bound rather than an exact length.
A `semantic_max_tokens` of 9000 reads as six minutes. Lower it to target a shorter
song, for example 4500 for roughly three minutes.

A prompt not connected to a song node behaves as an ordinary chat node.

## Programmatic use

The node is a thin client. The same request works from anything.

```bash
curl http://YOUR-SPARK-ADDRESS:7860/api/engine/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"","messages":[{"role":"user","content":"Write one line."}]}'
```

```python
import json, urllib.request
body = {"model": "", "messages": [{"role": "user", "content": "Write one line."}]}
req = urllib.request.Request(
    "http://YOUR-SPARK-ADDRESS:7860/api/engine/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

```javascript
const r = await fetch("http://YOUR-SPARK-ADDRESS:7860/api/engine/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model: "", messages: [{ role: "user", content: "Write one line." }] }),
});
console.log((await r.json()).choices[0].message.content);
```

Set `model` to a served id when the endpoint does not accept a blank one.

## Tests

```bash
python -m unittest discover -s tests
```

## License

MIT
