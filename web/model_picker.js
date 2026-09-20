import { app } from "../../scripts/app.js";

// Two small conveniences for the Spark Chat node:
//   1. fill the model field from whatever the server is running
//   2. hide the address on the node while screen recording

function findWidget(node, name) {
  return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

async function fetchModels(baseUrl) {
  try {
    const query = encodeURIComponent(baseUrl == null ? "" : baseUrl);
    const response = await fetch(`/sparkstudio/models?base_url=${query}`);
    if (!response.ok) return { models: [], error: `HTTP ${response.status}` };
    return await response.json();
  } catch (error) {
    return { models: [], error: String(error) };
  }
}

function chain(widget, extra) {
  const previous = widget.callback;
  widget.callback = function () {
    const result = previous ? previous.apply(this, arguments) : undefined;
    try {
      extra();
    } catch (error) {
      console.error("[SparkStudio]", error);
    }
    return result;
  };
}

app.registerExtension({
  name: "SparkStudio.NodeHelpers",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SparkStudioChat") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const created = onCreated ? onCreated.apply(this, arguments) : undefined;
      const node = this;
      const modelWidget = findWidget(node, "model");
      const urlWidget = findWidget(node, "base_url");
      const hideWidget = findWidget(node, "hide_address");
      if (!modelWidget || !urlWidget) return created;

      let known = [];
      const button = node.addWidget("button", "Models: not checked", null, async (v, w, n, pos, event) => {
        const data = await refresh();
        if (!data.models.length) return;
        new LiteGraph.ContextMenu(data.models, {
          event,
          title: "Served models",
          callback: (picked) => {
            modelWidget.value = picked;
            node.setDirtyCanvas(true, true);
          },
        });
      });
      button.serialize = false;

      async function refresh() {
        const data = await fetchModels(urlWidget.value);
        known = data.models || [];
        if (known.length) {
          button.name = known.length === 1 ? `Model: ${known[0]}` : `Models: ${known.length} available`;
          if (!String(modelWidget.value || "").trim()) modelWidget.value = known[0];
        } else {
          button.name = data.error ? "Models: server unreachable" : "Models: none reported";
        }
        node.setDirtyCanvas(true, true);
        return { models: known, error: data.error };
      }

      function applyPrivacy() {
        const on = !!(hideWidget && hideWidget.value);
        urlWidget.options = urlWidget.options || {};
        urlWidget.options.hidden = on;
        if (node.setSize && node.computeSize) node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
      }

      chain(urlWidget, () => refresh());
      if (hideWidget) chain(hideWidget, applyPrivacy);

      // Configure runs when a saved workflow loads, so re-apply then too.
      const onConfigure = node.onConfigure;
      node.onConfigure = function () {
        const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        setTimeout(applyPrivacy, 0);
        return result;
      };

      setTimeout(() => {
        applyPrivacy();
        refresh();
      }, 400);
      return created;
    };
  },
});
