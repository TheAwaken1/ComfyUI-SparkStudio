import { app } from "../../scripts/app.js";

// Fills the model widget with whatever the server is actually running, so the
// id never has to be typed or remembered.

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

app.registerExtension({
  name: "SparkStudio.ModelPicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SparkStudioChat") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const created = onCreated ? onCreated.apply(this, arguments) : undefined;
      const node = this;
      const modelWidget = findWidget(node, "model");
      const urlWidget = findWidget(node, "base_url");
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
          // Blank means auto-detect at run time; show what that will pick.
          if (!String(modelWidget.value || "").trim()) modelWidget.value = known[0];
        } else {
          button.name = data.error ? "Models: server unreachable" : "Models: none reported";
        }
        node.setDirtyCanvas(true, true);
        return { models: known, error: data.error };
      }

      const previousCallback = urlWidget.callback;
      urlWidget.callback = function () {
        const result = previousCallback ? previousCallback.apply(this, arguments) : undefined;
        refresh();
        return result;
      };

      setTimeout(refresh, 400);
      return created;
    };
  },
});
