// Initialize mermaid for blocks rendered as <div class="mermaid">...</div>
// (output produced by `pymdownx.superfences.fence_div_format` in mkdocs.yml).
//
// We deliberately use `startOnLoad: false` + an explicit `mermaid.run()` so
// rendering still works when the script tag is loaded after DOMContentLoaded
// (which happens with mkdocs `extra_javascript` injecting at the end of body).
function kdflowInitMermaid() {
  if (typeof mermaid === "undefined") {
    return;
  }
  mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    securityLevel: "loose",
  });
  mermaid.run({ querySelector: ".mermaid" });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", kdflowInitMermaid);
} else {
  kdflowInitMermaid();
}
