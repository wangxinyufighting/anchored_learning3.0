// MathJax v3 configuration for use with `pymdownx.arithmatex: generic: true`.
// arithmatex emits `<span class="arithmatex">\(...\)</span>` (inline) or
// `<div class="arithmatex">\[...\]</div>` (display); MathJax picks them up.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
