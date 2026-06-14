"use strict";

// Mobile nav toggle
const navToggle = document.getElementById("navToggle");
const nav = document.getElementById("nav");
if (navToggle && nav) {
  navToggle.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });
  // Close on nav link click
  nav.querySelectorAll("a").forEach((a) => {
    a.addEventListener("click", () => {
      nav.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  });
}

// Inline code-copy buttons (data-target points to a <pre> id)
document.querySelectorAll(".code-copy").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.target);
    if (!target) return;
    navigator.clipboard.writeText(target.textContent.trim()).then(() => {
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.innerHTML = copyIconSVG();
      }, 1800);
    });
  });
});

// BibTeX copy button
const copyBibtex = document.getElementById("copyBibtex");
const bibtexBlock = document.getElementById("bibtexBlock");
const copyStatus = document.getElementById("copyStatus");
if (copyBibtex && bibtexBlock) {
  copyBibtex.addEventListener("click", () => {
    navigator.clipboard.writeText(bibtexBlock.textContent.trim()).then(() => {
      if (copyStatus) copyStatus.textContent = "Copied to clipboard!";
      setTimeout(() => { if (copyStatus) copyStatus.textContent = ""; }, 2000);
    });
  });
}

function copyIconSVG() {
  return `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <path d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1H2z"/>
  </svg>`;
}
