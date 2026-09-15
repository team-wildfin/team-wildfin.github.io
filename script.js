const disabledLinks = document.querySelectorAll("a[aria-disabled='true']");

disabledLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
  });
});

const copyCitationButton = document.querySelector("[data-copy-citation]");
const citation = document.querySelector("#bibtex-citation");

if (copyCitationButton && citation) {
  copyCitationButton.addEventListener("click", async () => {
    const citationText = citation.textContent.trim();

    try {
      await navigator.clipboard.writeText(citationText);
    } catch {
      const textArea = document.createElement("textarea");
      textArea.value = citationText;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
    }

    copyCitationButton.classList.add("is-copied");
    copyCitationButton.setAttribute("aria-label", "BibTeX copied");
    copyCitationButton.setAttribute("title", "Copied!");

    window.setTimeout(() => {
      copyCitationButton.classList.remove("is-copied");
      copyCitationButton.setAttribute("aria-label", "Copy BibTeX");
      copyCitationButton.setAttribute("title", "Copy BibTeX");
    }, 1800);
  });
}
