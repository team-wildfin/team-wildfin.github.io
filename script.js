const tabGroups = document.querySelectorAll("[data-tabs]");
const disabledLinks = document.querySelectorAll("a[aria-disabled='true']");

disabledLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
  });
});


tabGroups.forEach((group) => {
  const tabs = Array.from(group.querySelectorAll("[role='tab']"));
  const panels = Array.from(group.querySelectorAll("[role='tabpanel']"));

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((item) => {
        const isSelected = item === tab;
        item.classList.toggle("is-active", isSelected);
        item.setAttribute("aria-selected", String(isSelected));
      });

      panels.forEach((panel) => {
        const isSelected = panel.id === tab.getAttribute("aria-controls");
        panel.classList.toggle("is-active", isSelected);
        panel.hidden = !isSelected;
      });
    });
  });
});
