const disabledLinks = document.querySelectorAll("a[aria-disabled='true']");

disabledLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
  });
});
