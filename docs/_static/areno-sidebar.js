(function () {
  function setupSidebarSections() {
    var toc = document.querySelector(".globaltoc");

    if (!toc || toc.dataset.arenoCollapsible === "ready") {
      return;
    }

    toc.dataset.arenoCollapsible = "ready";

    Array.prototype.forEach.call(toc.children, function (caption, index) {
      if (!caption.matches("p.caption")) {
        return;
      }

      var list = caption.nextElementSibling;

      if (!list || !list.matches("ul") || caption.querySelector(".areno-sidebar-toggle")) {
        return;
      }

      if (!list.id) {
        list.id = "areno-sidebar-section-" + index;
      }

      caption.classList.add("areno-sidebar-caption");
      list.classList.add("areno-sidebar-section-list");

      var button = document.createElement("button");
      button.type = "button";
      button.className = "areno-sidebar-toggle";
      button.setAttribute("aria-controls", list.id);
      button.setAttribute("aria-expanded", "true");

      while (caption.firstChild) {
        button.appendChild(caption.firstChild);
      }

      var caret = document.createElement("span");
      caret.className = "areno-sidebar-caret";
      caret.setAttribute("aria-hidden", "true");
      button.appendChild(caret);
      caption.appendChild(button);

      button.addEventListener("click", function () {
        var expanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", expanded ? "false" : "true");
        list.hidden = expanded;
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupSidebarSections);
  } else {
    setupSidebarSections();
  }
})();
