(function () {
  var copyResetDelay = 1600;

  function normalizeCopyText(text) {
    return text
      .replace(/\u00a0/g, " ")
      .split("\n")
      .map(function (line) {
        return line.trim();
      })
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function getPageCopyText(article) {
    var excludedNodes;
    var originalDisplays = [];
    var text;

    if (!article) {
      return "";
    }

    if (typeof article.querySelectorAll !== "function") {
      return normalizeCopyText(article.innerText || article.textContent || "");
    }

    excludedNodes = article.querySelectorAll(".areno-copy-page, .headerlink, script, style");

    Array.prototype.forEach.call(excludedNodes, function (node) {
      originalDisplays.push({
        node: node,
        display: node.style.display
      });
      node.style.display = "none";
    });

    try {
      text = article.innerText || article.textContent || "";
    } finally {
      Array.prototype.forEach.call(originalDisplays, function (item) {
        item.node.style.display = item.display;
      });
    }

    return normalizeCopyText(text);
  }

  function setCopyPageButtonLabel(button, label) {
    var labelNode = button.querySelector(".areno-copy-page-label");

    if (labelNode) {
      labelNode.textContent = label;
    }
  }

  function resetCopyPageButton(button) {
    window.clearTimeout(button._arenoCopyTimer);
    button.dataset.arenoCopyState = "idle";
    setCopyPageButtonLabel(button, "Copy page");
  }

  function showCopyPageButtonState(button, state, label) {
    window.clearTimeout(button._arenoCopyTimer);
    button.dataset.arenoCopyState = state;
    setCopyPageButtonLabel(button, label);
    button._arenoCopyTimer = window.setTimeout(function () {
      resetCopyPageButton(button);
    }, copyResetDelay);
  }

  function copyTextWithFallback(text) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      return navigator.clipboard.writeText(text).catch(function () {
        return copyTextWithTextarea(text);
      });
    }

    return copyTextWithTextarea(text);
  }

  function copyTextWithTextarea(text) {
    var textarea = document.createElement("textarea");

    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();

    try {
      if (!document.execCommand("copy")) {
        return Promise.reject(new Error("Copy command failed"));
      }

      return Promise.resolve();
    } catch (error) {
      return Promise.reject(error);
    } finally {
      document.body.removeChild(textarea);
    }
  }

  function createCopyPageButton(article) {
    var button = document.createElement("button");

    button.type = "button";
    button.className = "areno-copy-page";
    button.setAttribute("aria-label", "Copy this page");
    button.setAttribute("title", "Copy page");
    button.innerHTML = [
      '<svg aria-hidden="true" focusable="false" width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">',
      '<path d="M14.25 5.25H7.25C6.14543 5.25 5.25 6.14543 5.25 7.25V14.25C5.25 15.3546 6.14543 16.25 7.25 16.25H14.25C15.3546 16.25 16.25 15.3546 16.25 14.25V7.25C16.25 6.14543 15.3546 5.25 14.25 5.25Z"></path>',
      '<path d="M2.80103 11.998L1.77203 5.07397C1.61003 3.98097 2.36403 2.96397 3.45603 2.80197L10.38 1.77297C11.313 1.63397 12.19 2.16297 12.528 3.00097"></path>',
      "</svg>",
      '<span class="areno-copy-page-label">Copy page</span>'
    ].join("");

    button.addEventListener("click", function () {
      var text = getPageCopyText(article);

      if (!text) {
        showCopyPageButtonState(button, "error", "Nothing to copy");
        return;
      }

      copyTextWithFallback(text)
        .then(function () {
          showCopyPageButtonState(button, "copied", "Copied");
        })
        .catch(function () {
          showCopyPageButtonState(button, "error", "Copy failed");
        });
    });

    return button;
  }

  function setupCopyPageButton() {
    var article = document.querySelector(".yue");
    var header;
    var heading;
    var actions;

    if (
      !article ||
      article.classList.contains("landing-page") ||
      article.querySelector(".areno-copy-page")
    ) {
      return;
    }

    heading = article.querySelector("h1");

    if (!heading) {
      return;
    }

    if (!heading.parentNode) {
      return;
    }

    header = document.createElement("div");
    header.className = "areno-page-header";
    heading.parentNode.insertBefore(header, heading);
    header.appendChild(heading);

    actions = document.createElement("div");
    actions.className = "areno-page-actions";
    actions.appendChild(createCopyPageButton(article));
    header.appendChild(actions);
  }

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

  window.__arenoDocs = window.__arenoDocs || {};
  window.__arenoDocs.getPageCopyText = getPageCopyText;
  window.__arenoDocs.setupCopyPageButton = setupCopyPageButton;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      setupSidebarSections();
      setupCopyPageButton();
    });
  } else {
    setupSidebarSections();
    setupCopyPageButton();
  }
})();
