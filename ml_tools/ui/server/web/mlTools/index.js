(async () => {
  let mlToolsLoadedResolve;
  window.mlTools = {}
  window.mlTools.loaded = new Promise((resolve) => (mlToolsLoadedResolve = resolve));

  function loadScript(src) {
    return new Promise((resolve) => {
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      document.body.appendChild(script)
    });
  }

  await loadScript("/mlTools/control.js");

  mlToolsLoadedResolve();
})();
