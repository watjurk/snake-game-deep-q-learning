(() => {
  const control = {};
  window.mlTools.control = control;

  control.setValue = async (key, value) => {
    const response = await fetch(`/mlTools/control/set_value?key=${key}`, {
      method: "POST",
      body: value,
    });
    await response.text()
  };
})();
