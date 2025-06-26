document.addEventListener("DOMContentLoaded", function () {
  const modelSelect = document.getElementById("model-select");

  modelSelect.addEventListener("change", function () {
    const selectedModel = modelSelect.value;
    console.log("Selected model:", selectedModel); // Debugging

    if (selectedModel === "SARIMA") {
      window.location.href = "/forecast";
    } else if (selectedModel === "LSTM") {
      window.location.href = "/lstm";
    }
  });
});
