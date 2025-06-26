document.addEventListener("DOMContentLoaded", function () {
  const modelSelect = document.getElementById("model-select");

  modelSelect.addEventListener("change", function () {
    const selectedModel = modelSelect.value;

    if (selectedModel === "ARIMA") {
      window.location.href = "/arima";
    } else if (selectedModel === "LSTM") {
      window.location.href = "/lstm";
    }
  });
});
