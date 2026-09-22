// Eğitilmiş model parametrelerini (ağırlıklar, bias, standardizasyon) yükler.
let MODEL = null;

async function loadModel() {
  const res = await fetch("model.json");
  MODEL = await res.json();
  return MODEL;
}
