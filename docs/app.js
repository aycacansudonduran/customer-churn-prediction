const LABELS = {
  "Yas": "Yaş",
  "Abonelik Süresi (Ay)": "Abonelik Süresi (Ay)",
  "Destek Çağrısı Sayısı": "Destek Çağrısı Sayısı",
  "Aylık Ücret": "Aylık Ücret (TL)",
  "Toplam Ücret": "Toplam Ücret (TL)",
  "Cinsiyet": "Cinsiyet",
  "Partner": "Partneri Var mı?",
  "Bakmakla Yükümlü": "Bakmakla Yükümlü Kişi Var mı?",
  "Sözleşme Tipi": "Sözleşme Tipi",
  "İnternet Hizmeti": "İnternet Hizmeti",
  "Teknik Destek": "Teknik Destek",
  "Online Güvenlik": "Online Güvenlik",
  "Kağıtsız Fatura": "Kağıtsız Fatura",
  "Ödeme Yöntemi": "Ödeme Yöntemi",
};

const NUMERIC_CONFIG = {
  "Yas": { min: 18, max: 85, step: 1, value: 40 },
  "Abonelik Süresi (Ay)": { min: 0, max: 72, step: 1, value: 12 },
  "Destek Çağrısı Sayısı": { min: 0, max: 10, step: 1, value: 1 },
  "Aylık Ücret": { min: 18, max: 140, step: 1, value: 60 },
  "Toplam Ücret": { min: 0, max: 12000, step: 10, value: 720 },
};

function prettyFeatureName(name) {
  if (LABELS[name]) return LABELS[name];
  const idx = name.indexOf("_");
  if (idx > -1) {
    const base = name.slice(0, idx);
    const val = name.slice(idx + 1);
    return `${LABELS[base] || base}: ${val}`;
  }
  return name;
}

function buildForm(model) {
  const grid = document.getElementById("field-grid");
  grid.innerHTML = "";

  model.numeric_cols.forEach((col) => {
    const cfg = NUMERIC_CONFIG[col] || { min: 0, max: 1000, step: 1, value: 0 };
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = `
      <label for="f-${col}">${LABELS[col] || col}</label>
      <input type="number" id="f-${col}" name="${col}" min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${cfg.value}">
    `;
    grid.appendChild(field);
  });

  model.categorical_cols.forEach((col) => {
    const opts = model.categories[col].all_values;
    const field = document.createElement("div");
    field.className = "field";
    const options = opts.map((v) => `<option value="${v}">${v}</option>`).join("");
    field.innerHTML = `
      <label for="f-${col}">${LABELS[col] || col}</label>
      <select id="f-${col}" name="${col}">${options}</select>
    `;
    grid.appendChild(field);
  });

  // Abonelik süresi / aylık ücret değişince toplam ücreti otomatik güncelle
  const tenureInput = document.getElementById("f-Abonelik Süresi (Ay)");
  const chargeInput = document.getElementById("f-Aylık Ücret");
  const totalInput = document.getElementById("f-Toplam Ücret");
  const syncTotal = () => {
    const t = parseFloat(tenureInput.value) || 0;
    const m = parseFloat(chargeInput.value) || 0;
    totalInput.value = Math.round(t * m);
  };
  tenureInput.addEventListener("input", syncTotal);
  chargeInput.addEventListener("input", syncTotal);
}

function buildFeatureVector(model) {
  const values = {};
  model.numeric_cols.forEach((col) => {
    values[col] = parseFloat(document.getElementById(`f-${col}`).value) || 0;
  });
  model.categorical_cols.forEach((col) => {
    values[col] = document.getElementById(`f-${col}`).value;
  });

  return model.feature_order.map((featName) => {
    if (values.hasOwnProperty(featName)) return values[featName];
    const idx = featName.indexOf("_");
    const col = featName.slice(0, idx);
    const val = featName.slice(idx + 1);
    return values[col] === val ? 1 : 0;
  });
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predict(model, rawX) {
  const standardized = rawX.map((x, i) => (x - model.mean[i]) / model.std[i]);
  let z = model.bias;
  const contributions = [];
  standardized.forEach((xs, i) => {
    const contrib = xs * model.weights[i];
    z += contrib;
    contributions.push({ name: model.feature_order[i], contrib });
  });
  const proba = sigmoid(z);
  return { proba, contributions };
}

function renderGauge(proba) {
  const circumference = 251;
  const offset = circumference * (1 - proba);
  const fill = document.getElementById("gauge-fill");
  fill.style.strokeDashoffset = offset;

  let color = "var(--status-good)";
  if (proba >= 0.6) color = "var(--status-critical)";
  else if (proba >= 0.3) color = "var(--status-warning)";
  fill.style.stroke = color;

  const angle = Math.PI * (1 - proba);
  const cx = 100 + 74 * Math.cos(angle);
  const cy = 110 - 74 * Math.sin(angle);
  const needle = document.getElementById("gauge-needle");
  needle.setAttribute("cx", cx.toFixed(1));
  needle.setAttribute("cy", cy.toFixed(1));

  document.getElementById("proba-value").textContent = `%${Math.round(proba * 100)}`;
}

function renderRiskBadge(proba) {
  const badge = document.getElementById("risk-badge");
  badge.classList.remove("risk-low", "risk-mid", "risk-high");
  if (proba >= 0.6) {
    badge.textContent = "Yüksek Risk — Aksiyon Alınmalı";
    badge.classList.add("risk-high");
  } else if (proba >= 0.3) {
    badge.textContent = "Orta Risk — İzlenmeli";
    badge.classList.add("risk-mid");
  } else {
    badge.textContent = "Düşük Risk";
    badge.classList.add("risk-low");
  }
}

function renderFactors(contributions) {
  const sorted = [...contributions].sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib)).slice(0, 4);
  const container = document.getElementById("factor-items");
  container.innerHTML = sorted.map((f) => {
    const up = f.contrib > 0;
    return `
      <div class="factor-item">
        <span class="factor-name">${prettyFeatureName(f.name)}</span>
        <span class="factor-tag ${up ? "factor-up" : "factor-down"}">${up ? "Riski Artırıyor" : "Riski Azaltıyor"}</span>
      </div>
    `;
  }).join("");
}

async function init() {
  const model = await loadModel();
  document.getElementById("stat-accuracy").textContent = `%${Math.round(model.accuracy_on_full_data * 100)}`;
  buildForm(model);

  document.getElementById("predict-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const rawX = buildFeatureVector(model);
    const { proba, contributions } = predict(model, rawX);

    document.getElementById("result-empty").hidden = true;
    document.getElementById("result-content").hidden = false;
    renderGauge(proba);
    renderRiskBadge(proba);
    renderFactors(contributions);
  });
}

init();
