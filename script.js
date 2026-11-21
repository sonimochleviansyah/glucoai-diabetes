const FIELDS = [
  {k:"Pregnancies", label:"Jumlah kehamilan", unit:"kali", ph:"1", ex:1, nomissing:true},
  {k:"Glucose", label:"Glukosa plasma", unit:"mg/dL", ph:"120", ex:148},
  {k:"BloodPressure", label:"Tekanan diastolik", unit:"mm Hg", ph:"70", ex:72},
  {k:"SkinThickness", label:"Lipatan kulit trisep", unit:"mm", ph:"25", ex:35},
  {k:"Insulin", label:"Insulin serum 2 jam", unit:"mu U/ml", ph:"80", ex:""},
  {k:"BMI", label:"BMI", unit:"kg/m²", ph:"28", ex:33.6},
  {k:"DiabetesPedigreeFunction", label:"Pedigree diabetes", unit:"skor", ph:"0.40", ex:0.627},
  {k:"Age", label:"Usia", unit:"tahun", ph:"30", ex:50, nomissing:true},
];
const SHORT = {Pregnancies:"Kehamilan",Glucose:"Glukosa",BloodPressure:"Tekanan darah",SkinThickness:"Lipatan kulit",
  Insulin:"Insulin",BMI:"BMI",DiabetesPedigreeFunction:"Pedigree",Age:"Usia"};
const $ = s => document.querySelector(s);
let MODEL;

$("#fields").innerHTML = FIELDS.map(f => `
  <div class="field"><label for="${f.k}">${f.label}<small>${f.unit}</small></label>
  <input id="${f.k}" name="${f.k}" type="number" step="any" min="0" inputmode="decimal" placeholder="${f.ph}"></div>`).join("");

fetch("model.json").then(r => r.json()).then(m => { MODEL = m; renderMetrics(m); })
  .catch(() => { $("#note").textContent = "Gagal memuat model.json. Jalankan lewat server (GitHub Pages atau Live Server)."; });

function renderMetrics(m){
  const x = m.metrics, items = [["Akurasi",x.accuracy],["Presisi",x.precision],["Recall",x.recall],["F1-score",x.f1],["ROC-AUC",x.roc_auc]];
  $("#metrics").innerHTML = items.map(([l,v]) => `<div class="metric"><b>${v.toFixed(3)}</b><span>${l}</span></div>`).join("");
}

function predict(raw){
  const z = [], contrib = [];
  MODEL.features.forEach((name, i) => {
    const f = FIELDS[i];
    let v = raw[name];
    // nilai kosong, atau 0 pada kolom yang 0-nya berarti data hilang, diganti median (sama seperti saat pelatihan)
    if (v === "" || v === null || isNaN(v) || (v === 0 && !f.nomissing)) v = MODEL.median[i];
    const s = (v - MODEL.mean[i]) / MODEL.scale[i];
    z.push(s); contrib.push({name, c: MODEL.coef[i] * s});
  });
  const logit = MODEL.intercept + contrib.reduce((a, b) => a + b.c, 0);
  return {p: 1 / (1 + Math.exp(-logit)), contrib};
}

$("#form").addEventListener("submit", e => {
  e.preventDefault();
  if (!MODEL) return;
  const raw = {};
  FIELDS.forEach(f => { const el = document.getElementById(f.k); raw[f.k] = el.value === "" ? "" : parseFloat(el.value); });
  const {p, contrib} = predict(raw);
  const high = p >= 0.5, pct = Math.round(p * 100);
  $("#empty").hidden = true; $("#out").hidden = false;
  $("#out").className = "out " + (high ? "high" : "low");
  const ring = $("#ring");
  ring.style.setProperty("--c", high ? "var(--bad)" : "var(--good)");
  ring.style.setProperty("--p", pct);
  $("#pct").textContent = pct + "%";
  $("#verdict").textContent = high ? "Risiko tinggi" : "Risiko rendah";
  $("#note").textContent = high
    ? "Pola data ini mirip pasien yang didiagnosis diabetes. Disarankan memeriksakan diri ke tenaga kesehatan."
    : "Pola data ini lebih mirip pasien tanpa diabetes. Tetap jaga pola hidup sehat.";
  const max = Math.max(...contrib.map(x => Math.abs(x.c)), 0.01);
  $("#factors").innerHTML = contrib.sort((a, b) => Math.abs(b.c) - Math.abs(a.c)).map(x => {
    const w = Math.min(Math.abs(x.c) / max * 50, 50);
    return `<li><span>${SHORT[x.name]}</span><div class="bar"><b class="${x.c >= 0 ? "up" : "down"}" style="width:${w}%"></b></div><em>${x.c >= 0 ? "+" : ""}${x.c.toFixed(2)}</em></li>`;
  }).join("");
  if (window.innerWidth < 900) $("#result").scrollIntoView({behavior: "smooth", block: "start"});
});

$("#example").addEventListener("click", () => {
  FIELDS.forEach(f => document.getElementById(f.k).value = f.ex);
});
