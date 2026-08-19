const samplesRoot = document.querySelector("#samples");
const colorbarImage = document.querySelector("#colorbarImage");
const colorbarCaption = document.querySelector("#colorbarCaption");
const attributionRows = document.querySelector("#attributionRows");
const realExpDemoRoot = document.querySelector("#realExpDemo");
const dataUrl = "assets/data/project-data.json?v=20260819-demo-selection";

function fmtScore(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "N/A";
}

function fmtPercentile(value) {
  return Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: 1 }) : "N/A";
}

function trackNode(track, groundTruth) {
  const item = document.createElement("div");
  item.className = "track";
  const isCorrect = track.prediction && groundTruth
    ? track.prediction === groundTruth
    : track.correct;
  const status =
    typeof isCorrect === "boolean"
      ? `<span class="track__status ${isCorrect ? "is-correct" : "is-wrong"}">${
          isCorrect ? "Correct" : "Wrong"
        }</span>`
      : "";
  const prediction = track.prediction
    ? `<span class="track__prediction">Prediction: ${track.prediction}</span>`
    : "";
  const scores =
    typeof track.cap_score === "number" || typeof track.saj_score === "number"
      ? `<span class="track__score">CLAP Score: ${fmtScore(track.cap_score)} | SAJ Score: ${fmtScore(track.saj_score)}</span>`
      : "";
  item.innerHTML = `
    <img class="spectrogram" src="${track.spectrogram}" alt="${track.label} spectrogram" loading="lazy" />
    <div class="track__head">
      <span class="track__label">${track.label}</span>
      ${status}
    </div>
    ${prediction}
    ${scores}
    <audio controls preload="none" src="${track.audio}"></audio>
  `;
  return item;
}

function sampleNode(sample, index) {
  const section = document.createElement("article");
  section.className = "sample";

  const refs = sample.reference_tracks.map(trackNode);
  const methods = sample.method_tracks.map(track => trackNode(track, sample.ground_truth));

  const refGrid = document.createElement("div");
  refGrid.className = "tracks tracks--references";
  refGrid.replaceChildren(...refs);

  const methodGrid = document.createElement("div");
  methodGrid.className = "tracks tracks--methods";
  methodGrid.replaceChildren(...methods);

  section.innerHTML = `
    <header class="sample__head">
      <h2>${String(index + 1).padStart(2, "0")}. ${sample.class}</h2>
      <div class="sample__meta">
        <span class="pill">Robot: ${sample.robot}</span>
        <span class="pill">SNR: ${sample.snr_db} dB</span>
        <span class="pill">Ground Truth: ${sample.ground_truth || sample.class}</span>
      </div>
    </header>
    <h3>References</h3>
  `;
  section.append(refGrid);
  section.insertAdjacentHTML("beforeend", "<h3>Methods</h3>");
  section.append(methodGrid);
  return section;
}

function projectPageSectionNode(section) {
  const wrapper = document.createElement("section");
  wrapper.className = "project-page-section";
  if (section.title) {
    const title = document.createElement("h3");
    title.textContent = section.title;
    wrapper.append(title);
  }
  wrapper.append(...section.samples.map(sampleNode));
  return wrapper;
}

function demoGroupNode(title, sections) {
  const wrapper = document.createElement("section");
  wrapper.className = "demo-group";
  const heading = document.createElement("h2");
  heading.textContent = title;
  wrapper.append(heading, ...sections.map(projectPageSectionNode));
  return wrapper;
}

function demoSectionsNode(sections) {
  const separationTitles = new Set(["Speech Separation", "Music Separation"]);
  const separationSections = sections.filter(section => separationTitles.has(section.title));
  const classificationSections = sections.filter(section => !separationTitles.has(section.title));
  const groups = [];
  if (separationSections.length) {
    groups.push(demoGroupNode("Demo: Ego-Noise Separation", separationSections));
  }
  if (classificationSections.length) {
    groups.push(demoGroupNode("Demo: Anomalous Sound Classification After Ego-Noise Separation", classificationSections));
  }
  return groups;
}

function realDemoVideoNode(src, label, poster) {
  const figure = document.createElement("figure");
  figure.className = "real-demo__video";
  const video = document.createElement("video");
  video.controls = true;
  video.preload = "none";
  video.src = src;
  video.playsInline = true;
  if (poster) {
    video.poster = poster;
  }
  video.setAttribute("aria-label", label);
  const caption = document.createElement("figcaption");
  caption.textContent = label;
  figure.append(video, caption);
  return figure;
}

function realDemoSampleNode(sample, index) {
  const item = document.createElement("article");
  item.className = "real-demo__sample";

  const label = sample.environment_label || "";
  const sound = sample.sound_label || "";
  const badgeClass = label.toLowerCase() === "anomaly" ? "is-anomaly" : "is-normal";
  const displayNumber = sample.display_number ?? index + 1;
  const header = document.createElement("header");
  header.className = "real-demo__head";
  header.innerHTML = `
    <h3>${String(displayNumber).padStart(2, "0")}. ${sound}</h3>
    <span class="real-demo__badge ${badgeClass}">Environmental sound: ${label}</span>
  `;

  const grid = document.createElement("div");
  grid.className = "real-demo__grid";
  grid.append(
    realDemoVideoNode(sample.raw_video, "Recorded Audio", sample.poster),
    realDemoVideoNode(
      sample.denoised_video,
      "Ego-Noise Separated Audio by Proposed Method",
      sample.poster
    )
  );

  item.append(header, grid);
  return item;
}

function realDemoNode(realExpDemo) {
  const samples = realExpDemo?.samples || [];
  if (!samples.length) return [];
  return samples.map(realDemoSampleNode);
}

function attributionNode(item) {
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${item.sample}</td>
    <td>${item.category}</td>
    <td>${item.dataset}</td>
    <td>${item.license}</td>
  `;
  return row;
}

fetch(dataUrl)
  .then(response => response.json())
  .then(data => {
    colorbarImage.src = data.spectrogram.colorbar;
    const cut = data.spectrogram.percentile_cut;
    colorbarCaption.textContent = Number.isFinite(cut)
      ? `${fmtPercentile(cut)}% tile to ${fmtPercentile(100 - cut)}% tile`
      : `${data.spectrogram.db_min} to ${data.spectrogram.db_max} dBFS`;
    const sections = data.sections || [{ title: "Audio Examples", samples: data.samples || [] }];
    realExpDemoRoot.replaceChildren(...realDemoNode(data.real_exp_demo));
    samplesRoot.replaceChildren(...demoSectionsNode(sections));
    attributionRows.replaceChildren(...(data.attributions || []).map(attributionNode));
  });
