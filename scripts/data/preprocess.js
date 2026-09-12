// scripts/data/preprocess.js
// Preprocessing pipeline for USA Housing dataset:
// 1. Parses raw CSV
// 2. Engineers the FastSale binary label using dataset medians
// 3. Computes standardization parameters (mean, stdDev) and slider ranges (min, max)
// 4. Performs reproducible 80/20 train/test split with fixed seed
// 5. Outputs scripts/data/housing-data.json

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Mulberry32 deterministic PRNG for reproducible shuffling
export function mulberry32(seed) {
  return function() {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Manual CSV parser (avoids external libraries)
export function parseCSV(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length === 0) return [];
  
  function parseLine(line) {
    const fields = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        fields.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    fields.push(current.trim());
    return fields;
  }

  const headers = parseLine(lines[0]);
  const records = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const values = parseLine(line);
    if (values.length === headers.length) {
      const record = {};
      headers.forEach((h, idx) => {
        record[h] = values[idx];
      });
      records.push(record);
    }
  }
  return records;
}

export function calculateStats(numbers) {
  const n = numbers.length;
  if (n === 0) return { mean: 0, stdDev: 1, min: 0, max: 0, median: 0 };

  const sorted = [...numbers].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];

  const sum = numbers.reduce((acc, val) => acc + val, 0);
  const mean = sum / n;

  const variance = numbers.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
  const stdDev = Math.sqrt(variance) || 1;

  return { mean, stdDev, min, max, median };
}

export function preprocess() {
  const csvPath = path.resolve(__dirname, '../../dataset/USA_Housing.csv');
  const outPath = path.resolve(__dirname, 'housing-data.json');

  if (!fs.existsSync(csvPath)) {
    console.error(`CSV file not found at ${csvPath}`);
    process.exit(1);
  }

  console.log(`Reading CSV from ${csvPath}...`);
  const rawCSV = fs.readFileSync(csvPath, 'utf-8');
  const records = parseCSV(rawCSV);
  console.log(`Parsed ${records.length} records.`);

  const featureNames = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population'
  ];

  // Feature extraction
  const rawFeatures = [];
  const rawPrices = [];
  const pricePerRoomList = [];
  const houseAgeList = [];

  for (const row of records) {
    const income = parseFloat(row['Avg. Area Income']);
    const age = parseFloat(row['Avg. Area House Age']);
    const rooms = parseFloat(row['Avg. Area Number of Rooms']);
    const bedrooms = parseFloat(row['Avg. Area Number of Bedrooms']);
    const population = parseFloat(row['Area Population']);
    const price = parseFloat(row['Price']);

    rawFeatures.push([income, age, rooms, bedrooms, population]);
    rawPrices.push(price);
    pricePerRoomList.push(price / rooms);
    houseAgeList.push(age);
  }

  // FastSale label engineering heuristics
  const pprStats = calculateStats(pricePerRoomList);
  const ageStats = calculateStats(houseAgeList);
  const medianPPR = pprStats.median;
  const medianAge = ageStats.median;

  console.log(`Median PricePerRoom: $${medianPPR.toFixed(2)}`);
  console.log(`Median House Age: ${medianAge.toFixed(2)} years`);

  const rawFastSales = [];
  for (let i = 0; i < records.length; i++) {
    const isFast = (pricePerRoomList[i] < medianPPR && houseAgeList[i] < medianAge) ? 1 : 0;
    rawFastSales.push(isFast);
  }

  const fastCount = rawFastSales.reduce((a, b) => a + b, 0);
  console.log(`FastSale positive class count: ${fastCount} / ${records.length} (${(fastCount / records.length * 100).toFixed(2)}%)`);

  // Compute statistics for each feature
  const featureStats = featureNames.map((name, idx) => {
    const vals = rawFeatures.map(f => f[idx]);
    return {
      name,
      ...calculateStats(vals)
    };
  });

  const priceStats = calculateStats(rawPrices);

  // Train/Test Split (80/20) with fixed seed for reproducibility
  const indices = Array.from({ length: records.length }, (_, i) => i);
  const rng = mulberry32(42);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const splitIdx = Math.floor(records.length * 0.8);
  const trainIndices = indices.slice(0, splitIdx);
  const testIndices = indices.slice(splitIdx);

  console.log(`Train set: ${trainIndices.length} samples, Test set: ${testIndices.length} samples.`);

  function standardizeVector(vec) {
    return vec.map((val, j) => (val - featureStats[j].mean) / featureStats[j].stdDev);
  }

  const trainData = {
    X: trainIndices.map(i => standardizeVector(rawFeatures[i])),
    X_raw: trainIndices.map(i => rawFeatures[i]),
    y_price: trainIndices.map(i => rawPrices[i]),
    y_price_norm: trainIndices.map(i => (rawPrices[i] - priceStats.mean) / priceStats.stdDev),
    y_fast: trainIndices.map(i => rawFastSales[i])
  };

  const testData = {
    X: testIndices.map(i => standardizeVector(rawFeatures[i])),
    X_raw: testIndices.map(i => rawFeatures[i]),
    y_price: testIndices.map(i => rawPrices[i]),
    y_price_norm: testIndices.map(i => (rawPrices[i] - priceStats.mean) / priceStats.stdDev),
    y_fast: testIndices.map(i => rawFastSales[i])
  };

  const outputPayload = {
    metadata: {
      totalSamples: records.length,
      trainCount: trainIndices.length,
      testCount: testIndices.length,
      featureNames,
      featureStats,
      priceStats,
      fastSaleHeuristic: {
        medianPricePerRoom: medianPPR,
        medianHouseAge: medianAge,
        positiveCount: fastCount,
        positiveRatio: fastCount / records.length,
        description: 'FastSale is labeled 1 if PricePerRoom is below median ($' + medianPPR.toFixed(0) + ') AND House Age is below median (' + medianAge.toFixed(1) + ' yrs). Synthetic heuristic proxy.'
      }
    },
    train: trainData,
    test: testData
  };

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(outputPayload), 'utf-8');
  console.log(`Successfully generated dataset JSON at ${outPath} (${(fs.statSync(outPath).size / 1024).toFixed(1)} KB)`);
}

// Run directly if invoked as script
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  preprocess();
}
