// Bound SVG work while retaining both ends and the extrema of each time bucket.
// Full-resolution history stays available from the metrics API.
export function sampleMetricPoints(points, budget = 1200) {
  if (points.length <= budget) return points;
  const buckets = Math.max(1, Math.floor((budget - 2) / 2));
  const result = [points[0]];
  const width = (points.length - 2) / buckets;
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const start = 1 + Math.floor(bucket * width);
    const end = Math.min(points.length - 1, 1 + Math.floor((bucket + 1) * width));
    let low = start;
    let high = start;
    for (let index = start + 1; index < end; index += 1) {
      if (points[index].value < points[low].value) low = index;
      if (points[index].value > points[high].value) high = index;
    }
    if (low === high) result.push(points[low]);
    else if (low < high) result.push(points[low], points[high]);
    else result.push(points[high], points[low]);
  }
  result.push(points[points.length - 1]);
  return result;
}
