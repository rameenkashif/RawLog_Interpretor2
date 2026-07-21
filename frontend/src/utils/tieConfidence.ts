/**
 * tieConfidence.ts
 * ----------------
 * Shared low-confidence threshold for well-to-seismic tie / synthetic
 * seismogram results, matching the backend's
 * dashboard_upload_service.TIE_LOW_CONFIDENCE_THRESHOLD (0.3). Computed
 * client-side from fields every tie/synthetic response already carries
 * (correlation, boundary_pinned) rather than fetched separately, so
 * there's one source of truth and no extra request.
 */
export const TIE_LOW_CONFIDENCE_THRESHOLD = 0.3;

export function isLowConfidenceTie(
  correlation: number | null | undefined,
  boundaryPinned: boolean | null | undefined,
): boolean {
  if (boundaryPinned) return true;
  if (correlation === null || correlation === undefined) return false;
  return correlation < TIE_LOW_CONFIDENCE_THRESHOLD;
}
