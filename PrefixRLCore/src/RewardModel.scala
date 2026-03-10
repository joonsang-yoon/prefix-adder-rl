package PrefixRLCore

final case class RewardBreakdown(
  normalized:       NormalizedObjectives,
  aggregateScore:   Double,
  frontierBonus:    Double,
  spreadBonus:      Double,
  duplicatePenalty: Double,
  reward:           Double
) {
  def toJson: ujson.Value = ujson.Obj(
    "normalized" -> normalized.toJson,
    "aggregate_score" -> aggregateScore,
    "frontier_bonus" -> frontierBonus,
    "spread_bonus" -> spreadBonus,
    "duplicate_penalty" -> duplicatePenalty,
    "reward" -> reward
  )
}

object RewardModel {
  def score(
    candidate:  PpaTuple,
    preview:    FrontierPreview,
    normalizer: RunningObjectiveNormalizer,
    duplicate:  Boolean
  ): RewardBreakdown = {
    val normalized = normalizer.normalize(candidate)
    val aggregateScore = 1.0 - normalized.average
    val frontierBonus = if (preview.isDominated) {
      -0.35
    } else {
      0.6 + 0.15 * preview.dominatesCount.toDouble
    }
    val spreadBonus = 0.1 * math.min(1.0, preview.minDistanceToFrontier)
    val duplicatePenalty = if (duplicate) -0.25 else 0.0
    val reward = aggregateScore + frontierBonus + spreadBonus + duplicatePenalty
    RewardBreakdown(
      normalized = normalized,
      aggregateScore = aggregateScore,
      frontierBonus = frontierBonus,
      spreadBonus = spreadBonus,
      duplicatePenalty = duplicatePenalty,
      reward = reward
    )
  }
}
