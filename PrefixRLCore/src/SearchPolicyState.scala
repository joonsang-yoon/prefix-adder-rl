package PrefixRLCore

final case class SearchPolicyState(
  completedEvaluations: Int,
  frontierSize:         Int,
  cacheSize:            Int,
  duplicateCount:       Int,
  hasObservations:      Boolean,
  hasFrontier:          Boolean,
  duplicateRate:        Double,
  frontierFraction:     Double,
  uniqueFraction:       Double,
  normalizedBest:       NormalizedObjectives,
  normalizedMean:       NormalizedObjectives,
  normalizedWorst:      NormalizedObjectives,
  frontierSpread:       Double,
  lastReward:           Double,
  lastCacheHit:         Boolean
) {
  def toJson: ujson.Value = ujson.Obj(
    "completed_evaluations" -> completedEvaluations,
    "frontier_size" -> frontierSize,
    "cache_size" -> cacheSize,
    "duplicate_count" -> duplicateCount,
    "has_observations" -> hasObservations,
    "has_frontier" -> hasFrontier,
    "duplicate_rate" -> duplicateRate,
    "frontier_fraction" -> frontierFraction,
    "unique_fraction" -> uniqueFraction,
    "normalized_best" -> normalizedBest.toJson,
    "normalized_mean" -> normalizedMean.toJson,
    "normalized_worst" -> normalizedWorst.toJson,
    "frontier_spread" -> frontierSpread,
    "last_reward" -> lastReward,
    "last_cache_hit" -> lastCacheHit
  )
}

object SearchPolicyState {
  private val ZeroObjectives = NormalizedObjectives(0.0, 0.0, 0.0)

  val empty: SearchPolicyState = SearchPolicyState(
    completedEvaluations = 0,
    frontierSize = 0,
    cacheSize = 0,
    duplicateCount = 0,
    hasObservations = false,
    hasFrontier = false,
    duplicateRate = 0.0,
    frontierFraction = 0.0,
    uniqueFraction = 0.0,
    normalizedBest = ZeroObjectives,
    normalizedMean = ZeroObjectives,
    normalizedWorst = ZeroObjectives,
    frontierSpread = 0.0,
    lastReward = 0.0,
    lastCacheHit = false
  )

  def from(
    frontier:             Vector[EvaluatedTopology],
    normalizer:           RunningObjectiveNormalizer,
    completedEvaluations: Int,
    cacheSize:            Int,
    duplicateCount:       Int,
    lastReward:           Double,
    lastCacheHit:         Boolean
  ): SearchPolicyState = {
    val normalized = frontier.map(entry => normalizer.normalize(entry.ppa))
    val best = reduceObjectives(normalized)(math.min)
    val mean = averageObjectives(normalized)
    val worst = reduceObjectives(normalized)(math.max)
    val spread = averagePairwiseDistance(normalized)
    val completedDenom = math.max(1, completedEvaluations).toDouble

    SearchPolicyState(
      completedEvaluations = completedEvaluations,
      frontierSize = frontier.length,
      cacheSize = cacheSize,
      duplicateCount = duplicateCount,
      hasObservations = completedEvaluations > 0,
      hasFrontier = frontier.nonEmpty,
      duplicateRate = duplicateCount.toDouble / completedDenom,
      frontierFraction = frontier.length.toDouble / completedDenom,
      uniqueFraction = cacheSize.toDouble / completedDenom,
      normalizedBest = best,
      normalizedMean = mean,
      normalizedWorst = worst,
      frontierSpread = spread,
      lastReward = lastReward,
      lastCacheHit = lastCacheHit
    )
  }

  private def reduceObjectives(
    values: Vector[NormalizedObjectives]
  )(
    op: (Double, Double) => Double
  ): NormalizedObjectives = {
    if (values.isEmpty) {
      ZeroObjectives
    } else {
      values.tail.foldLeft(values.head) { case (acc, value) =>
        NormalizedObjectives(
          power = op(acc.power, value.power),
          delay = op(acc.delay, value.delay),
          area = op(acc.area, value.area)
        )
      }
    }
  }

  private def averageObjectives(values: Vector[NormalizedObjectives]): NormalizedObjectives = {
    if (values.isEmpty) {
      ZeroObjectives
    } else {
      val count = values.length.toDouble
      NormalizedObjectives(
        power = values.map(_.power).sum / count,
        delay = values.map(_.delay).sum / count,
        area = values.map(_.area).sum / count
      )
    }
  }

  private def averagePairwiseDistance(values: Vector[NormalizedObjectives]): Double = {
    if (values.length < 2) {
      0.0
    } else {
      var distanceSum = 0.0
      var pairCount = 0
      var i = 0
      while (i < values.length) {
        var j = i + 1
        while (j < values.length) {
          distanceSum += values(i).distanceTo(values(j))
          pairCount += 1
          j += 1
        }
        i += 1
      }
      distanceSum / pairCount.toDouble
    }
  }
}
