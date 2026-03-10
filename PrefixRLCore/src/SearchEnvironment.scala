package PrefixRLCore

import PrefixAdderLib.DependentTopology

import scala.collection.mutable

final case class SearchResult(
  evaluated:   EvaluatedTopology,
  preview:     FrontierPreview,
  update:      FrontierUpdate,
  reward:      RewardBreakdown,
  cacheHit:    Boolean,
  stateBefore: SearchPolicyState,
  stateAfter:  SearchPolicyState
) {
  def toJson: ujson.Value = ujson.Obj(
    "cache_hit" -> cacheHit,
    "search_state_before" -> stateBefore.toJson,
    "search_state_after" -> stateAfter.toJson,
    "preview" -> ujson.Obj(
      "is_dominated" -> preview.isDominated,
      "dominates_count" -> preview.dominatesCount,
      "min_distance_to_frontier" -> preview.minDistanceToFrontier
    ),
    "update" -> ujson.Obj(
      "added_to_frontier" -> update.addedToFrontier,
      "removed_count" -> update.removedCount,
      "frontier_size" -> update.frontierSize,
      "duplicate" -> update.duplicate
    ),
    "reward" -> reward.toJson,
    "evaluated" -> evaluated.toJson
  )
}

final class SearchEnvironment(
  evaluator: PhysicalEvaluator,
  logRoot:   os.Path
) {
  private val frontier = new ParetoFrontier
  private val normalizer = new RunningObjectiveNormalizer
  private val cache = mutable.HashMap.empty[String, EvaluatedTopology]

  private val episodesDir = logRoot / "episodes"
  private val frontierDir = logRoot / "frontier"

  private var completedEvaluations = 0
  private var duplicateCount = 0
  private var lastReward = 0.0
  private var lastCacheHit = false

  os.makeDir.all(episodesDir)
  os.makeDir.all(frontierDir)

  def policyState: SearchPolicyState = SearchPolicyState.from(
    frontier = frontier.snapshot,
    normalizer = normalizer,
    completedEvaluations = completedEvaluations,
    cacheSize = cache.size,
    duplicateCount = duplicateCount,
    lastReward = lastReward,
    lastCacheHit = lastCacheHit
  )

  def evaluate(episode: Int, topology: DependentTopology): SearchResult = {
    val stateBefore = policyState
    val fingerprint = topology.stats.fingerprint
    val cached = cache.get(fingerprint)

    val evaluated = cached match {
      case Some(existing) =>
        existing.copy(
          id = s"${episode}_${fingerprint}",
          episode = episode,
          duplicate = true
        )
      case None =>
        val physical = evaluator.evaluate(episode, topology)
        val result = EvaluatedTopology(
          id = s"${episode}_${fingerprint}",
          episode = episode,
          topology = topology,
          ppa = physical.ppa,
          topologyPath = physical.topologyPath.toString,
          rtlDir = physical.rtlDir.toString,
          metricsPath = physical.metricsPath.map(_.toString),
          duplicate = false
        )
        cache.update(fingerprint, result)
        result
    }

    normalizer.observe(evaluated.ppa)

    val preview = frontier.preview(evaluated.ppa, normalizer)
    val update = frontier.update(evaluated)

    val reward = RewardModel.score(
      candidate = evaluated.ppa,
      preview = preview,
      normalizer = normalizer,
      duplicate = cached.isDefined || update.duplicate
    )

    completedEvaluations += 1
    if (cached.isDefined || update.duplicate) {
      duplicateCount += 1
    }
    lastReward = reward.reward
    lastCacheHit = cached.isDefined

    val result = SearchResult(
      evaluated = evaluated,
      preview = preview,
      update = update,
      reward = reward,
      cacheHit = cached.isDefined,
      stateBefore = stateBefore,
      stateAfter = policyState
    )

    persistEpisode(result)
    persistFrontier()
    result
  }

  def frontierSnapshot: Vector[EvaluatedTopology] = frontier.snapshot

  private def persistEpisode(result: SearchResult): Unit = {
    val path = episodesDir / f"episode_${result.evaluated.episode}%05d.json"
    os.write.over(path, ujson.write(result.toJson, indent = 2))
  }

  private def persistFrontier(): Unit = {
    val snapshot = ujson.Arr.from(frontier.snapshot.map(_.toJson))
    os.write.over(frontierDir / "frontier.json", ujson.write(snapshot, indent = 2))
  }
}
