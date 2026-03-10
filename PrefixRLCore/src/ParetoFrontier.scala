package PrefixRLCore

import PrefixAdderLib.DependentTopology
import PrefixAdderLib.TopologyStats

import scala.collection.mutable

final case class EvaluatedTopology(
  id:           String,
  episode:      Int,
  topology:     DependentTopology,
  ppa:          PpaTuple,
  topologyPath: String,
  rtlDir:       String,
  metricsPath:  Option[String],
  duplicate:    Boolean
) {
  lazy val stats: TopologyStats = topology.stats

  def toJson: ujson.Value = ujson.Obj(
    "id" -> id,
    "episode" -> episode,
    "duplicate" -> duplicate,
    "topology_path" -> topologyPath,
    "rtl_dir" -> rtlDir,
    "metrics_path" -> ujson.Str(metricsPath.getOrElse("")),
    "fingerprint" -> stats.fingerprint,
    "stats" -> ujson.Obj(
      "unique_internal_nodes" -> stats.uniqueInternalNodes,
      "total_internal_nodes" -> stats.totalInternalNodes,
      "max_depth" -> stats.maxDepth,
      "average_depth" -> stats.averageDepth,
      "expression" -> topology.outputs.last.pretty
    ),
    "ppa" -> ppa.toJson,
    "topology" -> topology.toJson
  )
}

final case class FrontierPreview(
  isDominated:           Boolean,
  dominatesCount:        Int,
  minDistanceToFrontier: Double
)

final case class FrontierUpdate(
  addedToFrontier: Boolean,
  removedCount:    Int,
  frontierSize:    Int,
  duplicate:       Boolean
)

final class ParetoFrontier {
  private val entries = mutable.ArrayBuffer.empty[EvaluatedTopology]

  def snapshot: Vector[EvaluatedTopology] = entries.toVector.sortBy(_.id)

  def preview(candidate: PpaTuple, normalizer: RunningObjectiveNormalizer): FrontierPreview = {
    val dominated = entries.exists(existing => existing.ppa == candidate || existing.ppa.dominates(candidate))
    val dominatesCount = entries.count(existing => candidate.dominates(existing.ppa))
    val candidateNorm = normalizer.normalize(candidate)
    val minDistance = if (entries.isEmpty) {
      1.0
    } else {
      entries
        .map(existing => candidateNorm.distanceTo(normalizer.normalize(existing.ppa)))
        .min
    }
    FrontierPreview(
      isDominated = dominated,
      dominatesCount = dominatesCount,
      minDistanceToFrontier = minDistance
    )
  }

  def update(candidate: EvaluatedTopology): FrontierUpdate = {
    val equivalent = entries.exists(_.ppa == candidate.ppa)
    val dominated = entries.exists(_.ppa.dominates(candidate.ppa))

    if (equivalent || dominated) {
      FrontierUpdate(
        addedToFrontier = false,
        removedCount = 0,
        frontierSize = entries.size,
        duplicate = equivalent
      )
    } else {
      val survivors = entries.filterNot(existing => candidate.ppa.dominates(existing.ppa))
      val removedCount = entries.size - survivors.size
      entries.clear()
      entries ++= survivors
      entries += candidate
      FrontierUpdate(
        addedToFrontier = true,
        removedCount = removedCount,
        frontierSize = entries.size,
        duplicate = false
      )
    }
  }
}
